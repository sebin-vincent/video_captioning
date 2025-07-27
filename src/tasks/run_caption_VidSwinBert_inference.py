from __future__ import absolute_import, division, print_function
import os
import sys
pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(pythonpath)
sys.path.insert(0, pythonpath)
import numpy as np
from PIL import Image
import os.path as op
import json
import time
import torch
import torch.distributed as dist
from apex import amp
import deepspeed
import cv2
from src.configs.config import (basic_check_arguments, shared_configs)
from src.datasets.data_utils.video_ops import extract_frames_from_video_path
from src.datasets.data_utils.video_transforms import Compose, Resize, Normalize, CenterCrop
from src.datasets.data_utils.volume_transforms import ClipToTensor
from src.datasets.caption_tensorizer import build_tensorizer
from src.utils.deepspeed import fp32_to_fp16
from src.utils.logger import LOGGER as logger
from src.utils.logger import (TB_LOGGER, RunningMeter, add_log_to_file)
from src.utils.comm import (is_main_process,
                            get_rank, get_world_size, dist_init)
from src.utils.miscellaneous import (mkdir, set_seed, str_to_bool)
from src.modeling.video_captioning_e2e_vid_swin_bert import VideoTransformer
from src.modeling.load_swin import get_swin_model, reload_pretrained_swin
from src.modeling.load_bert import get_bert_model

def save_attention_maps(args, video_path, frames, generated_text, cross_attentions):

    output_dir = op.join(args.eval_model_dir, "visualizations")
    mkdir(output_dir)

    # a frame every 2 seconds
    frame_interval = 2

    num_frames_to_save = args.max_num_frames // frame_interval

    video_frames = []
    for i in range(num_frames_to_save):
        frame = frames[i * frame_interval].numpy()
        frame = np.transpose(frame, (1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame = (frame * std + mean) * 255
        frame = frame.astype(np.uint8)
        video_frames.append(frame)


    # process cross-attentions
    # cross_attentions: a list of 6 tensors, each of shape (1, 12, 786, 786)
    generated_tokens = generated_text.split()

    for i, token in enumerate(generated_tokens):
        # cross_attentions[i] is a tuple of 12 attention heads for the i-th word
        # each head is a tensor of shape (1, 12, 786, 786)

        # average across heads
        avg_head_attention = torch.mean(torch.stack(cross_attentions[i]), dim=0)

        # attention scores for the i-th generated token
        # The first 785 tokens are for the video patches
        attention_scores = avg_head_attention[0, 0, 1:785]

        # reshape to a 2D attention map
        attention_map = attention_scores.reshape(28, 28).cpu().numpy()

        # normalize and resize to frame size
        attention_map = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map))
        attention_map = cv2.resize(attention_map, (args.img_res, args.img_res))

        # apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)

        # overlay heatmap on each frame
        for j, frame in enumerate(video_frames):
            overlaid_frame = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

            # put text on the frame
            text = f"Word: {token}"
            cv2.putText(overlaid_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # save frame
            output_path = op.join(output_dir, f"word_{i}_{token}_frame_{j}.jpg")
            cv2.imwrite(output_path, overlaid_frame)


def _online_video_decode(args, video_path):
    decoder_num_frames = getattr(args, 'max_num_frames', 2)
    frames, _ = extract_frames_from_video_path(
                video_path, target_fps=3, num_frames=decoder_num_frames,
                multi_thread_decode=False, sampling_strategy="uniform",
                safeguard_duration=False, start=None, end=None)
    return frames

def _transforms(args, frames):
    raw_video_crop_list = [
        Resize(args.img_res),
        CenterCrop((args.img_res,args.img_res)),
        ClipToTensor(channel_nb=3),
        Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ]            
    raw_video_prcoess = Compose(raw_video_crop_list)

    frames = frames.numpy()
    frames = np.transpose(frames, (0, 2, 3, 1))
    num_of_frames, height, width, channels = frames.shape

    frame_list = []
    for i in range(args.max_num_frames):
        frame_list.append(Image.fromarray(frames[i]))

    # apply normalization, output tensor (C x T x H x W) in the range [0, 1.0]
    crop_frames = raw_video_prcoess(frame_list)
    # (C x T x H x W) --> (T x C x H x W)
    crop_frames = crop_frames.permute(1, 0, 2, 3)
    return crop_frames 

def inference(args, video_path, model, tokenizer, tensorizer):
    cls_token_id, sep_token_id, pad_token_id, mask_token_id, period_token_id = \
        tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token,
        tokenizer.pad_token, tokenizer.mask_token, '.'])

    model.float()
    model.eval()
    frames = _online_video_decode(args, video_path)
    preproc_frames = _transforms(args, frames)
    data_sample = tensorizer.tensorize_example_e2e('', preproc_frames)
    data_sample = tuple(t.to(args.device) for t in data_sample)
    with torch.no_grad():

        inputs = {'is_decode': True,
            'input_ids': data_sample[0][None,:], 'attention_mask': data_sample[1][None,:],
            'token_type_ids': data_sample[2][None,:], 'img_feats': data_sample[3][None,:],
            'masked_pos': data_sample[4][None,:],
            'do_sample': False,
            'bos_token_id': cls_token_id,
            'pad_token_id': pad_token_id,
            'eos_token_ids': [sep_token_id],
            'mask_token_id': mask_token_id,
            # for adding od labels
            'add_od_labels': args.add_od_labels, 'od_labels_start_posid': args.max_seq_a_length,
            # hyperparameters of beam search
            'max_length': args.max_gen_length,
            'num_beams': args.num_beams,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "length_penalty": args.length_penalty,
            "num_return_sequences": args.num_return_sequences,
            "num_keep_best": args.num_keep_best,
        }
        tic = time.time()

        outputs = model(**inputs)

        time_meter = time.time() - tic
        all_caps = outputs["sequences"]  # batch_size * num_keep_best * max_len
        all_confs = torch.exp(outputs["logprobs"])
        cross_attentions = outputs["cross_attentions"]



        print(f"cross_attentions type: {type(cross_attentions)}") # cross_attentions type: <class 'list'>
        print(f"cross_attentions len: {len(cross_attentions)}")   # cross_attentions len: 6
        print(f"number of heads for cross_attentions for first word: {len(cross_attentions[0])}")  # number of heads for cross_attentions for first word: 12
        print(f"Shape of first head attention scores for first word:{cross_attentions[0][0].shape}")  # Shape of first head attention scores for first word:torch.Size([1, 12, 786, 786])

        for caps, confs in zip(all_caps, all_confs):
            for cap, conf in zip(caps, confs):
                cap_text = tokenizer.decode(cap.tolist(), skip_special_tokens=True)
                print(f"Prediction: {cap_text}")
                print(f"Conf: {conf.item()}")
                if args.save_attention_maps:
                    save_attention_maps(args, video_path, frames, cap_text, cross_attentions)


    print(f"Inference model computing time: {time_meter} seconds")

def check_arguments(args):
    # shared basic checks
    basic_check_arguments(args)
    # additional sanity check:
    args.max_img_seq_length = int((args.max_num_frames/2)*(int(args.img_res)/32)*(int(args.img_res)/32))
    
    if args.freeze_backbone or args.backbone_coef_lr == 0:
        args.backbone_coef_lr = 0
        args.freeze_backbone = True
    
    if 'reload_pretrained_swin' not in args.keys():
        args.reload_pretrained_swin = False

    if not len(args.pretrained_checkpoint) and args.reload_pretrained_swin:
        logger.info("No pretrained_checkpoint to be loaded, disable --reload_pretrained_swin")
        args.reload_pretrained_swin = False

    if args.learn_mask_enabled==True: 
        args.attn_mask_type = 'learn_vid_att'

def update_existing_config_for_inference(args):
    ''' load swinbert args for evaluation and inference 
    '''
    assert args.do_test or args.do_eval
    checkpoint = args.eval_model_dir
    try:
        json_path = op.join(checkpoint, os.pardir, 'log', 'args.json')
        f = open(json_path,'r')
        json_data = json.load(f)

        from easydict import EasyDict
        train_args = EasyDict(json_data)
    except Exception as e:
        train_args = torch.load(op.join(checkpoint, 'training_args.bin'))

    train_args.eval_model_dir = args.eval_model_dir
    train_args.resume_checkpoint = args.eval_model_dir + 'model.bin'
    train_args.model_name_or_path = 'models/captioning/bert-base-uncased/'
    train_args.do_train = False
    train_args.do_eval = True
    train_args.do_test = True
    train_args.val_yaml = args.val_yaml
    train_args.test_video_fname = args.test_video_fname
    return train_args

def get_custom_args(base_config):
    parser = base_config.parser
    parser.add_argument('--max_num_frames', type=int, default=32)
    parser.add_argument('--img_res', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument("--grid_feat", type=str_to_bool, nargs='?', const=True, default=True)
    parser.add_argument("--kinetics", type=str, default='400', help="400 or 600")
    parser.add_argument("--pretrained_2d", type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument("--vidswin_size", type=str, default='base')
    parser.add_argument('--freeze_backbone', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--use_checkpoint', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--backbone_coef_lr', type=float, default=0.001)
    parser.add_argument("--reload_pretrained_swin", type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--learn_mask_enabled', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--loss_sparse_w', type=float, default=0)
    parser.add_argument('--sparse_mask_soft2hard', type=str_to_bool, nargs='?', const=True, default=False)
    parser.add_argument('--transfer_method', type=int, default=-1,
                        help="0: load all SwinBERT pre-trained weights, 1: load only pre-trained sparse mask")
    parser.add_argument('--att_mask_expansion', type=int, default=-1,
                        help="-1: random init, 0: random init and then diag-based copy, 1: interpolation")
    parser.add_argument('--resume_checkpoint', type=str, default='None')
    parser.add_argument('--test_video_fname', type=str, default='None')
    parser.add_argument("--save_attention_maps", type=str_to_bool, nargs='?', const=True, default=False)
    args = base_config.parse_args()
    return args

def main(args):
    args = update_existing_config_for_inference(args)
    # global training_saver
    args.device = torch.device(args.device)
    # Setup CUDA, GPU & distributed training
    dist_init(args)
    check_arguments(args)
    set_seed(args.seed, args.num_gpus)
    fp16_trainning = None
    logger.info(
        "device: {}, n_gpu: {}, rank: {}, "
        "16-bits training: {}".format(
            args.device, args.num_gpus, get_rank(), fp16_trainning))

    if not is_main_process():
        logger.disabled = True

    logger.info(f"Pytorch version is: {torch.__version__}")
    logger.info(f"Cuda version is: {torch.version.cuda}")
    logger.info(f"cuDNN version is : {torch.backends.cudnn.version()}" )

     # Get Video Swin model 
    swin_model = get_swin_model(args)
    # Get BERT and tokenizer 
    bert_model, config, tokenizer = get_bert_model(args)
    # build SwinBERT based on training configs
    vl_transformer = VideoTransformer(args, config, swin_model, bert_model) 
    vl_transformer.freeze_backbone(freeze=args.freeze_backbone)

    # load weights for inference
    logger.info(f"Loading state dict from checkpoint {args.resume_checkpoint}")
    cpu_device = torch.device('cpu')
    pretrained_model = torch.load(args.resume_checkpoint, map_location=cpu_device)

    if isinstance(pretrained_model, dict):
        vl_transformer.load_state_dict(pretrained_model, strict=False)
    else:
        vl_transformer.load_state_dict(pretrained_model.state_dict(), strict=False)

    vl_transformer.to(args.device)
    vl_transformer.eval()

    tensorizer = build_tensorizer(args, tokenizer, is_train=False)
    inference(args, args.test_video_fname, vl_transformer, tokenizer, tensorizer)

if __name__ == "__main__":
    shared_configs.shared_video_captioning_config(cbs=True, scst=True)
    args = get_custom_args(shared_configs)
    main(args)
