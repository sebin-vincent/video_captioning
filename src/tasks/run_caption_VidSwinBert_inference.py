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
# Imports for explainability
from src.explainability.attention_mapper import get_bert_text_to_visual_attention, generate_token_visual_explanations # get_swin_patch_importance_scores (not used yet)
from src.explainability.visualization import visualize_token_explanation


# Global flag for mode, can be turned into an arg later
EXTRACT_ATTENTIONS_MODE = True # Set to True to run attention extraction, False for normal inference.

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
    frames_raw_dec = _online_video_decode(args, video_path) # Renamed to avoid clash if used later
    preproc_frames = _transforms(args, frames_raw_dec)
    # data_sample is (input_ids, attention_mask, token_type_ids, img_feats_TCHW, masked_pos, lm_label_ids)
    data_sample = tensorizer.tensorize_example_e2e('', preproc_frames)
    data_sample = tuple(t.to(args.device) for t in data_sample)

    # img_feats for model input needs to be (B, S, C, H, W) as in VideoTransformer init
    # but tensorizer.tensorize_example_e2e returns img_feats as (T,C,H,W)
    # VideoTransformer expects 'img_feats' in kwargs as (B,S,C,H,W)
    # The data_sample[3] is (T,C,H,W). Need to unsqueeze to (1,T,C,H,W)
    # S in VideoTransformer is number of segments, which is T (num_frames) here.

    img_feats_for_model = data_sample[3].unsqueeze(0) # Becomes (1, T, C, H, W)

    with torch.no_grad():
        if EXTRACT_ATTENTIONS_MODE:
            logger.info("Running in attention extraction mode.")
            # Prepare inputs for a single forward pass (is_decode=False)
            inputs = {
                'is_decode': False,
                'input_ids': data_sample[0].unsqueeze(0), # Add batch dim
                'attention_mask': data_sample[1].unsqueeze(0), # Add batch dim
                'token_type_ids': data_sample[2].unsqueeze(0), # Add batch dim
                'img_feats': img_feats_for_model, # Already (1, T, C, H, W)
                'masked_pos': data_sample[4].unsqueeze(0), # Add batch dim
                'masked_ids': data_sample[5].unsqueeze(0)  # Add batch dim
            }

            outputs = model(**inputs)
            # Expected outputs from VideoTransformer.forward (when is_decode=False):
            # Tuple containing:
            # 1. masked_loss (if training) or logits (if eval and not training)
            # 2. class_logits (if training) or potentially nothing if eval (depends on BertForImageCaptioning.encode_forward)
            # Followed by:
            # ?. (optional) bert_hidden_states
            # ?. (optional) bert_attentions
            # ?. (optional) sparsity_loss (if learn_mask_enabled)
            # LAST. swin_attentions
            # The exact indexing requires care based on the model's configuration.

            bert_attentions = None
            swin_attentions = None
            bert_config = model.trans_encoder.bert.config # BertConfig

            # Base elements: loss, logits
            current_idx = 2
            if bert_config.output_hidden_states:
                # bert_hidden_states = outputs[current_idx]
                current_idx +=1

            if bert_config.output_attentions:
                bert_attentions = outputs[current_idx]
                current_idx +=1

            # Check for sparsity loss if learn_mask_enabled
            # model.learn_mask_enabled is not directly accessible here, check args used for model creation
            # The VideoTransformer instance is 'model' here.
            if hasattr(model, 'learn_mask_enabled') and model.learn_mask_enabled:
                # sparsity_loss = outputs[current_idx]
                current_idx += 1

            swin_attentions = outputs[current_idx] if len(outputs) > current_idx else None

            logger.info(f"Raw BERT attentions extracted: {'Yes' if bert_attentions is not None else 'No'}")
            if bert_attentions:
                logger.info(f"  Num layers: {len(bert_attentions)}")
                logger.info(f"  Shape of last layer attentions: {bert_attentions[-1].shape}")

            logger.info(f"Raw Swin attentions extracted: {'Yes' if swin_attentions is not None else 'No'}")
            if swin_attentions:
                logger.info(f"  Num Swin major layers: {len(swin_attentions)}")
                if swin_attentions[0] and isinstance(swin_attentions[0], list) and swin_attentions[0][0] is not None:
                     logger.info(f"  Shape of first block, first layer: {swin_attentions[0][0].shape}")

            # Decode input_ids to get text tokens for explanation
            # input_ids for BERT part already includes [CLS], [SEP]
            # For `get_bert_text_to_visual_attention`, text_len should be this full length.
            # For `generate_token_visual_explanations`, `text_tokens` should be human-readable.

            input_ids_list = inputs['input_ids'][0].tolist()
            processed_tokens_for_bert_att = tokenizer.convert_ids_to_tokens(input_ids_list)
            # For user display, filter out special tokens from the text part
            display_text_tokens = [t for t in processed_tokens_for_bert_att if t not in {tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token}]


            num_visual_tokens_bert = args.max_img_seq_length

            bert_text_to_vis_scores = get_bert_text_to_visual_attention(
                bert_attentions,
                text_len=inputs['input_ids'].shape[1], # Full length of tokens fed to BERT text part
                visual_len=num_visual_tokens_bert,
                batch_idx=0
            )

            # Swin temporal patch embed stride is typically 2 (e.g. patch_size[0]=2 for VideoSwin)
            # Swin spatial patch embed stride is typically 4 (e.g. patch_size[1]=4, patch_size[2]=4 for VideoSwin)
            # Final feature grid spatial resolution for BERT is (img_res/32, img_res/32)
            # Final feature grid temporal resolution for BERT is (max_num_frames / (swin_temporal_patch_embed_stride * any_other_temporal_downsampling))
            # From args.max_img_seq_length formula, seems like (max_num_frames/2) is the temporal dim into the spatial grid part.

            swin_temporal_dim_for_bert = args.max_num_frames // getattr(args, 'swin_patch_size', [2,4,4])[0] # e.g., 32/2 = 16
            # This swin_temporal_dim_for_bert is the D_out in D_out * (H_out/32)^2 = max_img_seq_length
            # The (img_res/32) is already H_patches and W_patches after all Swin stages.

            patches_rc_dim_per_frame = args.img_res // 32 # e.g. 224/32 = 7

            explanations = generate_token_visual_explanations(
                text_tokens=display_text_tokens, # User-friendly tokens
                bert_text_to_visual_attention_scores=bert_text_to_vis_scores, # (text_len_bert, visual_len_bert)
                num_frames=swin_temporal_dim_for_bert,
                patches_per_frame_h=patches_rc_dim_per_frame,
                patches_per_frame_w=patches_rc_dim_per_frame,
                top_k_frames_per_token=1 # For brevity in logs
            )

            if explanations:
                # Ensure output directory exists
                viz_output_dir = op.join(args.output_dir, "visualizations")
                mkdir(viz_output_dir)

                for token_idx, token_exp_data in enumerate(explanations):
                    # Sanitize token string for filename
                    sane_token_str = "".join(c if c.isalnum() else "_" for c in token_exp_data['token_str'])
                    if not sane_token_str: sane_token_str = f"token{token_idx}"

                    output_viz_path = op.join(viz_output_dir, f"exp_tok_{sane_token_str}.mp4")
                    vis_patch_display_size = (args.img_res // patches_rc_dim_per_frame, args.img_res // patches_rc_dim_per_frame)

                    logger.info(f"Visualizing for token: {token_exp_data['token_str']}")
                    visualize_token_explanation(
                        video_path,
                        token_exp_data,
                        output_viz_path,
                        patch_size=vis_patch_display_size
                    )
                    logger.info(f"Saved explanation for token '{token_exp_data['token_str']}' to {output_viz_path}")
            else:
                logger.info("No explanations generated.")

        else: # Original inference path
            inputs = {'is_decode': True,
                'input_ids': data_sample[0].unsqueeze(0),
                'attention_mask': data_sample[1].unsqueeze(0),
                'token_type_ids': data_sample[2].unsqueeze(0),
                'img_feats': img_feats_for_model, # Use the correctly shaped one
                'masked_pos': data_sample[4].unsqueeze(0),
                'do_sample': False,
                'bos_token_id': cls_token_id,
                'pad_token_id': pad_token_id,
                'eos_token_ids': [sep_token_id],
                'mask_token_id': mask_token_id,
                'add_od_labels': args.add_od_labels, 'od_labels_start_posid': args.max_seq_a_length,
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
            outputs = model(**inputs) # Standard generation
            time_meter = time.time() - tic
            all_caps = outputs[0]
            all_confs = torch.exp(outputs[1])

            for caps, confs in zip(all_caps, all_confs):
                for cap, conf in zip(caps, confs):
                    cap = tokenizer.decode(cap.tolist(), skip_special_tokens=True)
                    logger.info(f"Prediction: {cap}")
                    logger.info(f"Conf: {conf.item()}")
            logger.info(f"Inference model computing time: {time_meter} seconds")

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
