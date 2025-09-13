from __future__ import absolute_import, division, print_function

import os
import sys
pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(pythonpath)
sys.path.insert(0, pythonpath)

import torch
import torch.distributed as dist
from src.configs.config import shared_configs
from src.modeling.video_captioning_e2e_vid_swin_roberta import VideoTransformerRoBERTa
from src.modeling.load_swin import get_swin_model
from src.modeling.load_roberta import get_roberta_model
from src.utils.comm import dist_init
from src.utils.miscellaneous import set_seed, mkdir
from src.utils.logger import LOGGER as logger

def main(args):
    # Setup CUDA, GPU & distributed training
    dist_init(args)
    mkdir(args.output_dir)
    logger.info(f"creating output_dir at: {args.output_dir}")
    set_seed(args.seed, args.num_gpus)
    
    logger.info(f"Pytorch version is: {torch.__version__}")
    logger.info(f"Cuda version is: {torch.version.cuda}")

    # Get Video Swin model 
    swin_model = get_swin_model(args)
    # Get RoBERTa and tokenizer 
    roberta_model, config, tokenizer = get_roberta_model(args)
    # build SwinRoBERTa based on training configs
    vl_transformer = VideoTransformerRoBERTa(args, config, swin_model, roberta_model) 
    vl_transformer.freeze_backbone(freeze=args.freeze_backbone)

    vl_transformer.to(args.device)
    
    logger.info("RoBERTa model initialized successfully!")
    logger.info(f"Model total parameters: {sum(p.numel() for p in vl_transformer.parameters())}")
    
    if args.distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    import torch.multiprocessing

    torch.multiprocessing.set_sharing_strategy('file_system')
    shared_configs.shared_video_captioning_config(cbs=True, scst=True)
    args = shared_configs.parse_args()
    
    # Add RoBERTa-specific arguments
    args.max_num_frames = getattr(args, 'max_num_frames', 32)
    args.img_res = getattr(args, 'img_res', 224)
    args.patch_size = getattr(args, 'patch_size', 32)
    args.grid_feat = getattr(args, 'grid_feat', True)
    args.kinetics = getattr(args, 'kinetics', '400')
    args.vidswin_size = getattr(args, 'vidswin_size', 'base')
    args.freeze_backbone = getattr(args, 'freeze_backbone', False)
    args.use_checkpoint = getattr(args, 'use_checkpoint', False)
    
    main(args)
