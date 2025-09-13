# RoBERTa Implementation for SWINBERT

This document describes the implementation of RoBERTa as a replacement for BERT in the SWINBERT video captioning model.

## Overview

The implementation follows the existing BERT pattern but replaces the vanilla BERT multimodal encoder with RoBERTa to create a more powerful architecture. RoBERTa is known for its improved training methodology and better performance on various NLP tasks.

## Key Differences from BERT

1. **Tokenization**: RoBERTa uses different special tokens:
   - `<s>` instead of `[CLS]`
   - `</s>` instead of `[SEP]`
   - `<pad>` instead of `[PAD]`
   - `<unk>` instead of `[UNK]`
   - `<mask>` instead of `[MASK]`

2. **Architecture**: RoBERTa removes the Next Sentence Prediction (NSP) task and uses dynamic masking instead of static masking.

3. **Training**: RoBERTa uses larger batches and longer sequences for pre-training.

## Files Created

### Core RoBERTa Implementation
- `src/layers/roberta/__init__.py` - Module initialization
- `src/layers/roberta/file_utils.py` - File utilities for RoBERTa
- `src/layers/roberta/tokenization_utils.py` - Tokenization utilities
- `src/layers/roberta/tokenization_roberta.py` - RoBERTa tokenizer
- `src/layers/roberta/modeling_roberta.py` - Core RoBERTa model classes

### Model Loading and Training
- `src/modeling/load_roberta.py` - RoBERTa model loading function
- `src/modeling/video_captioning_e2e_vid_swin_roberta.py` - RoBERTa version of VideoTransformer
- `src/tasks/run_caption_VidSwinRoBERTa.py` - Training script for RoBERTa

### Configuration
- `src/configs/VidSwinRoBERTa/msrvtt_8frm_default.json` - MSRVTT configuration for RoBERTa

## Usage

### Training with RoBERTa

```bash
python src/tasks/run_caption_VidSwinRoBERTa.py \
    --config src/configs/VidSwinRoBERTa/msrvtt_8frm_default.json \
    --train_yaml MSRVTT-v2/train_32frames.yaml \
    --val_yaml MSRVTT-v2/val_32frames.yaml \
    --per_gpu_train_batch_size 6 \
    --num_train_epochs 15 \
    --learning_rate 0.0003 \
    --max_num_frames 32 \
    --pretrained_2d 0 \
    --backbone_coef_lr 0.05 \
    --mask_prob 0.5 \
    --max_masked_token 45 \
    --zero_opt_stage 1 \
    --mixed_precision_method deepspeed \
    --deepspeed_fp16 \
    --gradient_accumulation_steps 4 \
    --learn_mask_enabled \
    --loss_sparse_w 0.5 \
    --output_dir ./output_roberta
```

### Key Configuration Changes

1. **Model Path**: Change `model_name_or_path` to point to RoBERTa model:
   ```json
   "model_name_or_path": "models/captioning/roberta-base/"
   ```

2. **Output Directory**: Use separate output directory for RoBERTa:
   ```json
   "output_dir": "output/msrvtt_8frm_roberta_default"
   ```

## Model Architecture

The RoBERTa implementation maintains the same overall architecture as SWINBERT:

1. **Video Swin Transformer**: Extracts spatial-temporal representations from raw video frames
2. **RoBERTa Multimodal Encoder**: Processes video tokens and text tokens using RoBERTa architecture
3. **Sparse Attention Mask**: Learnable attention mask for reducing redundancy in video tokens
4. **Masked Language Modeling**: Training objective for caption generation

## Benefits of RoBERTa

1. **Better Pre-training**: RoBERTa's improved training methodology leads to better language understanding
2. **Dynamic Masking**: More robust to overfitting compared to BERT's static masking
3. **Larger Vocabulary**: Better handling of subword tokenization
4. **Improved Performance**: Generally better results on downstream tasks

## Compatibility

The RoBERTa implementation is designed to be a drop-in replacement for BERT:
- Same input/output interfaces
- Compatible with existing data loaders
- Same training and evaluation scripts (with updated imports)
- Maintains the sparse attention mechanism

## Future Enhancements

1. **Large RoBERTa Models**: Support for roberta-large and other variants
2. **Domain-Specific Pre-training**: Pre-training on video captioning datasets
3. **Multi-lingual Support**: Integration with multilingual RoBERTa models
4. **Efficient Training**: Optimizations for large-scale training

## Notes

- The original BERT implementation remains intact for comparison
- RoBERTa models should be downloaded separately (not included in this repository)
- Vocabulary files need to be compatible with the RoBERTa tokenizer
- Performance improvements may vary depending on the specific dataset and task
