# BERT to ALBERT Migration Guide for SwinBERT

This guide explains how to replace BERT with ALBERT in the SwinBERT video captioning model.

## What Has Been Implemented

### 1. ALBERT Layer Implementation (`src/layers/albert/`)
- ✅ **`__init__.py`** - Main ALBERT module exports
- ✅ **`modeling_albert.py`** - Complete ALBERT model implementation with:
  - `AlbertConfig` - Configuration class
  - `AlbertModel` - Base ALBERT model
  - `AlbertForImageCaptioning` - ALBERT adapted for video captioning
  - `AlbertImgModel` - ALBERT with image feature support
  - All necessary ALBERT layers with parameter sharing
- ✅ **`tokenization_albert.py`** - ALBERT tokenizer implementation
- ✅ **`tokenization_utils.py`** - Base tokenizer utilities
- ✅ **`modeling_utils.py`** - Base model utilities
- ✅ **`file_utils.py`** - File handling utilities

### 2. Loading Function (`src/modeling/load_albert.py`)
- ✅ **`get_albert_model()`** - Replaces `get_bert_model()` function
- ✅ Handles ALBERT model loading, configuration, and initialization

### 3. Configuration Files
- ✅ **`src/configs/VidSwinBert/vatex_8frm_albert.json`** - ALBERT training config
- ✅ Migration script **`migrate_to_albert.py`**

## Key Differences: BERT vs ALBERT

| Aspect | BERT | ALBERT |
|--------|------|--------|
| **Parameter Sharing** | Each layer has unique parameters | Layers share parameters across groups |
| **Embedding Size** | Same as hidden size (768) | Smaller embedding (128) projected to hidden size |
| **Model Size** | Larger due to unique layer params | Much smaller due to parameter sharing |
| **Architecture** | 12 unique transformer layers | Shared layers with configurable groups |
| **Memory Usage** | Higher | Significantly lower |
| **Training Speed** | Standard | Faster due to reduced parameters |

## Migration Steps

### Step 1: Run the Migration Script
```bash
python migrate_to_albert.py
```

This script will:
- Create ALBERT model directory structure
- Update import statements in Python files
- Copy and adapt configuration files

### Step 2: Update Training Scripts

Replace BERT imports with ALBERT imports in your training files:

```python
# OLD (BERT)
from src.layers.bert import BertTokenizer, BertConfig, BertForImageCaptioning
from src.modeling.load_bert import get_bert_model

# NEW (ALBERT)
from src.layers.albert import AlbertTokenizer, AlbertConfig, AlbertForImageCaptioning
from src.modeling.load_albert import get_albert_model
```

### Step 3: Update Configuration Files

Update your training configurations to use ALBERT:

```json
{
    "model_name_or_path": "models/captioning/albert-base-v2/",
    "output_dir": "output/vatex_8frm_albert",
    // ... other settings remain the same
}
```

### Step 4: Model Directory Structure

Ensure you have the ALBERT model files:
```
models/captioning/albert-base-v2/
├── config.json          # ALBERT configuration
├── vocab.txt            # Vocabulary file  
├── special_tokens_map.json
└── added_tokens.json
```

### Step 5: Key Code Changes

#### Main Training File Updates
In `src/tasks/run_caption_VidSwinBert.py`:

```python
# Change import
from src.modeling.load_albert import get_albert_model

# Update model loading call
model, config, tokenizer = get_albert_model(args)
```

#### Video Captioning Model Updates
In `src/modeling/video_captioning_e2e_vid_swin_bert.py`:

```python
# The trans_encoder will now use ALBERT instead of BERT
# No changes needed in VideoTransformer class - it works with any transformer encoder
```

## ALBERT-Specific Configuration Parameters

ALBERT introduces several new configuration parameters:

```python
{
    "embedding_size": 128,           # Smaller than hidden_size
    "num_hidden_groups": 1,          # Number of parameter groups
    "inner_group_num": 1,            # Layers per group
    "hidden_act": "gelu_new",        # ALBERT uses gelu_new activation
    "classifier_dropout_prob": 0.1   # Dropout for classification layers
}
```

## Testing the Migration

### Quick Test Script
```python
# Test ALBERT loading
from src.modeling.load_albert import get_albert_model
from argparse import Namespace

# Create test args
args = Namespace(
    model_name_or_path='models/captioning/albert-base-v2/',
    config_name='',
    tokenizer_name='',
    do_lower_case=True,
    drop_out=0.1,
    tie_weights=True,
    freeze_embedding=False,
    label_smoothing=0.0,
    drop_worst_ratio=0.0,
    drop_worst_after=0,
    img_feature_dim=512,
    num_hidden_layers=-1,
    hidden_size=-1,
    num_attention_heads=-1,
    intermediate_size=-1,
    load_partial_weights=False
)

# Test loading
model, config, tokenizer = get_albert_model(args)
print(f"Model loaded successfully! Total parameters: {sum(p.numel() for p in model.parameters())}")
```

### Training Command
```bash
python src/tasks/run_caption_VidSwinBert.py \
    --config src/configs/VidSwinBert/vatex_8frm_albert.json \
    --train_yaml VATEX/train_32frames.yaml \
    --val_yaml VATEX/public_test_32frames.yaml \
    --per_gpu_train_batch_size 6 \
    --per_gpu_eval_batch_size 6 \
    --num_train_epochs 15 \
    --learning_rate 0.0003 \
    --max_num_frames 32 \
    --output_dir ./output_albert
```

## Expected Benefits

1. **Reduced Memory Usage**: ALBERT uses significantly less GPU memory
2. **Faster Training**: Fewer parameters means faster forward/backward passes
3. **Better Generalization**: Parameter sharing can improve generalization
4. **Easier Deployment**: Smaller model size for production deployment

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all BERT imports are updated to ALBERT
2. **Config Errors**: Ensure ALBERT-specific config parameters are set
3. **Model Loading**: Verify the ALBERT model directory structure is correct
4. **Tokenizer Issues**: ALBERT uses the same tokenizer format as BERT

### Performance Comparison

Monitor these metrics when switching to ALBERT:
- GPU memory usage (should decrease)
- Training time per epoch (should decrease)
- Model accuracy (should be comparable or better)
- Inference speed (should improve)

## Next Steps

1. **Download Pre-trained ALBERT**: Get actual ALBERT v2 weights from Hugging Face
2. **Fine-tune**: Train on your video captioning dataset
3. **Evaluate**: Compare performance with original BERT model
4. **Optimize**: Experiment with ALBERT-specific hyperparameters

## Files Modified/Created

### Created Files:
- `src/layers/albert/` (entire directory)
- `src/modeling/load_albert.py`
- `src/configs/VidSwinBert/vatex_8frm_albert.json`
- `migrate_to_albert.py`
- `ALBERT_MIGRATION_GUIDE.md`

### Files to Modify (via migration script):
- `src/tasks/run_caption_VidSwinBert.py`
- `src/tasks/run_caption_VidSwinBert_inference.py`
- `src/modeling/video_captioning_e2e_vid_swin_bert.py`

The migration preserves all video processing functionality while replacing the text encoder with ALBERT's more efficient architecture.
