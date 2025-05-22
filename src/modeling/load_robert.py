# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.
"""
Load RoBERTa model.
"""
import logging
import torch
from transformers import RobertaConfig
# Ensure RobertaTokenizerModified is correctly imported if it's the intended custom tokenizer
from src.layers.roberta.tokenization_robert import RobertaTokenizerModified
from src.layers.roberta.modeling_roberta import RobertaForImageCaptioning

logger = logging.getLogger(__name__)

def get_roberta_model(args, config_name=None, model_name=None):
    """
    Load RoBERTa model and tokenizer, similar to get_bert_model.
    """
    # Determine config and model names based on args
    effective_config_name = config_name if config_name else args.model_name_or_path
    effective_model_name = model_name if model_name else args.model_name_or_path
    
    # Load RoBERTa configuration
    config = RobertaConfig.from_pretrained(effective_config_name, cache_dir=args.cache_dir)

    # Update config with specific arguments from 'args'
    # This follows the pattern in get_bert_model
    if hasattr(args, 'img_feature_dim') and args.img_feature_dim != -1:
        config.img_feature_dim = args.img_feature_dim
    
    if hasattr(args, 'img_feature_type') and args.img_feature_type != '':
        config.img_feature_type = args.img_feature_type

    if hasattr(args, 'use_img_layernorm') and args.use_img_layernorm != -1: # Assuming -1 means not set
        config.use_img_layernorm = args.use_img_layernorm

    if hasattr(args, 'img_layer_norm_eps') and args.img_layer_norm_eps != -1:
        config.img_layer_norm_eps = args.img_layer_norm_eps
        
    if hasattr(args, 'num_hidden_layers') and args.num_hidden_layers != -1:
        config.num_hidden_layers = args.num_hidden_layers

    # RoBERTa specific config updates from args if any (example)
    # if hasattr(args, 'roberta_specific_param'):
    #    config.roberta_specific_param = args.roberta_specific_param

    # Add output_attentions and output_hidden_states to config if not already present
    # These are often controlled by args in training scripts
    config.output_attentions = getattr(args, 'output_attentions', config.output_attentions)
    config.output_hidden_states = getattr(args, 'output_hidden_states', config.output_hidden_states)
    config.label_smoothing = getattr(args, 'label_smoothing', 0.0) # Example from Bert
    config.drop_worst_ratio = getattr(args, 'drop_worst_ratio', 0.0) # Example from Bert
    config.drop_worst_after = getattr(args, 'drop_worst_after', 0) # Example from Bert


    # Load Tokenizer (RobertaTokenizerModified)
    # effective_tokenizer_name = model_name if model_name else args.model_name_or_path
    # Using effective_model_name for tokenizer for consistency with model loading
    tokenizer = RobertaTokenizerModified.from_pretrained(effective_model_name, cache_dir=args.cache_dir)

    # Load Model (RobertaForImageCaptioning)
    model = RobertaForImageCaptioning.from_pretrained(effective_model_name, config=config, cache_dir=args.cache_dir)

    logger.info("Model: %s", type(model))
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Total Parameters: %d", total_params)
    
    # Handling fine-tuning from a checkpoint (similar to get_bert_model)
    if args.resume_checkpoint and hasattr(args, 'load_ συγκεκριμένα_weights_from_checkpoint') and args.load_ συγκεκριμένα_weights_from_checkpoint:
        logger.info("Loading weights from checkpoint: %s", args.resume_checkpoint)
        checkpoint = torch.load(args.resume_checkpoint, map_location='cpu')
        
        state_dict = checkpoint.get('model', checkpoint) # Check if 'model' key exists

        # Attempt to map weights if loading from a BERT checkpoint or a differently named RoBERTa checkpoint
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('roberta.'): # Standard RoBERTa prefix
                new_state_dict[k] = v
            elif k.startswith('bert.'): # If loading from BERT, map 'bert.' to 'roberta.'
                new_k = k.replace('bert.', 'roberta.', 1)
                logger.info(f"Mapping BERT weight {k} to RoBERTa {new_k}")
                new_state_dict[new_k] = v
            elif k.startswith('cls.'): # Captioning head, usually same name
                 new_state_dict[k] = v
            # Add more specific mappings if the model structure differs significantly
            # e.g., if RobertaForImageCaptioning has `caption_head.` instead of `cls.`
            # For now, assuming `cls.` is consistent or covered by direct load.
            else:
                # If no specific mapping, try to load as is (might be from a compatible RoBERTa checkpoint)
                new_state_dict[k] = v


        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys when loading checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
        
        logger.info(f"Weights loaded from {args.resume_checkpoint} with {len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys.")

    return model, config, tokenizer
