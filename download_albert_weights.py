#!/usr/bin/env python3
"""
Script to download and convert ALBERT weights for video captioning
"""

import os
import torch
from transformers import AlbertModel, AlbertConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_and_convert_albert_weights():
    """Download ALBERT weights and convert them to our format."""
    
    # Download original ALBERT model
    logger.info("Downloading ALBERT base v2 model...")
    original_model = AlbertModel.from_pretrained('albert-base-v2')
    original_config = AlbertConfig.from_pretrained('albert-base-v2')
    
    # Save the pytorch model
    model_dir = 'models/captioning/albert-base-v2/'
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model weights
    torch.save(original_model.state_dict(), os.path.join(model_dir, 'pytorch_model.bin'))
    logger.info(f"Saved model weights to {model_dir}pytorch_model.bin")
    
    # Update our config with missing attributes from original
    our_config_path = os.path.join(model_dir, 'config.json')
    
    # Add any missing attributes to our config
    original_config_dict = original_config.to_dict()
    
    # Read our custom config
    import json
    with open(our_config_path, 'r') as f:
        our_config = json.load(f)
    
    # Merge configs (keep our custom attributes, add missing standard ones)
    for key, value in original_config_dict.items():
        if key not in our_config:
            our_config[key] = value
            logger.info(f"Added missing config attribute: {key} = {value}")
    
    # Save updated config
    with open(our_config_path, 'w') as f:
        json.dump(our_config, f, indent=2)
    
    logger.info("ALBERT weights and config updated successfully!")
    logger.info(f"Model size: {sum(p.numel() for p in original_model.parameters())} parameters")

if __name__ == "__main__":
    download_and_convert_albert_weights()
