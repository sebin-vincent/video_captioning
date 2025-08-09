#!/usr/bin/env python3
"""
Migration script to replace BERT with ALBERT in SwinBERT codebase.
This script will:
1. Update import statements
2. Replace function calls
3. Update configuration files
4. Create ALBERT model directories
"""

import os
import re
import shutil
from pathlib import Path

def update_imports_in_file(file_path):
    """Update BERT imports to ALBERT imports in a Python file."""
    if not file_path.endswith('.py'):
        return
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Track if any changes were made
        original_content = content
        
        # Replace BERT imports with ALBERT imports
        replacements = [
            (r'from src\.layers\.bert import', 'from src.layers.albert import'),
            (r'from src\.modeling\.load_bert import', 'from src.modeling.load_albert import'),
            (r'BertTokenizer', 'AlbertTokenizer'),
            (r'BertConfig', 'AlbertConfig'),
            (r'BertForImageCaptioning', 'AlbertForImageCaptioning'),
            (r'BertImgModel', 'AlbertImgModel'),
            (r'get_bert_model', 'get_albert_model'),
        ]
        
        for old_pattern, new_pattern in replacements:
            content = re.sub(old_pattern, new_pattern, content)
        
        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated: {file_path}")
            return True
        
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False
    
    return False

def create_albert_model_structure():
    """Create the ALBERT model directory structure."""
    albert_dir = Path("models/captioning/albert-base-v2")
    albert_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy BERT files as base for ALBERT
    bert_dir = Path("models/captioning/bert-base-uncased")
    if bert_dir.exists():
        for file_name in ["config.json", "vocab.txt", "special_tokens_map.json", "added_tokens.json"]:
            src_file = bert_dir / file_name
            dst_file = albert_dir / file_name
            if src_file.exists():
                shutil.copy2(src_file, dst_file)
                print(f"Copied {src_file} to {dst_file}")
    
    # Update the config.json to have ALBERT-specific parameters
    config_file = albert_dir / "config.json"
    if config_file.exists():
        try:
            import json
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Update config for ALBERT
            config.update({
                "model_type": "albert",
                "embedding_size": 128,
                "num_hidden_groups": 1,
                "inner_group_num": 1,
                "hidden_act": "gelu_new",
                "classifier_dropout_prob": 0.1
            })
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"Updated ALBERT config: {config_file}")
            
        except Exception as e:
            print(f"Error updating config: {e}")

def update_main_files():
    """Update the main training and inference files."""
    main_files = [
        "src/tasks/run_caption_VidSwinBert.py",
        "src/tasks/run_caption_VidSwinBert_inference.py",
        "src/modeling/video_captioning_e2e_vid_swin_bert.py"
    ]
    
    for file_path in main_files:
        if os.path.exists(file_path):
            update_imports_in_file(file_path)

def main():
    """Main migration function."""
    print("Starting BERT to ALBERT migration...")
    
    print("\n1. Creating ALBERT model directory structure...")
    create_albert_model_structure()
    
    print("\n2. Updating main Python files...")
    update_main_files()
    
    print("\n3. Scanning for additional Python files to update...")
    # Update all Python files in src directory
    src_dir = Path("src")
    if src_dir.exists():
        for py_file in src_dir.rglob("*.py"):
            update_imports_in_file(str(py_file))
    
    print("\n4. Migration complete!")
    print("\nNext steps:")
    print("- Update your training scripts to use the new ALBERT configuration files")
    print("- Update model_name_or_path in config files to point to 'models/captioning/albert-base-v2/'")
    print("- Test the training pipeline with a small dataset")
    print("- Download actual ALBERT pre-trained weights if needed")

if __name__ == "__main__":
    main()
