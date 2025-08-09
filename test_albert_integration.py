#!/usr/bin/env python3
"""
Test script to verify ALBERT integration in SwinBERT.
This script tests the ALBERT model loading and basic functionality.
"""

import sys
import os
import torch
from argparse import Namespace

# Add src to path
sys.path.append('src')

def test_albert_import():
    """Test that ALBERT modules can be imported correctly."""
    print("Testing ALBERT imports...")
    try:
        from src.layers.albert import AlbertTokenizer, AlbertConfig, AlbertForImageCaptioning
        from src.modeling.load_albert import get_albert_model
        print("✅ ALBERT imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_albert_config():
    """Test ALBERT configuration loading."""
    print("\nTesting ALBERT configuration...")
    try:
        from src.layers.albert import AlbertConfig
        
        # Test creating config from parameters
        config = AlbertConfig(
            vocab_size_or_config_json_file=30522,
            embedding_size=128,
            hidden_size=768,
            num_hidden_layers=12,
            num_hidden_groups=1,
            num_attention_heads=12,
            intermediate_size=3072,
            inner_group_num=1
        )
        
        print(f"✅ ALBERT config created successfully")
        print(f"   - Vocab size: {config.vocab_size}")
        print(f"   - Embedding size: {config.embedding_size}")
        print(f"   - Hidden size: {config.hidden_size}")
        print(f"   - Hidden layers: {config.num_hidden_layers}")
        print(f"   - Hidden groups: {config.num_hidden_groups}")
        return True
    except Exception as e:
        print(f"❌ Config creation error: {e}")
        return False

def test_albert_tokenizer():
    """Test ALBERT tokenizer functionality."""
    print("\nTesting ALBERT tokenizer...")
    try:
        from src.layers.albert import AlbertTokenizer
        
        # Check if vocab file exists
        vocab_path = "models/captioning/albert-base-v2/vocab.txt"
        if not os.path.exists(vocab_path):
            print(f"⚠️  Vocab file not found at {vocab_path}")
            return False
        
        tokenizer = AlbertTokenizer(vocab_path)
        
        # Test tokenization
        test_text = "A person is walking down the street"
        tokens = tokenizer.tokenize(test_text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        print(f"✅ ALBERT tokenizer working")
        print(f"   - Input: {test_text}")
        print(f"   - Tokens: {tokens[:5]}...")  # Show first 5 tokens
        print(f"   - Token IDs: {token_ids[:5]}...")  # Show first 5 IDs
        return True
    except Exception as e:
        print(f"❌ Tokenizer error: {e}")
        return False

def test_albert_model_loading():
    """Test ALBERT model loading function."""
    print("\nTesting ALBERT model loading...")
    try:
        from src.modeling.load_albert import get_albert_model
        
        # Create test arguments
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
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ ALBERT model loaded successfully")
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")
        print(f"   - Model type: {type(model).__name__}")
        print(f"   - Config type: {type(config).__name__}")
        print(f"   - Tokenizer type: {type(tokenizer).__name__}")
        
        return True
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        return False

def test_albert_forward_pass():
    """Test a simple forward pass through ALBERT."""
    print("\nTesting ALBERT forward pass...")
    try:
        from src.layers.albert import AlbertForImageCaptioning, AlbertConfig
        
        # Create a simple config
        config = AlbertConfig(
            vocab_size_or_config_json_file=30522,
            embedding_size=128,
            hidden_size=256,  # Smaller for testing
            num_hidden_layers=2,
            num_hidden_groups=1,
            num_attention_heads=4,
            intermediate_size=512,
            inner_group_num=1
        )
        
        # Add image feature config
        config.img_feature_dim = 512
        config.img_feature_type = 'frcnn'
        config.use_img_layernorm = False
        config.tie_weights = False
        config.freeze_embedding = False
        
        # Create model
        model = AlbertForImageCaptioning(config)
        model.eval()
        
        # Create dummy inputs
        batch_size = 2
        seq_length = 10
        img_seq_length = 5
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length + img_seq_length)
        img_feats = torch.randn(batch_size, img_seq_length, config.img_feature_dim)
        masked_pos = torch.zeros(batch_size, seq_length, dtype=torch.long)
        masked_pos[:, 0] = 1  # Mask first token
        masked_ids = torch.randint(0, 1000, (batch_size,))
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                img_feats=img_feats,
                attention_mask=attention_mask,
                masked_pos=masked_pos,
                masked_ids=masked_ids,
                is_training=True
            )
        
        loss, logits = outputs[:2]
        
        print(f"✅ ALBERT forward pass successful")
        print(f"   - Loss: {loss.item():.4f}")
        print(f"   - Logits shape: {logits.shape}")
        print(f"   - Output length: {len(outputs)}")
        
        return True
    except Exception as e:
        print(f"❌ Forward pass error: {e}")
        return False

def main():
    """Run all ALBERT integration tests."""
    print("🚀 Starting ALBERT Integration Tests")
    print("=" * 50)
    
    tests = [
        test_albert_import,
        test_albert_config,
        test_albert_tokenizer,
        test_albert_model_loading,
        test_albert_forward_pass
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 All tests passed! ALBERT integration is working correctly.")
        print("\n📝 Next steps:")
        print("1. Update your training configurations to use ALBERT")
        print("2. Download pre-trained ALBERT weights if needed")
        print("3. Run training with ALBERT on your dataset")
        return True
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
