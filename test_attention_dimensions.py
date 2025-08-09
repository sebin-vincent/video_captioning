#!/usr/bin/env python3
"""
Test ALBERT attention mechanism dimension handling
"""

import sys
import os
sys.path.append('src')
import torch

def test_attention_dimensions():
    """Test that ALBERT attention handles tensor dimensions correctly."""
    print("Testing ALBERT attention dimension handling...")
    
    try:
        from src.layers.albert.modeling_albert import AlbertConfig, AlbertAttention
        
        # Create config
        config = AlbertConfig(
            vocab_size=30522,
            embedding_size=128,
            hidden_size=768,
            num_hidden_layers=12,
            num_hidden_groups=1,
            num_attention_heads=12,
            intermediate_size=3072,
            inner_group_num=1,
            hidden_act="gelu_new",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            classifier_dropout_prob=0.1,
            output_attentions=False,
            output_hidden_states=False,
            img_feature_dim=512,
            img_feature_type='frcnn',
            use_img_layernorm=True,
            img_layer_norm_eps=1e-12
        )
        
        print("✅ Config created")
        
        # Create attention module
        attention = AlbertAttention(config)
        print("✅ AlbertAttention created")
        
        # Test tensor dimensions
        batch_size = 2
        seq_length = 60  # text (10) + image (50) tokens
        hidden_size = config.hidden_size
        
        # Create dummy inputs
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        attention_mask = torch.ones(batch_size, 1, 1, seq_length) * -10000.0  # Extended mask
        
        print(f"Input shapes:")
        print(f"  hidden_states: {hidden_states.shape}")
        print(f"  attention_mask: {attention_mask.shape}")
        
        # Test without head_mask first
        print("\nTesting without head_mask...")
        outputs = attention(hidden_states, attention_mask=attention_mask)
        print(f"  output shape: {outputs[0].shape}")
        print("✅ Attention without head_mask works!")
        
        # Test with problematic head_mask (5D)
        print("\nTesting with 5D head_mask...")
        head_mask = torch.ones(1, batch_size, config.num_attention_heads, seq_length, seq_length)
        print(f"  head_mask shape: {head_mask.shape} (5D)")
        
        outputs = attention(hidden_states, attention_mask=attention_mask, head_mask=head_mask)
        print(f"  output shape: {outputs[0].shape}")
        print("✅ Attention with 5D head_mask works!")
        
        # Test with correct head_mask (4D)
        print("\nTesting with 4D head_mask...")
        head_mask = torch.ones(batch_size, config.num_attention_heads, seq_length, seq_length)
        print(f"  head_mask shape: {head_mask.shape} (4D)")
        
        outputs = attention(hidden_states, attention_mask=attention_mask, head_mask=head_mask)
        print(f"  output shape: {outputs[0].shape}")
        print("✅ Attention with 4D head_mask works!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Testing ALBERT attention dimension handling")
    print("=" * 60)
    
    success = test_attention_dimensions()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Attention dimension handling works! Training should proceed now.")
    else:
        print("❌ Test failed. Check the errors above.")
