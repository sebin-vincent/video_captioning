#!/usr/bin/env python3
"""
Test head_mask fix for ALBERT (disabling it like BERT)
"""

import sys
import os
sys.path.append('src')
import torch

def test_albert_without_head_mask():
    """Test that ALBERT works without head_mask support."""
    print("Testing ALBERT without head_mask support...")
    
    try:
        from src.layers.albert.modeling_albert import AlbertConfig, AlbertImgModel, AlbertForImageCaptioning
        
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
        
        # Create model
        model = AlbertImgModel(config)
        print("✅ AlbertImgModel created")
        
        # Create full captioning model
        captioning_model = AlbertForImageCaptioning(config)
        print("✅ AlbertForImageCaptioning created")
        
        # Test tensor dimensions
        batch_size = 2
        text_seq_length = 10 
        img_seq_length = 50  # 7x7 image patches
        hidden_size = config.hidden_size
        
        # Create dummy inputs (simulating actual training data)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, text_seq_length))
        img_feats = torch.randn(batch_size, img_seq_length, config.img_feature_dim)
        attention_mask = torch.ones(batch_size, text_seq_length + img_seq_length)
        
        print(f"Input shapes:")
        print(f"  input_ids: {input_ids.shape}")
        print(f"  img_feats: {img_feats.shape}")
        print(f"  attention_mask: {attention_mask.shape}")
        
        # Test forward pass without head_mask
        print("\nTesting forward pass without head_mask...")
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, img_feats=img_feats)
        
        print(f"  Output shapes:")
        print(f"    sequence_output: {outputs[0].shape}")
        print(f"    pooled_output: {outputs[1].shape}")
        print("✅ Forward pass works without head_mask!")
        
        # Test with captioning model
        print("\nTesting captioning model...")
        # Need to create masked inputs for training
        masked_pos = torch.zeros(batch_size, text_seq_length, dtype=torch.long)
        masked_pos[0, 2] = 1  # Mask one token
        masked_ids = torch.randint(0, config.vocab_size, (1,))  # One masked token
        
        outputs = captioning_model.encode_forward(
            input_ids=input_ids,
            img_feats=img_feats, 
            attention_mask=attention_mask,
            masked_pos=masked_pos,
            masked_ids=masked_ids,
            is_training=True
        )
        
        print(f"  Captioning output: loss={outputs[0].item():.4f}")
        print("✅ Captioning model works!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Testing ALBERT head_mask fix")
    print("=" * 60)
    
    success = test_albert_without_head_mask()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Head_mask fix works! Training should proceed now.")
    else:
        print("❌ Test failed. Check the errors above.")
