#!/usr/bin/env python3
"""
Test embedding dimension compatibility fix for ALBERT
"""

import sys
import os
sys.path.append('src')
import torch

def test_embedding_dimensions():
    """Test that text and image embeddings have compatible dimensions."""
    print("Testing ALBERT embedding dimension fix...")
    
    try:
        from src.layers.albert.modeling_albert import AlbertConfig, AlbertImgModel
        
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
        
        print(f"✅ Config created - embedding_size: {config.embedding_size}, hidden_size: {config.hidden_size}")
        
        # Create model
        model = AlbertImgModel(config)
        print("✅ AlbertImgModel created")
        
        # Test embedding dimensions
        batch_size = 2
        seq_length = 10
        img_seq_length = 49  # 7x7 image patches
        
        # Create dummy inputs
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        img_feats = torch.randn(batch_size, img_seq_length, config.img_feature_dim)
        
        print(f"Input shapes:")
        print(f"  text input_ids: {input_ids.shape}")
        print(f"  img_feats: {img_feats.shape}")
        
        # Test text embeddings
        text_embeddings = model.embeddings(input_ids)
        print(f"  text_embeddings: {text_embeddings.shape}")
        
        # Test image embeddings
        img_embeddings = model.img_embedding(img_feats)
        print(f"  img_embeddings: {img_embeddings.shape}")
        
        # Verify dimensions match for concatenation
        assert text_embeddings.shape[-1] == img_embeddings.shape[-1], f"Embedding dimensions don't match: {text_embeddings.shape[-1]} vs {img_embeddings.shape[-1]}"
        print(f"✅ Embedding dimensions match: {text_embeddings.shape[-1]}")
        
        # Test concatenation
        combined_embeddings = torch.cat((text_embeddings, img_embeddings), 1)
        print(f"  combined_embeddings: {combined_embeddings.shape}")
        print("✅ Concatenation successful!")
        
        # Test that transformer can handle combined embeddings
        encoder_output = model.encoder(combined_embeddings)
        print(f"  encoder_output: {encoder_output[0].shape}")
        print("✅ Transformer encoder processing successful!")
        
        # Verify final dimensions
        assert encoder_output[0].shape[-1] == config.hidden_size, f"Final hidden size should be {config.hidden_size}, got {encoder_output[0].shape[-1]}"
        print(f"✅ Final hidden size correct: {encoder_output[0].shape[-1]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Testing ALBERT embedding dimension fix")
    print("=" * 60)
    
    success = test_embedding_dimensions()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Embedding dimension fix works! Tensor concatenation should succeed now.")
    else:
        print("❌ Test failed. Check the errors above.")
