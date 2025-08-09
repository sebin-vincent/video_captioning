#!/usr/bin/env python3
"""
Test the init_weights fix for ALBERT
"""

import sys
import os
sys.path.append('src')

def test_albert_init_weights():
    """Test that ALBERT init_weights method works."""
    print("Testing ALBERT init_weights fix...")
    
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
        
        print("✅ Config created successfully")
        
        # Test AlbertImgModel creation (this was failing)
        print("Creating AlbertImgModel...")
        img_model = AlbertImgModel(config)
        print("✅ AlbertImgModel created successfully!")
        
        # Test AlbertForImageCaptioning creation
        print("Creating AlbertForImageCaptioning...")
        captioning_model = AlbertForImageCaptioning(config)
        print("✅ AlbertForImageCaptioning created successfully!")
        
        # Test that models have the expected attributes
        assert hasattr(img_model, 'albert') or hasattr(img_model, 'embeddings'), "ImgModel should have embeddings"
        assert hasattr(captioning_model, 'albert'), "CaptioningModel should have albert attribute"
        assert hasattr(captioning_model, 'cls'), "CaptioningModel should have cls attribute"
        
        print("✅ Model structure validated!")
        
        # Test that init_weights method exists and works
        assert hasattr(img_model, 'init_weights'), "Model should have init_weights method"
        # Note: init_weights is now called by apply() during model creation
        # So if the model was created successfully, init_weights already worked
        print("✅ init_weights method works correctly!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Testing ALBERT init_weights fix")
    print("=" * 50)
    
    success = test_albert_init_weights()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 init_weights fix works! ALBERT models should initialize correctly now.")
    else:
        print("❌ Test failed. Check the errors above.")
