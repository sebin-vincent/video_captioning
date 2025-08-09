#!/usr/bin/env python3
"""
Test script to verify ALBERT LayerNorm compatibility fix for Google Colab
"""

import sys
import os
sys.path.append('src')

def test_albert_imports():
    """Test that ALBERT modules can be imported without LayerNorm issues."""
    print("Testing ALBERT imports...")
    
    try:
        # Test basic imports
        from src.layers.albert.modeling_albert import AlbertConfig, AlbertForImageCaptioning
        print("✅ ALBERT classes imported successfully")
        
        # Test LayerNorm specifically
        from src.layers.albert.modeling_albert import AlbertLayerNorm, LayerNormClass
        print("✅ LayerNorm classes imported successfully")
        
        # Test creating a LayerNorm instance
        layer_norm = AlbertLayerNorm(768)
        print(f"✅ LayerNorm instance created: {type(layer_norm)}")
        
        # Test creating ALBERT config
        config = AlbertConfig.from_pretrained('models/captioning/albert-base-v2/')
        print("✅ ALBERT config loaded successfully")
        print(f"   - Model type: {config.model_type}")
        print(f"   - Hidden size: {config.hidden_size}")
        print(f"   - Embedding size: {config.embedding_size}")
        print(f"   - Has output_attentions: {hasattr(config, 'output_attentions')}")
        print(f"   - Has classifier_dropout_prob: {hasattr(config, 'classifier_dropout_prob')}")
        
        # Test creating ALBERT model (this was failing before)
        print("Testing ALBERT model creation...")
        model = AlbertForImageCaptioning(config)
        print("✅ ALBERT model created successfully!")
        print(f"   - Model type: {model.model_type}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """Test the get_albert_model function that was failing."""
    print("\nTesting get_albert_model function...")
    
    try:
        from src.modeling.load_albert import get_albert_model
        
        # Create a mock args object
        class Args:
            model_name_or_path = 'models/captioning/albert-base-v2/'
            config_name = None
            tokenizer_name = None
            do_lower_case = True
            img_feature_dim = 512
            img_feature_type = 'frcnn'
            use_img_layernorm = True
            img_layer_norm_eps = 1e-12
            max_img_seq_length = 196
            num_labels = 2
            tie_weights = True
            freeze_embedding = False
            
        args = Args()
        model, config, tokenizer = get_albert_model(args)
        print("✅ get_albert_model() executed successfully!")
        print(f"   - Model class: {type(model)}")
        print(f"   - Config class: {type(config)}")
        print(f"   - Tokenizer class: {type(tokenizer)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in get_albert_model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Testing ALBERT LayerNorm compatibility fix for Google Colab")
    print("=" * 60)
    
    success1 = test_albert_imports()
    success2 = test_model_loading()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 All tests passed! ALBERT should work in Google Colab now.")
    else:
        print("❌ Some tests failed. Check the errors above.")
