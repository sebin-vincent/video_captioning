#!/usr/bin/env python3
"""
Test the ALBERT loading fix
"""

import sys
import os
sys.path.append('src')

def test_albert_loading():
    """Test that ALBERT loads correctly with our fix."""
    print("Testing ALBERT model loading fix...")
    
    try:
        from src.modeling.load_albert import get_albert_model
        
        # Create a mock args object with all required attributes
        class Args:
            model_name_or_path = 'models/captioning/albert-base-v2/'
            config_name = None
            tokenizer_name = None
            do_lower_case = True
            img_feature_dim = 512
            num_hidden_layers = -1  # Don't change
            hidden_size = -1  # Don't change
            num_attention_heads = -1  # Don't change  
            intermediate_size = -1  # Don't change
            drop_out = 0.1
            tie_weights = True
            freeze_embedding = False
            label_smoothing = 0.0
            drop_worst_ratio = 0.0
            drop_worst_after = 0
            load_partial_weights = False
            
        args = Args()
        
        print("Calling get_albert_model...")
        model, config, tokenizer = get_albert_model(args)
        
        print("✅ Model loading successful!")
        print(f"   - Model class: {type(model)}")
        print(f"   - Model device: {next(model.parameters()).device}")
        print(f"   - Config class: {type(config)}")
        print(f"   - Tokenizer class: {type(tokenizer)}")
        print(f"   - Model type: {getattr(config, 'model_type', 'unknown')}")
        
        # Test that model has expected attributes
        assert hasattr(model, 'albert'), "Model should have 'albert' attribute"
        assert hasattr(model, 'cls'), "Model should have 'cls' attribute"
        print("✅ Model structure validated!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Testing ALBERT loading fix")
    print("=" * 50)
    
    success = test_albert_loading()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALBERT loading fix works! Training should work now.")
    else:
        print("❌ Loading test failed. Check the errors above.")
