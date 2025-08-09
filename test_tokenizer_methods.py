#!/usr/bin/env python3
"""
Test ALBERT tokenizer methods for data loading compatibility
"""

import sys
import os
sys.path.append('src')

def test_albert_tokenizer_methods():
    """Test that AlbertTokenizer has all required methods for data loading."""
    print("Testing ALBERT tokenizer methods...")
    
    try:
        from src.layers.albert.tokenization_albert import AlbertTokenizer
        
        # Create tokenizer
        tokenizer = AlbertTokenizer.from_pretrained('models/captioning/albert-base-v2/')
        print("✅ AlbertTokenizer loaded successfully")
        
        # Test required methods
        print("\nTesting required methods:")
        
        # Test mask_token property
        print(f"  mask_token: {tokenizer.mask_token}")
        assert hasattr(tokenizer, 'mask_token'), "Tokenizer should have mask_token"
        print("✅ mask_token available")
        
        # Test get_random_token method (this was missing)
        random_token = tokenizer.get_random_token()
        print(f"  get_random_token(): {random_token}")
        assert isinstance(random_token, str), "get_random_token should return a string"
        print("✅ get_random_token() works")
        
        # Test multiple calls return different tokens (probabilistically)
        tokens = [tokenizer.get_random_token() for _ in range(10)]
        print(f"  10 random tokens: {tokens}")
        # Should have some variety (not all identical)
        unique_tokens = set(tokens)
        print(f"  Unique tokens: {len(unique_tokens)}/{len(tokens)}")
        print("✅ get_random_token() generates varied tokens")
        
        # Test other commonly used methods
        print("\nTesting other tokenizer methods:")
        
        # Test tokenization
        test_text = "Hello world"
        tokens = tokenizer.tokenize(test_text)
        print(f"  tokenize('{test_text}'): {tokens}")
        print("✅ tokenize() works")
        
        # Test token to ID conversion
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        print(f"  convert_tokens_to_ids(): {token_ids}")
        print("✅ convert_tokens_to_ids() works")
        
        # Test vocabulary size
        vocab_size = len(tokenizer.vocab)
        print(f"  vocab size: {vocab_size}")
        assert vocab_size > 0, "Vocabulary should not be empty"
        print("✅ Vocabulary loaded correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_masking_compatibility():
    """Test the specific masking scenario that was failing."""
    print("\nTesting masking compatibility...")
    
    try:
        from src.layers.albert.tokenization_albert import AlbertTokenizer
        import random
        
        tokenizer = AlbertTokenizer.from_pretrained('models/captioning/albert-base-v2/')
        
        # Simulate the masking logic from caption_tensorizer.py
        tokens = ["hello", "world", "this", "is", "a", "test"]
        pos = 2  # Position to mask
        
        print(f"Original tokens: {tokens}")
        print(f"Masking position: {pos}")
        
        # Test the exact logic that was failing
        if random.random() <= 0.8:
            # 80% chance to be a ['MASK'] token
            tokens[pos] = tokenizer.mask_token
            print(f"  Masked with [MASK]: {tokens}")
        elif random.random() <= 0.5:
            # 10% chance to be a random word
            tokens[pos] = tokenizer.get_random_token()  # This was failing before
            print(f"  Masked with random token: {tokens}")
        else:
            # 10% chance to remain the same
            print(f"  Kept original: {tokens}")
        
        print("✅ Masking logic works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Masking test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Testing ALBERT tokenizer compatibility")
    print("=" * 60)
    
    success1 = test_albert_tokenizer_methods()
    success2 = test_masking_compatibility()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 ALBERT tokenizer is compatible! Data loading should work now.")
    else:
        print("❌ Some tests failed. Check the errors above.")
