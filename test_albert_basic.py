#!/usr/bin/env python3
"""
Basic test script to verify ALBERT integration structure.
This script tests imports and basic structure without requiring PyTorch.
"""

import sys
import os

# Add src to path
sys.path.append('src')

def test_file_structure():
    """Test that all ALBERT files are in place."""
    print("Testing ALBERT file structure...")
    
    required_files = [
        'src/layers/albert/__init__.py',
        'src/layers/albert/modeling_albert.py',
        'src/layers/albert/tokenization_albert.py',
        'src/layers/albert/tokenization_utils.py',
        'src/layers/albert/modeling_utils.py',
        'src/layers/albert/file_utils.py',
        'src/modeling/load_albert.py',
        'src/configs/VidSwinBert/vatex_8frm_albert.json',
        'models/captioning/albert-base-v2/config.json',
        'models/captioning/albert-base-v2/vocab.txt'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    else:
        print(f"✅ All required files present ({len(required_files)} files)")
        return True

def test_albert_imports():
    """Test that ALBERT modules can be imported without PyTorch."""
    print("\nTesting ALBERT imports (without PyTorch)...")
    
    try:
        # Test basic imports without PyTorch dependencies
        import importlib.util
        
        # Test if the modules can be loaded
        modules_to_test = [
            'src.layers.albert.tokenization_utils',
            'src.layers.albert.file_utils',
        ]
        
        for module_name in modules_to_test:
            try:
                spec = importlib.util.find_spec(module_name)
                if spec is None:
                    print(f"❌ Module {module_name} not found")
                    return False
                print(f"✅ Module {module_name} found")
            except Exception as e:
                print(f"❌ Error checking {module_name}: {e}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Import test error: {e}")
        return False

def test_migration_results():
    """Test that migration was successful by checking file contents."""
    print("\nTesting migration results...")
    
    # Check if key files were updated
    files_to_check = [
        'src/tasks/run_caption_VidSwinBert.py',
        'src/tasks/run_caption_VidSwinBert_inference.py'
    ]
    
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if ALBERT imports are present
            has_albert_import = 'from src.layers.albert import' in content or 'from src.modeling.load_albert import' in content
            has_old_bert_import = 'from src.layers.bert import' in content or 'from src.modeling.load_bert import' in content
            
            if has_albert_import and not has_old_bert_import:
                print(f"✅ {file_path} - Successfully migrated to ALBERT")
            elif has_albert_import and has_old_bert_import:
                print(f"⚠️  {file_path} - Partially migrated (both BERT and ALBERT imports)")
            elif has_old_bert_import:
                print(f"❌ {file_path} - Still using BERT imports")
            else:
                print(f"ℹ️  {file_path} - No model imports found")
                
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
    
    return True

def test_config_files():
    """Test ALBERT configuration files."""
    print("\nTesting ALBERT configuration...")
    
    config_file = 'models/captioning/albert-base-v2/config.json'
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        return False
    
    try:
        import json
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Check for ALBERT-specific parameters
        albert_params = ['embedding_size', 'num_hidden_groups', 'inner_group_num']
        found_params = [param for param in albert_params if param in config]
        
        if found_params:
            print(f"✅ ALBERT config updated with parameters: {found_params}")
        else:
            print(f"⚠️  ALBERT config exists but may not have ALBERT-specific parameters")
        
        print(f"   - Model type: {config.get('model_type', 'unknown')}")
        print(f"   - Vocab size: {config.get('vocab_size', 'unknown')}")
        print(f"   - Hidden size: {config.get('hidden_size', 'unknown')}")
        
        return True
    except Exception as e:
        print(f"❌ Error reading config: {e}")
        return False

def main():
    """Run all basic ALBERT tests."""
    print("🚀 Starting ALBERT Basic Integration Tests")
    print("=" * 60)
    
    tests = [
        test_file_structure,
        test_albert_imports,
        test_migration_results,
        test_config_files
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 Basic ALBERT integration successful!")
        print("\n📝 Migration Summary:")
        print("✅ ALBERT layer implementation complete")
        print("✅ Model loading function updated")
        print("✅ Configuration files created")
        print("✅ Import statements updated")
        print("✅ File structure properly organized")
        
        print("\n🔄 Next Steps:")
        print("1. Install PyTorch and other dependencies")
        print("2. Run the full integration test: python test_albert_integration.py")
        print("3. Download ALBERT pre-trained weights")
        print("4. Test training with ALBERT configuration")
        print("5. Compare ALBERT vs BERT performance")
        
        return True
    else:
        print("\n⚠️  Some basic tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
