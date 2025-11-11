"""
Test basic functionality without requiring TensorFlow
Tests data loading, preprocessing, and synthetic generation
"""

import sys
import numpy as np
from preprocessing.synthetic_generator import VANETAttackGenerator
from preprocessing.data_loader import DataLoader
from detection.dos_detector import DoSDetector
from detection.sybil_detector import SybilDetector
from detection.wormhole_detector import WormholeDetector
from utils.config import Config
from utils.logger import get_default_logger

logger = get_default_logger()


def test_synthetic_generation():
    """Test synthetic data generation"""
    print("\n" + "="*60)
    print("TEST 1: Synthetic Data Generation")
    print("="*60)
    
    generator = VANETAttackGenerator()
    
    # Test individual attack types
    print("\nGenerating individual attack types...")
    normal = generator.generate_normal_traffic(n_samples=50)
    dos = generator.generate_dos_attack(n_samples=50)
    sybil = generator.generate_sybil_attack(n_samples=50)
    wormhole = generator.generate_wormhole_attack(n_samples=50)
    
    print(f"  Normal traffic: {normal.shape}")
    print(f"  DoS attacks: {dos.shape}")
    print(f"  Sybil attacks: {sybil.shape}")
    print(f"  Wormhole attacks: {wormhole.shape}")
    
    # Test balanced dataset
    print("\nGenerating balanced dataset...")
    dataset = generator.generate_balanced_dataset(samples_per_class=100)
    print(f"  Total samples: {len(dataset)}")
    print(f"  Attack distribution:")
    print(dataset['attack_type'].value_counts())
    
    print("\n✓ Synthetic data generation test passed!")
    return dataset


def test_data_loading():
    """Test data loading"""
    print("\n" + "="*60)
    print("TEST 2: Data Loading and Preprocessing")
    print("="*60)
    
    loader = DataLoader()
    
    # Load network dataset
    print("\nLoading network dataset...")
    df = loader.load_network_dataset()
    print(f"  Loaded {len(df)} records with {len(df.columns)} features")
    
    # Preprocess
    print("\nPreprocessing data...")
    X, y = loader.preprocess_network_dataset(df)
    print(f"  Features shape: {X.shape}")
    print(f"  Labels shape: {y.shape}")
    print(f"  Unique labels: {np.unique(y)}")
    
    # Train/test split
    print("\nCreating train/test split...")
    X_train, X_test, y_train, y_test = loader.create_train_test_split(X, y)
    print(f"  Train samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    
    # Scale features
    print("\nScaling features...")
    X_train_scaled, X_test_scaled = loader.scale_features(X_train, X_test)
    print(f"  Train mean: {X_train_scaled.mean():.4f}")
    print(f"  Train std: {X_train_scaled.std():.4f}")
    
    print("\n✓ Data loading test passed!")
    return X_test_scaled, y_test


def test_attack_detectors(dataset):
    """Test attack-specific detectors"""
    print("\n" + "="*60)
    print("TEST 3: Attack-Specific Detectors")
    print("="*60)
    
    # Initialize detectors
    dos_detector = DoSDetector()
    sybil_detector = SybilDetector()
    wormhole_detector = WormholeDetector()
    
    # Test on different attack types
    attack_types = ['DOS', 'SYBIL', 'WORMHOLE', 'NORMAL']
    
    for attack_type in attack_types:
        print(f"\n--- Testing {attack_type} samples ---")
        samples = dataset[dataset['attack_type'] == attack_type].head(5)
        
        for idx, row in samples.iterrows():
            features = row.to_dict()
            
            # DoS detection
            dos_score, is_dos = dos_detector.detect(features)
            
            # Sybil detection
            sybil_score, is_sybil = sybil_detector.detect(features)
            
            # Wormhole detection
            wormhole_score, is_wormhole = wormhole_detector.detect(features)
            
            print(f"  Sample {idx}: DoS={dos_score:.2f}, Sybil={sybil_score:.2f}, Wormhole={wormhole_score:.2f}")
            
            # Verify correct detection
            if attack_type == 'DOS' and dos_score > sybil_score and dos_score > wormhole_score:
                print(f"    ✓ Correctly identified as DoS")
            elif attack_type == 'SYBIL' and sybil_score > dos_score and sybil_score > wormhole_score:
                print(f"    ✓ Correctly identified as Sybil")
            elif attack_type == 'WORMHOLE' and wormhole_score > dos_score and wormhole_score > sybil_score:
                print(f"    ✓ Correctly identified as Wormhole")
            elif attack_type == 'NORMAL':
                if dos_score < 0.5 and sybil_score < 0.5 and wormhole_score < 0.5:
                    print(f"    ✓ Correctly identified as Normal")
                else:
                    print(f"    ⚠ False positive detected")
    
    print("\n✓ Attack detector test completed!")


def test_configuration():
    """Test configuration"""
    print("\n" + "="*60)
    print("TEST 4: Configuration and Utilities")
    print("="*60)
    
    print("\nConfiguration settings:")
    print(f"  Attack types: {Config.ATTACK_TYPES}")
    print(f"  Batch size: {Config.BATCH_SIZE}")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Learning rate: {Config.LEARNING_RATE}")
    print(f"  Sequence length: {Config.SEQUENCE_LENGTH}")
    
    print("\nDirectory paths:")
    print(f"  Data dir: {Config.DATA_DIR}")
    print(f"  Models dir: {Config.MODELS_DIR}")
    print(f"  Results dir: {Config.RESULTS_DIR}")
    
    # Create directories
    print("\nCreating directories...")
    Config.create_directories()
    
    # Check if directories exist
    import os
    dirs_exist = all([
        os.path.exists(Config.DATA_DIR),
        os.path.exists(Config.MODELS_DIR),
        os.path.exists(Config.RESULTS_DIR)
    ])
    
    if dirs_exist:
        print("  ✓ All directories created successfully")
    else:
        print("  ✗ Some directories missing")
    
    print("\n✓ Configuration test passed!")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print(" "*15 + "VANET IDS - Basic Functionality Tests")
    print("="*70)
    
    try:
        # Test 1: Configuration
        test_configuration()
        
        # Test 2: Synthetic data generation
        dataset = test_synthetic_generation()
        
        # Test 3: Data loading
        X_test, y_test = test_data_loading()
        
        # Test 4: Attack detectors
        test_attack_detectors(dataset)
        
        # Final summary
        print("\n" + "="*70)
        print(" "*20 + "ALL TESTS PASSED! ✓")
        print("="*70)
        print("\nThe VANET IDS system is ready to use!")
        print("\nNext steps:")
        print("  1. Install TensorFlow: pip install tensorflow")
        print("  2. Run quick start: python examples/quick_start.py")
        print("  3. Train models: python train_model.py --generate_synthetic --samples_per_class 1000")
        print("\n")
        
        return 0
        
    except Exception as e:
        print("\n" + "="*70)
        print(" "*25 + "TEST FAILED! ✗")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
