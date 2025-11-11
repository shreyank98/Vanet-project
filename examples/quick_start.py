"""
Quick start example for VANET IDS
Demonstrates basic usage with synthetic data
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
from utils.logger import get_default_logger
from preprocessing.synthetic_generator import VANETAttackGenerator
from preprocessing.data_loader import DataLoader
from models.cnn_lstm_ids import CNNLSTM_IDS, create_sequences
from evaluation.metrics import IDSMetrics

logger = get_default_logger()


def main():
    """Quick start example"""
    print("="*60)
    print("VANET IDS - Quick Start Example")
    print("="*60)
    
    # Create directories
    Config.create_directories()
    
    # Step 1: Generate synthetic data
    print("\n[Step 1/5] Generating synthetic VANET attack data...")
    generator = VANETAttackGenerator()
    dataset = generator.generate_balanced_dataset(samples_per_class=200)
    print(f"  Generated {len(dataset)} samples")
    print(f"  Attack distribution:\n{dataset['attack_type'].value_counts()}")
    
    # Step 2: Prepare data
    print("\n[Step 2/5] Preparing data for training...")
    
    # Convert attack types to numeric labels
    label_map = {'NORMAL': 0, 'DOS': 1, 'SYBIL': 2, 'WORMHOLE': 3}
    dataset['label'] = dataset['attack_type'].map(label_map)
    
    # Extract features and labels
    X_df = dataset.drop(['attack_type', 'label'], axis=1)
    y = dataset['label'].values
    X = X_df.values
    
    # Split and scale
    loader = DataLoader()
    X_train, X_test, y_train, y_test = loader.create_train_test_split(X, y, test_size=0.2)
    X_train, X_test = loader.scale_features(X_train, X_test)
    
    print(f"  Train size: {X_train.shape}, Test size: {X_test.shape}")
    
    # Step 3: Create sequences
    print("\n[Step 3/5] Creating sequences for LSTM...")
    sequence_length = 5  # Shorter sequence for quick demo
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)
    
    print(f"  Train sequences: {X_train_seq.shape}")
    print(f"  Test sequences: {X_test_seq.shape}")
    
    # Step 4: Build and train model
    print("\n[Step 4/5] Building and training CNN-LSTM model...")
    print("  (Training for 5 epochs as demo - use more for better results)")
    
    input_shape = (sequence_length, X_train.shape[1])
    num_classes = 4
    
    model = CNNLSTM_IDS(input_shape=input_shape, num_classes=num_classes)
    model.build_model()
    model.compile_model()
    
    # Quick training
    history = model.train(
        X_train_seq, y_train_seq,
        X_val=X_test_seq, y_val=y_test_seq,
        epochs=5,
        batch_size=32
    )
    
    # Step 5: Evaluate
    print("\n[Step 5/5] Evaluating model...")
    
    y_pred_proba = model.predict(X_test_seq)
    y_pred = model.predict_classes(X_test_seq)
    
    # Calculate metrics
    metrics_calculator = IDSMetrics()
    detailed_metrics = metrics_calculator.calculate_all_metrics(
        y_test_seq, y_pred, y_pred_proba
    )
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    metrics_calculator.print_classification_report(y_test_seq, y_pred)
    metrics_calculator.summarize_performance(detailed_metrics)
    
    # Detection rates
    detection_rates = metrics_calculator.calculate_detection_rates(y_test_seq, y_pred)
    print("\nDetection Rates:")
    for attack, rates in detection_rates.items():
        print(f"  {attack:12}: {rates['detection_rate']:.2%} "
              f"({rates['true_positives']}/{rates['total_actual']})")
    
    print("\n" + "="*60)
    print("Quick start example completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Train with more epochs: python train_model.py --epochs 50")
    print("  2. Generate more data: python train_model.py --samples_per_class 2000")
    print("  3. Try both models: python train_model.py --model both")


if __name__ == "__main__":
    main()
