"""
Main training script for VANET IDS
"""

import numpy as np
import argparse
from pathlib import Path

from utils.config import Config
from utils.logger import get_default_logger
from preprocessing.data_loader import DataLoader
from preprocessing.synthetic_generator import VANETAttackGenerator
from models.cnn_lstm_ids import CNNLSTM_IDS, create_sequences
from models.autoencoder_ids import AutoencoderIDS
from evaluation.metrics import IDSMetrics

logger = get_default_logger()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train VANET IDS models')
    
    parser.add_argument('--model', type=str, default='cnn_lstm', 
                       choices=['cnn_lstm', 'autoencoder', 'both'],
                       help='Model to train')
    parser.add_argument('--use_cicids', action='store_true',
                       help='Use CICIDS2017 dataset')
    parser.add_argument('--generate_synthetic', action='store_true',
                       help='Generate synthetic attack data')
    parser.add_argument('--samples_per_class', type=int, default=1000,
                       help='Number of synthetic samples per class')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--sequence_length', type=int, default=10,
                       help='Sequence length for LSTM')
    
    return parser.parse_args()


def generate_synthetic_data(samples_per_class=1000):
    """Generate synthetic VANET attack data"""
    logger.info("Generating synthetic VANET attack data")
    
    generator = VANETAttackGenerator()
    dataset = generator.generate_balanced_dataset(samples_per_class=samples_per_class)
    
    # Save dataset
    output_file = 'vanet_synthetic_attacks.csv'
    generator.save_synthetic_data(dataset, output_file)
    
    logger.info(f"Synthetic data generated: {len(dataset)} samples")
    return dataset


def prepare_data_for_training(use_cicids=False, use_synthetic=False, synthetic_samples=1000):
    """
    Prepare training data
    
    Args:
        use_cicids: Whether to use CICIDS2017
        use_synthetic: Whether to use synthetic data
        synthetic_samples: Number of synthetic samples per class
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info("Preparing training data")
    
    data_loader = DataLoader()
    
    if use_synthetic:
        # Generate synthetic data
        synthetic_df = generate_synthetic_data(synthetic_samples)
        
        # Convert attack types to numeric labels
        label_map = {'NORMAL': 0, 'DOS': 1, 'SYBIL': 2, 'WORMHOLE': 3}
        synthetic_df['label'] = synthetic_df['attack_type'].map(label_map)
        
        # Drop non-feature columns
        X_df = synthetic_df.drop(['attack_type', 'label'], axis=1)
        y = synthetic_df['label'].values
        X = X_df.values
        
        # Split data
        X_train, X_test, y_train, y_test = data_loader.create_train_test_split(X, y)
        
        # Scale features
        X_train, X_test = data_loader.scale_features(X_train, X_test)
        
    else:
        # Use existing dataset
        X_train, X_test, y_train, y_test = data_loader.prepare_data_pipeline(use_cicids=use_cicids)
    
    logger.info(f"Data prepared - Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def train_cnn_lstm(X_train, X_test, y_train, y_test, sequence_length=10, 
                   epochs=None, batch_size=None):
    """
    Train CNN-LSTM model
    
    Args:
        X_train, X_test, y_train, y_test: Training and test data
        sequence_length: Length of sequences for LSTM
        epochs: Number of epochs
        batch_size: Batch size
        
    Returns:
        Trained model and evaluation metrics
    """
    logger.info("Training CNN-LSTM model")
    
    # Create sequences
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)
    
    # Build model
    input_shape = (sequence_length, X_train.shape[1])
    num_classes = len(np.unique(y_train))
    
    model = CNNLSTM_IDS(input_shape=input_shape, num_classes=num_classes)
    model.build_model()
    model.compile_model()
    
    # Print summary
    print("\n" + "="*60)
    print("CNN-LSTM MODEL ARCHITECTURE")
    print("="*60)
    model.summary()
    
    # Train model
    history = model.train(
        X_train_seq, y_train_seq,
        X_val=X_test_seq, y_val=y_test_seq,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Evaluate model
    logger.info("Evaluating CNN-LSTM model")
    eval_metrics = model.evaluate(X_test_seq, y_test_seq)
    
    # Get predictions
    y_pred_proba = model.predict(X_test_seq)
    y_pred = model.predict_classes(X_test_seq)
    
    # Calculate detailed metrics
    metrics_calculator = IDSMetrics()
    detailed_metrics = metrics_calculator.calculate_all_metrics(
        y_test_seq, y_pred, y_pred_proba
    )
    
    # Print results
    metrics_calculator.print_classification_report(y_test_seq, y_pred)
    metrics_calculator.print_confusion_matrix(y_test_seq, y_pred)
    metrics_calculator.summarize_performance(detailed_metrics)
    
    # Save model
    model.save_model()
    logger.info("CNN-LSTM model training completed")
    
    return model, detailed_metrics


def train_autoencoder(X_train, X_test, y_train, y_test, epochs=None, batch_size=None):
    """
    Train Autoencoder model
    
    Args:
        X_train, X_test, y_train, y_test: Training and test data
        epochs: Number of epochs
        batch_size: Batch size
        
    Returns:
        Trained model and evaluation metrics
    """
    logger.info("Training Autoencoder model")
    
    # Build model
    input_dim = X_train.shape[1]
    
    model = AutoencoderIDS(input_dim=input_dim, encoding_dim=Config.ENCODING_DIM)
    model.build_model()
    model.compile_model()
    
    # Print summary
    print("\n" + "="*60)
    print("AUTOENCODER MODEL ARCHITECTURE")
    print("="*60)
    model.summary()
    
    # Train on normal data only (for anomaly detection)
    X_train_normal = X_train[y_train == 0]  # Assuming 0 is NORMAL
    X_test_normal = X_test[y_test == 0]
    
    logger.info(f"Training on {len(X_train_normal)} normal samples")
    
    history = model.train(
        X_train_normal,
        X_val=X_test_normal,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Set anomaly threshold
    model.set_threshold(X_train_normal)
    
    # Evaluate
    logger.info("Evaluating Autoencoder model")
    eval_metrics = model.evaluate(X_test)
    
    # Detect anomalies on test set
    anomalies, errors = model.detect_anomalies(X_test)
    
    # Compare with true labels (0 = normal, >0 = attack)
    y_true_binary = (y_test > 0).astype(int)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_true_binary, anomalies)
    precision = precision_score(y_true_binary, anomalies, zero_division=0)
    recall = recall_score(y_true_binary, anomalies, zero_division=0)
    f1 = f1_score(y_true_binary, anomalies, zero_division=0)
    
    print("\n" + "="*60)
    print("AUTOENCODER ANOMALY DETECTION PERFORMANCE")
    print("="*60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("="*60 + "\n")
    
    # Save model
    model.save_model()
    logger.info("Autoencoder model training completed")
    
    return model, eval_metrics


def main():
    """Main training function"""
    args = parse_arguments()
    
    # Create directories
    Config.create_directories()
    
    logger.info("Starting VANET IDS training")
    logger.info(f"Arguments: {vars(args)}")
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data_for_training(
        use_cicids=args.use_cicids,
        use_synthetic=args.generate_synthetic,
        synthetic_samples=args.samples_per_class
    )
    
    # Train models
    if args.model in ['cnn_lstm', 'both']:
        train_cnn_lstm(
            X_train, X_test, y_train, y_test,
            sequence_length=args.sequence_length,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    
    if args.model in ['autoencoder', 'both']:
        train_autoencoder(
            X_train, X_test, y_train, y_test,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
    
    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()
