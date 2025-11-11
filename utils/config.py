"""
Configuration management for VANET IDS project
"""

import os
from pathlib import Path

class Config:
    """Configuration class for VANET IDS"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    SYNTHETIC_DATA_DIR = DATA_DIR / 'synthetic'
    
    MODELS_DIR = PROJECT_ROOT / 'saved_models'
    RESULTS_DIR = PROJECT_ROOT / 'results'
    LOGS_DIR = PROJECT_ROOT / 'logs'
    
    # Dataset configuration
    CICIDS2017_PATH = RAW_DATA_DIR / 'CICIDS2017'
    NETWORK_DATASET_PATH = RAW_DATA_DIR / 'network_dataset_labeled.csv'
    
    # Attack types
    ATTACK_TYPES = ['NORMAL', 'DOS', 'SYBIL', 'WORMHOLE']
    NUM_CLASSES = len(ATTACK_TYPES)
    
    # Model hyperparameters
    CNN_FILTERS = [64, 128, 256]
    LSTM_UNITS = [128, 64]
    ENCODING_DIM = 32
    DROPOUT_RATE = 0.3
    
    # Training configuration
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.15
    TEST_SPLIT = 0.15
    EARLY_STOPPING_PATIENCE = 10
    
    # Feature engineering
    SEQUENCE_LENGTH = 10  # For temporal sequences
    NUM_FEATURES = None  # Will be set dynamically
    
    # Synthetic data generation
    SYNTHETIC_SAMPLES_PER_ATTACK = 1000
    
    # Performance thresholds
    ANOMALY_THRESHOLD_PERCENTILE = 95
    DOS_THRESHOLD = 0.7
    SYBIL_THRESHOLD = 0.6
    WORMHOLE_THRESHOLD = 0.6
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        directories = [
            cls.DATA_DIR, cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR,
            cls.SYNTHETIC_DATA_DIR, cls.MODELS_DIR, cls.RESULTS_DIR, cls.LOGS_DIR
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_model_path(cls, model_name):
        """Get path for saving/loading model"""
        return cls.MODELS_DIR / f"{model_name}.h5"
    
    @classmethod
    def get_results_path(cls, filename):
        """Get path for saving results"""
        return cls.RESULTS_DIR / filename
