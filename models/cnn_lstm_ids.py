"""
CNN-LSTM Hybrid Model for VANET Intrusion Detection
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import numpy as np
from utils.config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)


class CNNLSTM_IDS:
    """CNN-LSTM Hybrid model for intrusion detection"""
    
    def __init__(self, input_shape, num_classes=4):
        """
        Initialize CNN-LSTM model
        
        Args:
            input_shape: Shape of input data (sequence_length, num_features)
            num_classes: Number of attack classes (NORMAL, DOS, SYBIL, WORMHOLE)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build CNN-LSTM architecture"""
        logger.info(f"Building CNN-LSTM model with input shape {self.input_shape}")
        
        # Input layer
        inputs = layers.Input(shape=self.input_shape)
        
        # CNN layers for feature extraction
        x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(Config.DROPOUT_RATE)(x)
        
        x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Dropout(Config.DROPOUT_RATE)(x)
        
        x = layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(Config.DROPOUT_RATE)(x)
        
        # LSTM layers for temporal pattern recognition
        x = layers.LSTM(units=128, return_sequences=True)(x)
        x = layers.Dropout(Config.DROPOUT_RATE)(x)
        
        x = layers.LSTM(units=64, return_sequences=False)(x)
        x = layers.Dropout(Config.DROPOUT_RATE)(x)
        
        # Dense layers for classification
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(Config.DROPOUT_RATE)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(Config.DROPOUT_RATE)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        self.model = models.Model(inputs=inputs, outputs=outputs, name='CNN_LSTM_IDS')
        
        logger.info("CNN-LSTM model built successfully")
        return self.model
    
    def compile_model(self, learning_rate=None):
        """
        Compile the model
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        if learning_rate is None:
            learning_rate = Config.LEARNING_RATE
            
        logger.info(f"Compiling model with learning rate {learning_rate}")
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        logger.info("Model compiled successfully")
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=None, batch_size=None, callbacks_list=None):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of epochs
            batch_size: Batch size
            callbacks_list: List of callbacks
            
        Returns:
            Training history
        """
        if epochs is None:
            epochs = Config.EPOCHS
        if batch_size is None:
            batch_size = Config.BATCH_SIZE
            
        logger.info(f"Training model for {epochs} epochs with batch size {batch_size}")
        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val) if X_val is not None else 0}")
        
        # Default callbacks
        if callbacks_list is None:
            callbacks_list = self._get_default_callbacks()
        
        # Validation data
        validation_data = (X_val, y_val) if X_val is not None else None
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        logger.info("Training completed")
        return self.history
    
    def _get_default_callbacks(self):
        """Get default training callbacks"""
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=Config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=str(Config.get_model_path('cnn_lstm_best')),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        return callbacks_list
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predicted class probabilities
        """
        logger.info(f"Making predictions for {len(X)} samples")
        predictions = self.model.predict(X, verbose=0)
        return predictions
    
    def predict_classes(self, X):
        """
        Predict class labels
        
        Args:
            X: Input features
            
        Returns:
            Predicted class labels
        """
        probabilities = self.predict(X)
        return np.argmax(probabilities, axis=1)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating model on {len(X_test)} test samples")
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        metrics = {
            'loss': results[0],
            'accuracy': results[1],
            'precision': results[2],
            'recall': results[3]
        }
        
        # Calculate F1 score
        if metrics['precision'] > 0 and metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / \
                                 (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1_score'] = 0.0
        
        logger.info(f"Evaluation results: {metrics}")
        return metrics
    
    def save_model(self, filepath=None):
        """
        Save model to file
        
        Args:
            filepath: Path to save model
        """
        if filepath is None:
            filepath = Config.get_model_path('cnn_lstm_ids')
            
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath=None):
        """
        Load model from file
        
        Args:
            filepath: Path to load model from
        """
        if filepath is None:
            filepath = Config.get_model_path('cnn_lstm_ids')
            
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def summary(self):
        """Print model summary"""
        if self.model:
            self.model.summary()
        else:
            logger.warning("Model not built yet")


def create_sequences(X, y, sequence_length=10):
    """
    Create sequences from data for LSTM input
    
    Args:
        X: Feature matrix
        y: Labels
        sequence_length: Length of sequences
        
    Returns:
        X_seq: Sequenced features (n_samples, sequence_length, n_features)
        y_seq: Corresponding labels
    """
    logger.info(f"Creating sequences with length {sequence_length}")
    
    X_seq = []
    y_seq = []
    
    for i in range(len(X) - sequence_length + 1):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length-1])  # Use last label in sequence
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    logger.info(f"Created {len(X_seq)} sequences with shape {X_seq.shape}")
    return X_seq, y_seq


if __name__ == "__main__":
    # Example usage
    logger.info("Testing CNN-LSTM model")
    
    # Create dummy data
    sequence_length = 10
    num_features = 17
    num_samples = 1000
    num_classes = 4
    
    X_dummy = np.random.randn(num_samples, sequence_length, num_features)
    y_dummy = np.random.randint(0, num_classes, num_samples)
    
    # Build and compile model
    model = CNNLSTM_IDS(input_shape=(sequence_length, num_features), num_classes=num_classes)
    model.build_model()
    model.compile_model()
    model.summary()
    
    logger.info("CNN-LSTM model test completed")
