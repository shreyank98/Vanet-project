"""
Autoencoder Model for VANET Anomaly Detection
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import numpy as np
from utils.config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)


class AutoencoderIDS:
    """Autoencoder model for anomaly detection in VANET"""
    
    def __init__(self, input_dim, encoding_dim=None):
        """
        Initialize Autoencoder
        
        Args:
            input_dim: Dimension of input features
            encoding_dim: Dimension of encoding layer
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim or Config.ENCODING_DIM
        self.model = None
        self.encoder = None
        self.decoder = None
        self.history = None
        self.threshold = None
        
    def build_model(self):
        """Build Autoencoder architecture"""
        logger.info(f"Building Autoencoder with input dim {self.input_dim}, encoding dim {self.encoding_dim}")
        
        # Input layer
        input_layer = layers.Input(shape=(self.input_dim,))
        
        # Encoder
        encoded = layers.Dense(128, activation='relu')(input_layer)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dropout(Config.DROPOUT_RATE)(encoded)
        
        encoded = layers.Dense(64, activation='relu')(encoded)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dropout(Config.DROPOUT_RATE)(encoded)
        
        # Bottleneck (latent representation)
        bottleneck = layers.Dense(self.encoding_dim, activation='relu', name='bottleneck')(encoded)
        
        # Decoder
        decoded = layers.Dense(64, activation='relu')(bottleneck)
        decoded = layers.BatchNormalization()(decoded)
        decoded = layers.Dropout(Config.DROPOUT_RATE)(decoded)
        
        decoded = layers.Dense(128, activation='relu')(decoded)
        decoded = layers.BatchNormalization()(decoded)
        decoded = layers.Dropout(Config.DROPOUT_RATE)(decoded)
        
        # Output layer
        output_layer = layers.Dense(self.input_dim, activation='sigmoid')(decoded)
        
        # Create autoencoder model
        self.model = models.Model(inputs=input_layer, outputs=output_layer, name='Autoencoder_IDS')
        
        # Create encoder model (for feature extraction)
        self.encoder = models.Model(inputs=input_layer, outputs=bottleneck, name='Encoder')
        
        logger.info("Autoencoder model built successfully")
        return self.model
    
    def compile_model(self, learning_rate=None):
        """
        Compile the model
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        if learning_rate is None:
            learning_rate = Config.LEARNING_RATE
            
        logger.info(f"Compiling Autoencoder with learning rate {learning_rate}")
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        logger.info("Autoencoder compiled successfully")
    
    def train(self, X_train, X_val=None, epochs=None, batch_size=None, callbacks_list=None):
        """
        Train the Autoencoder
        
        Args:
            X_train: Training features (input = output for autoencoder)
            X_val: Validation features
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
            
        logger.info(f"Training Autoencoder for {epochs} epochs with batch size {batch_size}")
        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val) if X_val is not None else 0}")
        
        # Default callbacks
        if callbacks_list is None:
            callbacks_list = self._get_default_callbacks()
        
        # Validation data
        validation_data = (X_val, X_val) if X_val is not None else None
        
        # Train model (X_train is both input and target)
        self.history = self.model.fit(
            X_train, X_train,
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
                filepath=str(Config.get_model_path('autoencoder_best')),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        return callbacks_list
    
    def predict_reconstruction_error(self, X):
        """
        Calculate reconstruction error for samples
        
        Args:
            X: Input features
            
        Returns:
            Reconstruction errors (MSE per sample)
        """
        logger.info(f"Calculating reconstruction errors for {len(X)} samples")
        
        # Reconstruct
        reconstructed = self.model.predict(X, verbose=0)
        
        # Calculate MSE per sample
        mse = np.mean(np.square(X - reconstructed), axis=1)
        
        return mse
    
    def set_threshold(self, X_train, percentile=None):
        """
        Set anomaly threshold based on training data
        
        Args:
            X_train: Training features
            percentile: Percentile for threshold (default from config)
            
        Returns:
            Threshold value
        """
        if percentile is None:
            percentile = Config.ANOMALY_THRESHOLD_PERCENTILE
            
        logger.info(f"Setting anomaly threshold at {percentile}th percentile")
        
        # Calculate reconstruction errors on training data
        errors = self.predict_reconstruction_error(X_train)
        
        # Set threshold at specified percentile
        self.threshold = np.percentile(errors, percentile)
        
        logger.info(f"Anomaly threshold set to {self.threshold:.6f}")
        return self.threshold
    
    def detect_anomalies(self, X, threshold=None):
        """
        Detect anomalies based on reconstruction error
        
        Args:
            X: Input features
            threshold: Anomaly threshold (uses self.threshold if None)
            
        Returns:
            Binary array (1 = anomaly, 0 = normal)
        """
        if threshold is None:
            if self.threshold is None:
                raise ValueError("Threshold not set. Call set_threshold() first.")
            threshold = self.threshold
        
        logger.info(f"Detecting anomalies with threshold {threshold:.6f}")
        
        # Calculate reconstruction errors
        errors = self.predict_reconstruction_error(X)
        
        # Classify as anomaly if error exceeds threshold
        anomalies = (errors > threshold).astype(int)
        
        num_anomalies = np.sum(anomalies)
        logger.info(f"Detected {num_anomalies} anomalies out of {len(X)} samples ({num_anomalies/len(X)*100:.2f}%)")
        
        return anomalies, errors
    
    def encode(self, X):
        """
        Encode data to latent representation
        
        Args:
            X: Input features
            
        Returns:
            Encoded features
        """
        if self.encoder is None:
            raise ValueError("Encoder not available. Build model first.")
            
        return self.encoder.predict(X, verbose=0)
    
    def evaluate(self, X_test):
        """
        Evaluate autoencoder on test data
        
        Args:
            X_test: Test features
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating Autoencoder on {len(X_test)} test samples")
        
        results = self.model.evaluate(X_test, X_test, verbose=0)
        
        metrics = {
            'loss': results[0],
            'mae': results[1]
        }
        
        # Calculate reconstruction error statistics
        errors = self.predict_reconstruction_error(X_test)
        metrics['mean_error'] = np.mean(errors)
        metrics['std_error'] = np.std(errors)
        metrics['max_error'] = np.max(errors)
        metrics['min_error'] = np.min(errors)
        
        logger.info(f"Evaluation results: {metrics}")
        return metrics
    
    def save_model(self, filepath=None):
        """
        Save model to file
        
        Args:
            filepath: Path to save model
        """
        if filepath is None:
            filepath = Config.get_model_path('autoencoder_ids')
            
        self.model.save(filepath)
        
        # Save threshold if set
        if self.threshold is not None:
            threshold_path = str(filepath).replace('.h5', '_threshold.npy')
            np.save(threshold_path, self.threshold)
            logger.info(f"Threshold saved to {threshold_path}")
            
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath=None):
        """
        Load model from file
        
        Args:
            filepath: Path to load model from
        """
        if filepath is None:
            filepath = Config.get_model_path('autoencoder_ids')
            
        self.model = keras.models.load_model(filepath)
        
        # Load threshold if exists
        threshold_path = str(filepath).replace('.h5', '_threshold.npy')
        try:
            self.threshold = np.load(threshold_path)
            logger.info(f"Threshold loaded: {self.threshold:.6f}")
        except FileNotFoundError:
            logger.warning("Threshold file not found")
            
        # Recreate encoder
        self.encoder = models.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('bottleneck').output
        )
        
        logger.info(f"Model loaded from {filepath}")
    
    def summary(self):
        """Print model summary"""
        if self.model:
            self.model.summary()
        else:
            logger.warning("Model not built yet")


if __name__ == "__main__":
    # Example usage
    logger.info("Testing Autoencoder model")
    
    # Create dummy data
    num_samples = 1000
    num_features = 17
    
    X_dummy = np.random.randn(num_samples, num_features)
    
    # Build and compile model
    autoencoder = AutoencoderIDS(input_dim=num_features, encoding_dim=32)
    autoencoder.build_model()
    autoencoder.compile_model()
    autoencoder.summary()
    
    # Set threshold
    autoencoder.set_threshold(X_dummy[:800])
    
    # Detect anomalies
    anomalies, errors = autoencoder.detect_anomalies(X_dummy[800:])
    
    logger.info("Autoencoder model test completed")
