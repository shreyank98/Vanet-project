"""
Data loading and preprocessing for VANET IDS
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils.config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)


class DataLoader:
    """Load and preprocess datasets for VANET IDS"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def load_network_dataset(self, filepath=None):
        """
        Load the existing network dataset from the repository
        
        Args:
            filepath: Path to network_dataset_labeled.csv
            
        Returns:
            pandas DataFrame
        """
        if filepath is None:
            filepath = Config.NETWORK_DATASET_PATH
            
        logger.info(f"Loading network dataset from {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} records with {len(df.columns)} features")
            return df
        except Exception as e:
            logger.error(f"Error loading network dataset: {e}")
            raise
    
    def load_cicids2017_file(self, filepath):
        """
        Load a single CICIDS2017 CSV file
        
        Args:
            filepath: Path to CICIDS2017 CSV file
            
        Returns:
            pandas DataFrame
        """
        logger.info(f"Loading CICIDS2017 file: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} records from CICIDS2017")
            return df
        except Exception as e:
            logger.error(f"Error loading CICIDS2017 file: {e}")
            raise
    
    def load_cicids2017_dos_attacks(self, data_dir=None):
        """
        Load DoS/DDoS attack data from CICIDS2017
        
        Args:
            data_dir: Directory containing CICIDS2017 files
            
        Returns:
            pandas DataFrame with DoS/DDoS attacks
        """
        if data_dir is None:
            data_dir = Config.CICIDS2017_PATH
        
        logger.info(f"Loading DoS/DDoS attacks from {data_dir}")
        
        # CICIDS2017 file patterns for DoS attacks
        dos_patterns = [
            '*DoS*', '*DDoS*', '*Friday*'  # Friday contains DoS attacks
        ]
        
        dataframes = []
        data_path = Path(data_dir)
        
        if not data_path.exists():
            logger.warning(f"CICIDS2017 directory not found: {data_dir}")
            logger.info("Please download CICIDS2017 dataset and place in data/raw/CICIDS2017/")
            return None
        
        for pattern in dos_patterns:
            for file in data_path.glob(pattern):
                if file.suffix == '.csv':
                    try:
                        df = pd.read_csv(file)
                        dataframes.append(df)
                        logger.info(f"Loaded {len(df)} records from {file.name}")
                    except Exception as e:
                        logger.error(f"Error loading {file.name}: {e}")
        
        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            logger.info(f"Total DoS/DDoS records loaded: {len(combined_df)}")
            return combined_df
        else:
            logger.warning("No CICIDS2017 files found")
            return None
    
    def preprocess_network_dataset(self, df):
        """
        Preprocess the network dataset
        
        Args:
            df: pandas DataFrame
            
        Returns:
            X: Feature matrix
            y: Labels
        """
        logger.info("Preprocessing network dataset")
        
        # Drop timestamp if present
        if 'timestamp' in df.columns:
            df = df.drop('timestamp', axis=1)
        
        # Handle missing values
        df = df.fillna(df.mean(numeric_only=True))
        
        # Separate features and labels
        if 'anomaly' in df.columns:
            y = df['anomaly'].values
            X = df.drop('anomaly', axis=1)
        else:
            logger.warning("No 'anomaly' column found, assuming all normal")
            y = np.zeros(len(df))
            X = df.copy()
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Convert to numpy arrays
        X = X.values
        
        logger.info(f"Preprocessed data shape: {X.shape}")
        return X, y
    
    def preprocess_cicids2017(self, df):
        """
        Preprocess CICIDS2017 dataset
        
        Args:
            df: pandas DataFrame
            
        Returns:
            X: Feature matrix
            y: Labels
        """
        logger.info("Preprocessing CICIDS2017 dataset")
        
        # Handle missing values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        # Extract labels (usually last column or 'Label' column)
        label_col = 'Label' if 'Label' in df.columns else df.columns[-1]
        y = df[label_col].values
        X = df.drop(label_col, axis=1)
        
        # Remove non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        self.feature_names = X.columns.tolist()
        X = X.values
        
        logger.info(f"Preprocessed CICIDS2017 shape: {X.shape}")
        logger.info(f"Unique labels: {np.unique(y)}")
        
        return X, y
    
    def map_labels_to_vanet_attacks(self, labels):
        """
        Map CICIDS2017 labels to VANET attack categories
        
        Args:
            labels: Original labels
            
        Returns:
            Mapped labels (0: NORMAL, 1: DOS, 2: SYBIL, 3: WORMHOLE)
        """
        logger.info("Mapping labels to VANET attack types")
        
        mapped_labels = []
        for label in labels:
            label_str = str(label).upper()
            if 'BENIGN' in label_str or 'NORMAL' in label_str:
                mapped_labels.append(0)  # NORMAL
            elif 'DOS' in label_str or 'DDOS' in label_str:
                mapped_labels.append(1)  # DOS
            else:
                # For now, map other attacks to NORMAL
                # We'll generate synthetic Sybil and Wormhole later
                mapped_labels.append(0)  # NORMAL
        
        return np.array(mapped_labels)
    
    def scale_features(self, X_train, X_test):
        """
        Scale features using StandardScaler
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            Scaled X_train, X_test
        """
        logger.info("Scaling features")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    def create_train_test_split(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into train and test sets
        
        Args:
            X: Features
            y: Labels
            test_size: Test set proportion
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info(f"Creating train/test split (test_size={test_size})")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def prepare_data_pipeline(self, use_cicids=False):
        """
        Complete data preparation pipeline
        
        Args:
            use_cicids: Whether to use CICIDS2017 data
            
        Returns:
            X_train, X_test, y_train, y_test (scaled)
        """
        logger.info("Starting data preparation pipeline")
        
        # Load network dataset
        df_network = self.load_network_dataset()
        X_net, y_net = self.preprocess_network_dataset(df_network)
        
        # Optionally load CICIDS2017
        if use_cicids:
            df_cicids = self.load_cicids2017_dos_attacks()
            if df_cicids is not None:
                X_cic, y_cic = self.preprocess_cicids2017(df_cicids)
                y_cic = self.map_labels_to_vanet_attacks(y_cic)
                
                # Combine datasets (feature alignment needed)
                # For now, use only network dataset
                logger.warning("Feature alignment for CICIDS2017 not implemented yet")
                X, y = X_net, y_net
            else:
                X, y = X_net, y_net
        else:
            X, y = X_net, y_net
        
        # Create train/test split
        X_train, X_test, y_train, y_test = self.create_train_test_split(X, y)
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        logger.info("Data preparation pipeline completed")
        return X_train_scaled, X_test_scaled, y_train, y_test
