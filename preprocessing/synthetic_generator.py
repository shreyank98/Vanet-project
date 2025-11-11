"""
Synthetic VANET attack data generation
"""

import numpy as np
import pandas as pd
from utils.config import Config
from utils.logger import setup_logger

logger = setup_logger(__name__)


class VANETAttackGenerator:
    """Generate synthetic VANET attack scenarios"""
    
    def __init__(self, num_features=None):
        self.num_features = num_features
        
    def generate_dos_attack(self, n_samples=1000, base_stats=None):
        """
        Generate synthetic DoS attack samples
        
        DoS characteristics:
        - High packet loss (>50%)
        - Extreme latency spikes (>1000ms)
        - Network congestion (>0.8)
        - Throughput degradation (<50% of normal)
        
        Args:
            n_samples: Number of samples to generate
            base_stats: Dictionary with mean and std of normal traffic
            
        Returns:
            DataFrame with DoS attack samples
        """
        logger.info(f"Generating {n_samples} DoS attack samples")
        
        dos_data = {
            'throughput': np.random.uniform(0.5, 1.5, n_samples),  # Low throughput
            'congestion': np.random.uniform(0.7, 1.0, n_samples),  # High congestion
            'packet_loss': np.random.uniform(0.3, 0.9, n_samples),  # High packet loss
            'latency': np.random.uniform(800, 2000, n_samples),  # Very high latency
            'jitter': np.random.uniform(50, 200, n_samples),  # High jitter
            'Routers': np.random.randint(1, 5, n_samples),
            'Planned route': np.random.randint(0, 2, n_samples),
            'Network target': np.random.randint(0, 2, n_samples),
            'Percentage video occupancy': np.random.uniform(0, 0.3, n_samples),
            'Bitrate video': np.random.uniform(0, 500, n_samples),
            'Number videos': np.random.randint(0, 3, n_samples),
            'anomaly_throughput': np.ones(n_samples),  # All marked as anomaly
            'anomaly_congestion': np.ones(n_samples),
            'anomaly_packet_loss': np.ones(n_samples),
            'anomaly_latency': np.ones(n_samples),
            'anomaly_jitter': np.ones(n_samples),
            'anomaly': np.ones(n_samples),
            'attack_type': np.full(n_samples, 'DOS')
        }
        
        df = pd.DataFrame(dos_data)
        logger.info(f"Generated {len(df)} DoS attack samples")
        return df
    
    def generate_sybil_attack(self, n_samples=1000, base_stats=None):
        """
        Generate synthetic Sybil attack samples
        
        Sybil characteristics:
        - Multiple identities from same source
        - Rapid identity changes
        - Abnormal routing patterns
        - High router count from single source
        
        Args:
            n_samples: Number of samples to generate
            base_stats: Dictionary with mean and std of normal traffic
            
        Returns:
            DataFrame with Sybil attack samples
        """
        logger.info(f"Generating {n_samples} Sybil attack samples")
        
        sybil_data = {
            'throughput': np.random.uniform(1.5, 3.0, n_samples),  # Normal to high
            'congestion': np.random.uniform(0.3, 0.7, n_samples),  # Moderate
            'packet_loss': np.random.uniform(0.05, 0.2, n_samples),  # Low to moderate
            'latency': np.random.uniform(50, 300, n_samples),  # Normal range
            'jitter': np.random.uniform(5, 50, n_samples),  # Normal range
            'Routers': np.random.randint(5, 15, n_samples),  # Abnormally high router count
            'Planned route': np.random.randint(0, 2, n_samples),
            'Network target': np.random.randint(0, 2, n_samples),
            'Percentage video occupancy': np.random.uniform(0, 0.5, n_samples),
            'Bitrate video': np.random.uniform(500, 2000, n_samples),
            'Number videos': np.random.randint(0, 5, n_samples),
            'anomaly_throughput': np.zeros(n_samples),  # Not throughput anomaly
            'anomaly_congestion': np.zeros(n_samples),
            'anomaly_packet_loss': np.zeros(n_samples),
            'anomaly_latency': np.zeros(n_samples),
            'anomaly_jitter': np.zeros(n_samples),
            'anomaly': np.ones(n_samples),  # But overall anomaly
            'attack_type': np.full(n_samples, 'SYBIL')
        }
        
        df = pd.DataFrame(sybil_data)
        logger.info(f"Generated {len(df)} Sybil attack samples")
        return df
    
    def generate_wormhole_attack(self, n_samples=1000, base_stats=None):
        """
        Generate synthetic Wormhole attack samples
        
        Wormhole characteristics:
        - Artificially reduced latency between distant nodes
        - Abnormal route hops (fewer than expected)
        - Geographic impossibilities in timing
        - Tunnel detection patterns
        
        Args:
            n_samples: Number of samples to generate
            base_stats: Dictionary with mean and std of normal traffic
            
        Returns:
            DataFrame with Wormhole attack samples
        """
        logger.info(f"Generating {n_samples} Wormhole attack samples")
        
        wormhole_data = {
            'throughput': np.random.uniform(2.0, 4.0, n_samples),  # Higher than normal
            'congestion': np.random.uniform(0.1, 0.4, n_samples),  # Low congestion
            'packet_loss': np.random.uniform(0.01, 0.1, n_samples),  # Very low packet loss
            'latency': np.random.uniform(1, 50, n_samples),  # Artificially low latency
            'jitter': np.random.uniform(0.5, 10, n_samples),  # Very low jitter
            'Routers': np.random.randint(1, 3, n_samples),  # Abnormally low router count
            'Planned route': np.ones(n_samples),  # Route mismatch
            'Network target': np.random.randint(0, 2, n_samples),
            'Percentage video occupancy': np.random.uniform(0.3, 0.8, n_samples),
            'Bitrate video': np.random.uniform(1000, 3000, n_samples),
            'Number videos': np.random.randint(1, 5, n_samples),
            'anomaly_throughput': np.ones(n_samples),
            'anomaly_congestion': np.zeros(n_samples),
            'anomaly_packet_loss': np.zeros(n_samples),
            'anomaly_latency': np.ones(n_samples),  # Anomalously low
            'anomaly_jitter': np.ones(n_samples),  # Anomalously low
            'anomaly': np.ones(n_samples),
            'attack_type': np.full(n_samples, 'WORMHOLE')
        }
        
        df = pd.DataFrame(wormhole_data)
        logger.info(f"Generated {len(df)} Wormhole attack samples")
        return df
    
    def generate_normal_traffic(self, n_samples=1000):
        """
        Generate synthetic normal VANET traffic
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with normal traffic samples
        """
        logger.info(f"Generating {n_samples} normal traffic samples")
        
        normal_data = {
            'throughput': np.random.uniform(1.8, 2.5, n_samples),
            'congestion': np.random.uniform(0.1, 0.5, n_samples),
            'packet_loss': np.random.uniform(0.0, 0.1, n_samples),
            'latency': np.random.uniform(20, 100, n_samples),
            'jitter': np.random.uniform(5, 30, n_samples),
            'Routers': np.random.randint(1, 5, n_samples),
            'Planned route': np.random.randint(0, 2, n_samples),
            'Network target': np.random.randint(0, 2, n_samples),
            'Percentage video occupancy': np.random.uniform(0, 0.6, n_samples),
            'Bitrate video': np.random.uniform(500, 2000, n_samples),
            'Number videos': np.random.randint(0, 5, n_samples),
            'anomaly_throughput': np.zeros(n_samples),
            'anomaly_congestion': np.zeros(n_samples),
            'anomaly_packet_loss': np.zeros(n_samples),
            'anomaly_latency': np.zeros(n_samples),
            'anomaly_jitter': np.zeros(n_samples),
            'anomaly': np.zeros(n_samples),
            'attack_type': np.full(n_samples, 'NORMAL')
        }
        
        df = pd.DataFrame(normal_data)
        logger.info(f"Generated {len(df)} normal traffic samples")
        return df
    
    def generate_balanced_dataset(self, samples_per_class=1000):
        """
        Generate a balanced dataset with all attack types
        
        Args:
            samples_per_class: Number of samples per attack type
            
        Returns:
            Combined DataFrame with all attack types
        """
        logger.info("Generating balanced VANET attack dataset")
        
        # Generate all attack types
        normal = self.generate_normal_traffic(samples_per_class)
        dos = self.generate_dos_attack(samples_per_class)
        sybil = self.generate_sybil_attack(samples_per_class)
        wormhole = self.generate_wormhole_attack(samples_per_class)
        
        # Combine all datasets
        combined = pd.concat([normal, dos, sybil, wormhole], ignore_index=True)
        
        # Shuffle
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"Generated balanced dataset with {len(combined)} total samples")
        logger.info(f"Attack distribution:\n{combined['attack_type'].value_counts()}")
        
        return combined
    
    def save_synthetic_data(self, df, filename):
        """
        Save synthetic data to CSV
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        output_path = Config.SYNTHETIC_DATA_DIR / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved synthetic data to {output_path}")


if __name__ == "__main__":
    # Example usage
    generator = VANETAttackGenerator()
    
    # Generate balanced dataset
    dataset = generator.generate_balanced_dataset(samples_per_class=1000)
    
    # Save to file
    generator.save_synthetic_data(dataset, 'vanet_synthetic_attacks.csv')
    
    print(f"Dataset shape: {dataset.shape}")
    print(f"\nAttack type distribution:")
    print(dataset['attack_type'].value_counts())
    print(f"\nSample statistics:")
    print(dataset.describe())
