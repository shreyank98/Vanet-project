"""
DoS Attack Detection Module for VANET
"""

import numpy as np
from utils.logger import setup_logger
from utils.config import Config

logger = setup_logger(__name__)


class DoSDetector:
    """Detect DoS attacks in VANET traffic"""
    
    def __init__(self, threshold=None):
        """
        Initialize DoS detector
        
        Args:
            threshold: Detection threshold
        """
        self.threshold = threshold or Config.DOS_THRESHOLD
        
    def detect(self, features, predictions=None):
        """
        Detect DoS attacks based on network features
        
        DoS indicators:
        - High packet loss (>50%)
        - Extreme latency (>1000ms)
        - High congestion (>0.8)
        - Low throughput (<1.5)
        
        Args:
            features: Network features (dict or DataFrame)
            predictions: Model predictions (optional)
            
        Returns:
            DoS detection score (0-1)
        """
        logger.info("Detecting DoS attacks")
        
        # Extract relevant features
        if isinstance(features, dict):
            packet_loss = features.get('packet_loss', 0)
            latency = features.get('latency', 0)
            congestion = features.get('congestion', 0)
            throughput = features.get('throughput', 0)
        else:
            # Assume numpy array or DataFrame
            packet_loss = features[:, features.columns.get_loc('packet_loss')] if hasattr(features, 'columns') else features[:, 2]
            latency = features[:, features.columns.get_loc('latency')] if hasattr(features, 'columns') else features[:, 3]
            congestion = features[:, features.columns.get_loc('congestion')] if hasattr(features, 'columns') else features[:, 1]
            throughput = features[:, features.columns.get_loc('throughput')] if hasattr(features, 'columns') else features[:, 0]
        
        # Calculate DoS score
        dos_score = 0.0
        
        # High packet loss indicator
        if np.mean(packet_loss) > 0.5:
            dos_score += 0.3
        
        # Extreme latency indicator
        if np.mean(latency) > 1000:
            dos_score += 0.3
        
        # High congestion indicator
        if np.mean(congestion) > 0.8:
            dos_score += 0.2
        
        # Low throughput indicator
        if np.mean(throughput) < 1.5:
            dos_score += 0.2
        
        # Incorporate model prediction if available
        if predictions is not None:
            # Assuming predictions are probabilities for [NORMAL, DOS, SYBIL, WORMHOLE]
            dos_score = (dos_score + predictions[1]) / 2
        
        is_dos = dos_score > self.threshold
        
        logger.info(f"DoS detection score: {dos_score:.4f}, Threshold: {self.threshold}, Detected: {is_dos}")
        return dos_score, is_dos
    
    def analyze_dos_pattern(self, features_sequence):
        """
        Analyze DoS attack pattern over time
        
        Args:
            features_sequence: Sequence of feature snapshots
            
        Returns:
            Pattern analysis results
        """
        logger.info("Analyzing DoS attack pattern")
        
        analysis = {
            'duration': len(features_sequence),
            'avg_packet_loss': 0,
            'avg_latency': 0,
            'avg_congestion': 0,
            'avg_throughput': 0,
            'pattern': 'Unknown'
        }
        
        # Calculate averages
        if len(features_sequence) > 0:
            analysis['avg_packet_loss'] = np.mean([f.get('packet_loss', 0) for f in features_sequence])
            analysis['avg_latency'] = np.mean([f.get('latency', 0) for f in features_sequence])
            analysis['avg_congestion'] = np.mean([f.get('congestion', 0) for f in features_sequence])
            analysis['avg_throughput'] = np.mean([f.get('throughput', 0) for f in features_sequence])
            
            # Identify pattern type
            if analysis['avg_packet_loss'] > 0.7:
                analysis['pattern'] = 'Flooding Attack'
            elif analysis['avg_latency'] > 1500:
                analysis['pattern'] = 'Slowloris Attack'
            elif analysis['avg_congestion'] > 0.9:
                analysis['pattern'] = 'Resource Exhaustion'
            else:
                analysis['pattern'] = 'Generic DoS'
        
        logger.info(f"DoS pattern analysis: {analysis}")
        return analysis
