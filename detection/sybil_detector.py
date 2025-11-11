"""
Sybil Attack Detection Module for VANET
"""

import numpy as np
from utils.logger import setup_logger
from utils.config import Config

logger = setup_logger(__name__)


class SybilDetector:
    """Detect Sybil attacks in VANET traffic"""
    
    def __init__(self, threshold=None):
        """
        Initialize Sybil detector
        
        Args:
            threshold: Detection threshold
        """
        self.threshold = threshold or Config.SYBIL_THRESHOLD
        
    def detect(self, features, predictions=None):
        """
        Detect Sybil attacks based on network features
        
        Sybil indicators:
        - Abnormally high router count (>5 from single source)
        - Multiple identities with similar patterns
        - Rapid identity changes
        - Normal traffic patterns but suspicious identity behavior
        
        Args:
            features: Network features (dict or DataFrame)
            predictions: Model predictions (optional)
            
        Returns:
            Sybil detection score (0-1)
        """
        logger.info("Detecting Sybil attacks")
        
        # Extract relevant features
        if isinstance(features, dict):
            routers = features.get('Routers', 0)
            throughput = features.get('throughput', 0)
            packet_loss = features.get('packet_loss', 0)
        else:
            routers = features[:, features.columns.get_loc('Routers')] if hasattr(features, 'columns') else features[:, 5]
            throughput = features[:, features.columns.get_loc('throughput')] if hasattr(features, 'columns') else features[:, 0]
            packet_loss = features[:, features.columns.get_loc('packet_loss')] if hasattr(features, 'columns') else features[:, 2]
        
        # Calculate Sybil score
        sybil_score = 0.0
        
        # Abnormal router count indicator (main Sybil indicator)
        if np.mean(routers) > 5:
            sybil_score += 0.5
        
        # Normal throughput but high router count (suspicious)
        if np.mean(throughput) > 1.5 and np.mean(routers) > 4:
            sybil_score += 0.3
        
        # Low packet loss with high routers (identity spoofing)
        if np.mean(packet_loss) < 0.2 and np.mean(routers) > 4:
            sybil_score += 0.2
        
        # Incorporate model prediction if available
        if predictions is not None:
            # Assuming predictions are probabilities for [NORMAL, DOS, SYBIL, WORMHOLE]
            sybil_score = (sybil_score + predictions[2]) / 2
        
        is_sybil = sybil_score > self.threshold
        
        logger.info(f"Sybil detection score: {sybil_score:.4f}, Threshold: {self.threshold}, Detected: {is_sybil}")
        return sybil_score, is_sybil
    
    def analyze_identity_pattern(self, features_sequence):
        """
        Analyze identity spoofing pattern
        
        Args:
            features_sequence: Sequence of feature snapshots
            
        Returns:
            Identity analysis results
        """
        logger.info("Analyzing Sybil identity pattern")
        
        analysis = {
            'duration': len(features_sequence),
            'avg_router_count': 0,
            'router_variance': 0,
            'identity_changes': 0,
            'pattern': 'Unknown'
        }
        
        if len(features_sequence) > 0:
            router_counts = [f.get('Routers', 0) for f in features_sequence]
            analysis['avg_router_count'] = np.mean(router_counts)
            analysis['router_variance'] = np.var(router_counts)
            
            # Count rapid changes in router count (identity switches)
            analysis['identity_changes'] = sum(
                abs(router_counts[i] - router_counts[i-1]) > 2 
                for i in range(1, len(router_counts))
            )
            
            # Identify pattern type
            if analysis['avg_router_count'] > 8:
                analysis['pattern'] = 'Massive Identity Spoofing'
            elif analysis['identity_changes'] > len(features_sequence) * 0.3:
                analysis['pattern'] = 'Rapid Identity Switching'
            elif analysis['router_variance'] > 5:
                analysis['pattern'] = 'Variable Identity Attack'
            else:
                analysis['pattern'] = 'Moderate Sybil Activity'
        
        logger.info(f"Sybil pattern analysis: {analysis}")
        return analysis
