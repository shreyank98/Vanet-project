"""
Wormhole Attack Detection Module for VANET
"""

import numpy as np
from utils.logger import setup_logger
from utils.config import Config

logger = setup_logger(__name__)


class WormholeDetector:
    """Detect Wormhole attacks in VANET traffic"""
    
    def __init__(self, threshold=None):
        """
        Initialize Wormhole detector
        
        Args:
            threshold: Detection threshold
        """
        self.threshold = threshold or Config.WORMHOLE_THRESHOLD
        
    def detect(self, features, predictions=None):
        """
        Detect Wormhole attacks based on network features
        
        Wormhole indicators:
        - Artificially low latency (<50ms for distant nodes)
        - Abnormally low router count (<3)
        - High throughput with low latency
        - Route anomalies (planned vs actual)
        
        Args:
            features: Network features (dict or DataFrame)
            predictions: Model predictions (optional)
            
        Returns:
            Wormhole detection score (0-1)
        """
        logger.info("Detecting Wormhole attacks")
        
        # Extract relevant features
        if isinstance(features, dict):
            latency = features.get('latency', 0)
            routers = features.get('Routers', 0)
            throughput = features.get('throughput', 0)
            planned_route = features.get('Planned route', 0)
            jitter = features.get('jitter', 0)
        else:
            latency = features[:, features.columns.get_loc('latency')] if hasattr(features, 'columns') else features[:, 3]
            routers = features[:, features.columns.get_loc('Routers')] if hasattr(features, 'columns') else features[:, 5]
            throughput = features[:, features.columns.get_loc('throughput')] if hasattr(features, 'columns') else features[:, 0]
            planned_route = features[:, features.columns.get_loc('Planned route')] if hasattr(features, 'columns') else features[:, 6]
            jitter = features[:, features.columns.get_loc('jitter')] if hasattr(features, 'columns') else features[:, 4]
        
        # Calculate Wormhole score
        wormhole_score = 0.0
        
        # Artificially low latency indicator (main wormhole indicator)
        if np.mean(latency) < 50:
            wormhole_score += 0.4
        
        # Low router count with high throughput (tunnel indicator)
        if np.mean(routers) < 3 and np.mean(throughput) > 2.0:
            wormhole_score += 0.3
        
        # Very low jitter (too stable, suspicious)
        if np.mean(jitter) < 10:
            wormhole_score += 0.2
        
        # Route mismatch (if planned_route is 1, indicates anomaly)
        if np.mean(planned_route) > 0.5:
            wormhole_score += 0.1
        
        # Incorporate model prediction if available
        if predictions is not None:
            # Assuming predictions are probabilities for [NORMAL, DOS, SYBIL, WORMHOLE]
            wormhole_score = (wormhole_score + predictions[3]) / 2
        
        is_wormhole = wormhole_score > self.threshold
        
        logger.info(f"Wormhole detection score: {wormhole_score:.4f}, Threshold: {self.threshold}, Detected: {is_wormhole}")
        return wormhole_score, is_wormhole
    
    def analyze_tunnel_pattern(self, features_sequence):
        """
        Analyze wormhole tunnel pattern
        
        Args:
            features_sequence: Sequence of feature snapshots
            
        Returns:
            Tunnel analysis results
        """
        logger.info("Analyzing Wormhole tunnel pattern")
        
        analysis = {
            'duration': len(features_sequence),
            'avg_latency': 0,
            'avg_router_count': 0,
            'avg_throughput': 0,
            'latency_variance': 0,
            'pattern': 'Unknown'
        }
        
        if len(features_sequence) > 0:
            latencies = [f.get('latency', 0) for f in features_sequence]
            analysis['avg_latency'] = np.mean(latencies)
            analysis['latency_variance'] = np.var(latencies)
            analysis['avg_router_count'] = np.mean([f.get('Routers', 0) for f in features_sequence])
            analysis['avg_throughput'] = np.mean([f.get('throughput', 0) for f in features_sequence])
            
            # Identify pattern type
            if analysis['avg_latency'] < 30 and analysis['latency_variance'] < 5:
                analysis['pattern'] = 'High-Speed Tunnel'
            elif analysis['avg_router_count'] < 2:
                analysis['pattern'] = 'Direct Tunnel'
            elif analysis['avg_throughput'] > 3.0:
                analysis['pattern'] = 'High-Capacity Wormhole'
            else:
                analysis['pattern'] = 'Standard Wormhole'
        
        logger.info(f"Wormhole pattern analysis: {analysis}")
        return analysis
