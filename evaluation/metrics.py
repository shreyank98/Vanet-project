"""
Evaluation metrics for VANET IDS
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from utils.logger import setup_logger
from utils.config import Config

logger = setup_logger(__name__)


class IDSMetrics:
    """Calculate and report IDS performance metrics"""
    
    def __init__(self, class_names=None):
        """
        Initialize metrics calculator
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names or Config.ATTACK_TYPES
        
    def calculate_all_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate comprehensive metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        logger.info("Calculating evaluation metrics")
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Weighted metrics (accounting for class imbalance)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            if i < len(precision_per_class):
                metrics[f'precision_{class_name}'] = precision_per_class[i]
                metrics[f'recall_{class_name}'] = recall_per_class[i]
                metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # ROC-AUC (if probabilities available)
        if y_pred_proba is not None and len(np.unique(y_true)) > 1:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
            except ValueError as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}")
                metrics['roc_auc'] = None
        else:
            metrics['roc_auc'] = None
        
        logger.info(f"Metrics calculated - Accuracy: {metrics['accuracy']:.4f}, "
                   f"F1: {metrics['f1_macro']:.4f}")
        
        return metrics
    
    def calculate_detection_rates(self, y_true, y_pred):
        """
        Calculate attack-specific detection rates
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of detection rates
        """
        logger.info("Calculating attack detection rates")
        
        detection_rates = {}
        
        for i, attack_type in enumerate(self.class_names):
            # True positives: correctly identified attacks
            true_mask = (y_true == i)
            pred_mask = (y_pred == i)
            
            true_positives = np.sum(true_mask & pred_mask)
            actual_count = np.sum(true_mask)
            
            if actual_count > 0:
                detection_rates[attack_type] = {
                    'detection_rate': true_positives / actual_count,
                    'true_positives': int(true_positives),
                    'total_actual': int(actual_count)
                }
            else:
                detection_rates[attack_type] = {
                    'detection_rate': 0.0,
                    'true_positives': 0,
                    'total_actual': 0
                }
        
        logger.info(f"Detection rates: {detection_rates}")
        return detection_rates
    
    def calculate_false_positive_rate(self, y_true, y_pred):
        """
        Calculate false positive rate (important for IDS)
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Overall and per-class false positive rates
        """
        logger.info("Calculating false positive rates")
        
        fpr = {}
        
        # Overall FPR (normal traffic incorrectly classified as attack)
        normal_mask = (y_true == 0)  # Assuming 0 is NORMAL
        false_positives = np.sum(normal_mask & (y_pred != 0))
        total_normal = np.sum(normal_mask)
        
        if total_normal > 0:
            fpr['overall'] = false_positives / total_normal
        else:
            fpr['overall'] = 0.0
        
        # Per-attack FPR
        for i, attack_type in enumerate(self.class_names):
            if i == 0:  # Skip normal class
                continue
                
            # False positives for this attack: other classes predicted as this attack
            other_mask = (y_true != i)
            fp = np.sum(other_mask & (y_pred == i))
            total_other = np.sum(other_mask)
            
            if total_other > 0:
                fpr[attack_type] = fp / total_other
            else:
                fpr[attack_type] = 0.0
        
        logger.info(f"False positive rates: {fpr}")
        return fpr
    
    def print_classification_report(self, y_true, y_pred):
        """
        Print detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            zero_division=0
        )
        
        logger.info(f"\nClassification Report:\n{report}")
        print("\nClassification Report:")
        print(report)
        
    def print_confusion_matrix(self, y_true, y_pred):
        """
        Print confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        cm = confusion_matrix(y_true, y_pred)
        
        logger.info(f"\nConfusion Matrix:\n{cm}")
        print("\nConfusion Matrix:")
        print(cm)
        
        # Print with class names
        print("\nDetailed Confusion Matrix:")
        print(f"{'':12} ", end='')
        for name in self.class_names:
            print(f"{name:12}", end=' ')
        print()
        
        for i, name in enumerate(self.class_names):
            print(f"{name:12} ", end='')
            for j in range(len(self.class_names)):
                if i < len(cm) and j < len(cm[i]):
                    print(f"{cm[i][j]:12}", end=' ')
            print()
    
    def summarize_performance(self, metrics):
        """
        Print performance summary
        
        Args:
            metrics: Dictionary of metrics
        """
        print("\n" + "="*60)
        print("VANET IDS PERFORMANCE SUMMARY")
        print("="*60)
        
        print(f"\nOverall Performance:")
        print(f"  Accuracy:        {metrics['accuracy']:.4f}")
        print(f"  Precision (avg): {metrics['precision_weighted']:.4f}")
        print(f"  Recall (avg):    {metrics['recall_weighted']:.4f}")
        print(f"  F1-Score (avg):  {metrics['f1_weighted']:.4f}")
        
        if metrics.get('roc_auc'):
            print(f"  ROC-AUC:         {metrics['roc_auc']:.4f}")
        
        print(f"\nPer-Attack Performance:")
        for attack in self.class_names:
            precision = metrics.get(f'precision_{attack}', 0)
            recall = metrics.get(f'recall_{attack}', 0)
            f1 = metrics.get(f'f1_{attack}', 0)
            print(f"  {attack:12} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        print("="*60 + "\n")
        
        logger.info("Performance summary displayed")
