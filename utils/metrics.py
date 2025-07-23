"""
Evaluation Metrics Module

This module provides various metrics for evaluating model performance
and attention map quality in the context of anterior segment disease classification.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculator for various evaluation metrics used in the study.
    """
    
    def __init__(self, class_names: Dict[int, str]):
        """
        Initialize metrics calculator.
        
        Args:
            class_names: Dictionary mapping class indices to names
        """
        self.class_names = class_names
        self.n_classes = len(class_names)
        
    def compute_classification_accuracy(self, y_true: np.ndarray, 
                                      y_pred: np.ndarray) -> float:
        """
        Compute overall classification accuracy.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            Accuracy score
        """
        return float(np.mean(y_true == y_pred))
    
    def compute_per_class_accuracy(self, y_true: np.ndarray,
                                  y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute per-class accuracy.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of per-class accuracies
        """
        accuracies = {}
        
        for class_idx, class_name in self.class_names.items():
            mask = y_true == class_idx
            if mask.sum() > 0:
                class_acc = np.mean(y_pred[mask] == class_idx)
                accuracies[class_name] = float(class_acc)
            else:
                accuracies[class_name] = 0.0
                
        return accuracies
    
    def compute_confusion_matrix(self, y_true: np.ndarray,
                               y_pred: np.ndarray,
                               normalize: bool = False) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            normalize: Whether to normalize the matrix
            
        Returns:
            Confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred, labels=list(self.class_names.keys()))
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
        return cm
    
    def compute_aoi_50(self, cam_heatmap: np.ndarray, 
                      threshold: float = 0.5) -> float:
        """
        Compute Area of Interest (AOI_50) metric.
        
        Args:
            cam_heatmap: CAM heatmap array
            threshold: Threshold for AOI calculation
            
        Returns:
            AOI_50 value
        """
        # Normalize heatmap
        cam_norm = cam_heatmap / (cam_heatmap.max() + 1e-8)
        
        # Calculate proportion above threshold
        aoi = np.sum(cam_norm > threshold) / cam_norm.size
        
        return float(aoi)
    
    def compute_iou(self, pred_mask: np.ndarray,
                   gt_mask: np.ndarray) -> float:
        """
        Compute Intersection over Union.
        
        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask
            
        Returns:
            IoU score
        """
        # Ensure binary
        pred_mask = (pred_mask > 0).astype(np.uint8)
        gt_mask = (gt_mask > 0).astype(np.uint8)
        
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        if union == 0:
            return 0.0 if intersection == 0 else 1.0
            
        return float(intersection / union)
    
    def compute_pointing_game_accuracy(self, cam_heatmaps: List[np.ndarray],
                                     gt_masks: List[np.ndarray]) -> float:
        """
        Compute Pointing Game accuracy over multiple samples.
        
        Args:
            cam_heatmaps: List of CAM heatmaps
            gt_masks: List of ground truth masks
            
        Returns:
            Pointing Game accuracy
        """
        hits = 0
        total = len(cam_heatmaps)
        
        for cam, mask in zip(cam_heatmaps, gt_masks):
            # Find max activation point
            max_loc = np.unravel_index(cam.argmax(), cam.shape)
            
            # Check if within GT region
            if mask[max_loc] > 0:
                hits += 1
                
        return float(hits / total) if total > 0 else 0.0
    
    def compute_layer_statistics(self, layer_aoi_dict: Dict[str, List[float]]) -> pd.DataFrame:
        """
        Compute statistics for each layer's AOI values.
        
        Args:
            layer_aoi_dict: Dictionary mapping layer names to AOI values
            
        Returns:
            DataFrame with layer statistics
        """
        stats = []
        
        for layer, aoi_values in layer_aoi_dict.items():
            if aoi_values:
                stats.append({
                    'layer': layer,
                    'mean_aoi': np.mean(aoi_values),
                    'median_aoi': np.median(aoi_values),
                    'std_aoi': np.std(aoi_values),
                    'min_aoi': np.min(aoi_values),
                    'max_aoi': np.max(aoi_values),
                    'q25_aoi': np.percentile(aoi_values, 25),
                    'q75_aoi': np.percentile(aoi_values, 75),
                    'n_samples': len(aoi_values)
                })
                
        return pd.DataFrame(stats)
    
    def compute_cut_paste_metrics(self, accuracy_matrix: np.ndarray) -> Dict:
        """
        Compute metrics from cut-and-paste validation results.
        
        Args:
            accuracy_matrix: Accuracy matrix (source x background)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Diagonal accuracy (same background)
        diagonal_acc = np.diag(accuracy_matrix).mean()
        metrics['same_background_accuracy'] = float(diagonal_acc)
        
        # Off-diagonal accuracy (different background)
        mask = ~np.eye(accuracy_matrix.shape[0], dtype=bool)
        off_diagonal_acc = accuracy_matrix[mask].mean()
        metrics['different_background_accuracy'] = float(off_diagonal_acc)
        
        # Context dependency score
        metrics['context_dependency'] = float(diagonal_acc - off_diagonal_acc)
        
        # Per-class metrics
        per_class = {}
        for i, class_name in self.class_names.items():
            if i < accuracy_matrix.shape[0]:
                per_class[class_name] = {
                    'same_bg_acc': float(accuracy_matrix[i, i]),
                    'avg_diff_bg_acc': float(np.mean([accuracy_matrix[i, j] 
                                                     for j in range(accuracy_matrix.shape[1]) 
                                                     if j != i])),
                    'context_dependency': float(accuracy_matrix[i, i] - 
                                              np.mean([accuracy_matrix[i, j] 
                                                      for j in range(accuracy_matrix.shape[1]) 
                                                      if j != i]))
                }
        metrics['per_class'] = per_class
        
        return metrics
    
    def generate_classification_report(self, y_true: np.ndarray,
                                     y_pred: np.ndarray,
                                     output_dict: bool = True) -> Union[str, Dict]:
        """
        Generate detailed classification report.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            output_dict: Whether to return as dictionary
            
        Returns:
            Classification report
        """
        target_names = [self.class_names[i] for i in sorted(self.class_names.keys())]
        
        report = classification_report(
            y_true, y_pred,
            labels=list(self.class_names.keys()),
            target_names=target_names,
            output_dict=output_dict,
            zero_division=0
        )
        
        return report
    
    def compute_attention_consistency(self, cam_heatmaps: Dict[str, np.ndarray]) -> float:
        """
        Compute consistency of attention across different layers.
        
        Args:
            cam_heatmaps: Dictionary of CAM heatmaps for different layers
            
        Returns:
            Consistency score
        """
        if len(cam_heatmaps) < 2:
            return 1.0
            
        # Compute pairwise correlations
        correlations = []
        layers = list(cam_heatmaps.keys())
        
        for i in range(len(layers)):
            for j in range(i + 1, len(layers)):
                cam1 = cam_heatmaps[layers[i]].flatten()
                cam2 = cam_heatmaps[layers[j]].flatten()
                
                # Ensure same size
                min_len = min(len(cam1), len(cam2))
                cam1 = cam1[:min_len]
                cam2 = cam2[:min_len]
                
                corr = np.corrcoef(cam1, cam2)[0, 1]
                correlations.append(corr)
                
        return float(np.mean(correlations))
    
    def compute_robustness_metrics(self, original_pred: int,
                                 augmented_preds: List[int]) -> Dict:
        """
        Compute robustness metrics based on predictions under augmentations.
        
        Args:
            original_pred: Original prediction
            augmented_preds: Predictions under various augmentations
            
        Returns:
            Robustness metrics
        """
        metrics = {}
        
        # Consistency rate
        consistency = np.mean([pred == original_pred for pred in augmented_preds])
        metrics['consistency_rate'] = float(consistency)
        
        # Prediction distribution
        unique, counts = np.unique(augmented_preds, return_counts=True)
        pred_dist = dict(zip([self.class_names[i] for i in unique], 
                           counts / len(augmented_preds)))
        metrics['prediction_distribution'] = pred_dist
        
        # Stability score (inverse of entropy)
        probs = counts / len(augmented_preds)
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        metrics['stability_score'] = float(1 / (1 + entropy))
        
        return metrics