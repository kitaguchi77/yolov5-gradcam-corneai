"""
Expert Annotation Comparison Module

This module compares Grad-CAM++ attention maps with expert-annotated
lesion areas to validate the clinical relevance of model explanations.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
import pandas as pd
from scipy import ndimage
import logging

logger = logging.getLogger(__name__)


class ExpertComparisonValidator:
    """
    Validates Grad-CAM++ attention maps against expert annotations.
    
    This class computes metrics like IoU and Pointing Game accuracy
    to assess whether the model focuses on clinically relevant regions.
    """
    
    def __init__(self, gradcam_model, aoi_threshold: float = 0.5):
        """
        Initialize expert comparison validator.
        
        Args:
            gradcam_model: YOLOv5GradCAMPlusPlus instance
            aoi_threshold: Threshold for binarizing attention maps
        """
        self.gradcam_model = gradcam_model
        self.aoi_threshold = aoi_threshold
        self.expert_annotations = {}
        
    def load_expert_annotations(self, annotation_path: str):
        """
        Load expert annotations from file.
        
        Args:
            annotation_path: Path to annotation file (JSON format)
        """
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
            
        for ann in annotations:
            image_path = ann['image_path']
            self.expert_annotations[image_path] = {
                'lesion_mask': ann.get('lesion_mask'),
                'lesion_bbox': ann.get('lesion_bbox'),
                'lesion_polygon': ann.get('lesion_polygon'),
                'diagnosis': ann.get('diagnosis'),
                'annotator': ann.get('annotator', 'unknown')
            }
            
        logger.info(f"Loaded {len(self.expert_annotations)} expert annotations")
        
    def load_mask_from_polygon(self, polygon_points: List[List[int]], 
                             image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create binary mask from polygon points.
        
        Args:
            polygon_points: List of [x, y] coordinates
            image_shape: (height, width) of image
            
        Returns:
            Binary mask
        """
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        if polygon_points:
            pts = np.array(polygon_points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
            
        return mask
    
    def load_mask_from_file(self, mask_path: str) -> np.ndarray:
        """
        Load binary mask from image file.
        
        Args:
            mask_path: Path to mask image
            
        Returns:
            Binary mask
        """
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return mask
    
    def compute_iou(self, pred_mask: np.ndarray, 
                   gt_mask: np.ndarray) -> float:
        """
        Compute Intersection over Union between predicted and ground truth masks.
        
        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask
            
        Returns:
            IoU score
        """
        # Ensure binary masks
        pred_mask = (pred_mask > 0).astype(np.uint8)
        gt_mask = (gt_mask > 0).astype(np.uint8)
        
        # Compute intersection and union
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        # Avoid division by zero
        if union == 0:
            return 0.0 if intersection == 0 else 1.0
            
        iou = intersection / union
        return float(iou)
    
    def compute_pointing_game_accuracy(self, cam_heatmap: np.ndarray,
                                     gt_mask: np.ndarray) -> bool:
        """
        Compute Pointing Game accuracy metric.
        
        Args:
            cam_heatmap: CAM heatmap (continuous values)
            gt_mask: Ground truth binary mask
            
        Returns:
            True if maximum activation point is within GT region
        """
        # Find location of maximum activation
        max_loc = np.unravel_index(cam_heatmap.argmax(), cam_heatmap.shape)
        
        # Check if max point is within GT mask
        hit = gt_mask[max_loc] > 0
        
        return bool(hit)
    
    def binarize_cam(self, cam_heatmap: np.ndarray, 
                    threshold: Optional[float] = None) -> np.ndarray:
        """
        Binarize CAM heatmap using threshold.
        
        Args:
            cam_heatmap: Continuous CAM values
            threshold: Threshold value (uses self.aoi_threshold if None)
            
        Returns:
            Binary mask
        """
        if threshold is None:
            threshold = self.aoi_threshold
            
        # Normalize CAM to [0, 1]
        cam_norm = cam_heatmap / (cam_heatmap.max() + 1e-8)
        
        # Apply threshold
        binary_mask = (cam_norm > threshold).astype(np.uint8) * 255
        
        return binary_mask
    
    def validate_single_image(self, image_path: str, 
                            class_idx: int,
                            target_layer: str) -> Dict:
        """
        Validate CAM for a single image against expert annotation.
        
        Args:
            image_path: Path to image
            class_idx: Predicted class index
            target_layer: Target layer for CAM
            
        Returns:
            Validation metrics dictionary
        """
        # Check if we have expert annotation
        if image_path not in self.expert_annotations:
            logger.warning(f"No expert annotation found for {image_path}")
            return None
            
        # Generate CAM
        cam, metadata = self.gradcam_model.generate_cam(
            image_path, class_idx, target_layer
        )
        
        # Load expert mask
        expert_ann = self.expert_annotations[image_path]
        
        # Load image to get shape
        img = cv2.imread(image_path)
        img_shape = img.shape[:2]
        
        # Get ground truth mask
        if expert_ann.get('lesion_mask'):
            gt_mask = self.load_mask_from_file(expert_ann['lesion_mask'])
        elif expert_ann.get('lesion_polygon'):
            gt_mask = self.load_mask_from_polygon(
                expert_ann['lesion_polygon'], img_shape
            )
        else:
            logger.warning(f"No mask available for {image_path}")
            return None
            
        # Resize CAM to match image size
        cam_resized = cv2.resize(cam, (img_shape[1], img_shape[0]))
        
        # Binarize CAM
        cam_binary = self.binarize_cam(cam_resized)
        
        # Compute metrics
        iou = self.compute_iou(cam_binary, gt_mask)
        pointing_game = self.compute_pointing_game_accuracy(cam_resized, gt_mask)
        
        # Additional metrics
        results = {
            'image_path': image_path,
            'class_idx': class_idx,
            'target_layer': target_layer,
            'iou': iou,
            'pointing_game_accuracy': pointing_game,
            'aoi_50': metadata['aoi_50'],
            'expert_diagnosis': expert_ann.get('diagnosis'),
            'annotator': expert_ann.get('annotator')
        }
        
        return results
    
    def validate_dataset(self, image_paths: List[str],
                        predictions: List[int],
                        target_layers: Optional[List[str]] = None,
                        output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Validate entire dataset against expert annotations.
        
        Args:
            image_paths: List of image paths
            predictions: List of predicted classes
            target_layers: Layers to analyze (uses all if None)
            output_path: Optional path to save results
            
        Returns:
            Results DataFrame
        """
        if target_layers is None:
            target_layers = self.gradcam_model.target_layers
            
        all_results = []
        
        for img_path, pred_class in zip(image_paths, predictions):
            for layer in target_layers:
                result = self.validate_single_image(img_path, pred_class, layer)
                
                if result is not None:
                    all_results.append(result)
                    
        # Convert to DataFrame
        df_results = pd.DataFrame(all_results)
        
        # Save if requested
        if output_path:
            df_results.to_csv(output_path, index=False)
            logger.info(f"Validation results saved to {output_path}")
            
        return df_results
    
    def analyze_by_correctness(self, validation_df: pd.DataFrame,
                              ground_truth: Dict[str, int]) -> Dict:
        """
        Analyze metrics separately for correct and incorrect predictions.
        
        Args:
            validation_df: Validation results DataFrame
            ground_truth: Dictionary mapping image paths to true labels
            
        Returns:
            Analysis results dictionary
        """
        # Add correctness column
        validation_df['correct'] = validation_df.apply(
            lambda row: row['class_idx'] == ground_truth.get(row['image_path'], -1),
            axis=1
        )
        
        # Group by correctness
        analysis = {}
        
        for correct in [True, False]:
            subset = validation_df[validation_df['correct'] == correct]
            
            if len(subset) == 0:
                continue
                
            label = 'correct' if correct else 'incorrect'
            
            analysis[label] = {
                'n_samples': len(subset),
                'mean_iou': subset['iou'].mean(),
                'std_iou': subset['iou'].std(),
                'mean_pointing_game': subset['pointing_game_accuracy'].mean(),
                'mean_aoi_50': subset['aoi_50'].mean(),
                'std_aoi_50': subset['aoi_50'].std()
            }
            
            # Per-layer analysis
            layer_stats = {}
            for layer in subset['target_layer'].unique():
                layer_subset = subset[subset['target_layer'] == layer]
                layer_stats[layer] = {
                    'mean_iou': layer_subset['iou'].mean(),
                    'mean_pointing_game': layer_subset['pointing_game_accuracy'].mean(),
                    'mean_aoi_50': layer_subset['aoi_50'].mean()
                }
            analysis[label]['per_layer'] = layer_stats
            
        return analysis
    
    def compute_clinical_relevance_score(self, validation_df: pd.DataFrame) -> Dict:
        """
        Compute overall clinical relevance scores.
        
        Args:
            validation_df: Validation results DataFrame
            
        Returns:
            Clinical relevance scores
        """
        scores = {}
        
        # Overall scores
        scores['overall'] = {
            'mean_iou': validation_df['iou'].mean(),
            'median_iou': validation_df['iou'].median(),
            'pointing_game_accuracy': validation_df['pointing_game_accuracy'].mean(),
            'high_iou_proportion': (validation_df['iou'] > 0.5).mean()
        }
        
        # Per-layer scores
        scores['per_layer'] = {}
        for layer in validation_df['target_layer'].unique():
            layer_df = validation_df[validation_df['target_layer'] == layer]
            scores['per_layer'][layer] = {
                'mean_iou': layer_df['iou'].mean(),
                'pointing_game_accuracy': layer_df['pointing_game_accuracy'].mean()
            }
            
        # Per-class scores
        scores['per_class'] = {}
        for class_idx in validation_df['class_idx'].unique():
            class_df = validation_df[validation_df['class_idx'] == class_idx]
            class_name = self.gradcam_model.model.class_names.get(class_idx, f"Class_{class_idx}")
            scores['per_class'][class_name] = {
                'mean_iou': class_df['iou'].mean(),
                'pointing_game_accuracy': class_df['pointing_game_accuracy'].mean(),
                'n_samples': len(class_df)
            }
            
        return scores