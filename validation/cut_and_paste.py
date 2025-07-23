"""
Cut-and-Paste Validation for YOLOv5 Model

This module implements the cut-and-paste validation technique to assess
the model's reliance on contextual information vs. primary pathological features.
"""

import numpy as np
import cv2
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import pandas as pd
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CorneaAnnotation:
    """Data class for cornea annotation."""
    image_path: str
    center_x: float
    center_y: float
    major_axis: float
    minor_axis: float
    angle: float = 0.0  # Rotation angle for ellipse
    

class CutAndPasteValidator:
    """
    Cut-and-paste validation for evaluating model's context dependency.
    
    This class extracts corneal regions from images and systematically
    pastes them onto different backgrounds to test the model's robustness.
    """
    
    def __init__(self, model, background_criteria: Optional[Dict] = None):
        """
        Initialize cut-and-paste validator.
        
        Args:
            model: YOLOv5Model instance
            background_criteria: Criteria for selecting background templates
        """
        self.model = model
        
        # Default background selection criteria
        self.background_criteria = background_criteria or {
            'min_confidence': 0.9,
            'correct_prediction': True,
            'minimal_eyelid_overlap': True
        }
        
        self.backgrounds = []
        self.cornea_annotations = []
        
    def load_annotations(self, annotation_file: str):
        """
        Load cornea annotations from file.
        
        Args:
            annotation_file: Path to annotation file (JSON or CSV)
        """
        file_path = Path(annotation_file)
        
        if file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            for item in data:
                annotation = CorneaAnnotation(
                    image_path=item['image_path'],
                    center_x=item['center_x'],
                    center_y=item['center_y'],
                    major_axis=item['major_axis'],
                    minor_axis=item['minor_axis'],
                    angle=item.get('angle', 0.0)
                )
                self.cornea_annotations.append(annotation)
                
        elif file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                annotation = CorneaAnnotation(
                    image_path=row['image_path'],
                    center_x=row['center_x'],
                    center_y=row['center_y'],
                    major_axis=row['major_axis'],
                    minor_axis=row['minor_axis'],
                    angle=row.get('angle', 0.0)
                )
                self.cornea_annotations.append(annotation)
                
        logger.info(f"Loaded {len(self.cornea_annotations)} cornea annotations")
        
    def extract_cornea(self, image: np.ndarray, 
                      annotation: CorneaAnnotation) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract cornea region from image using elliptical mask.
        
        Args:
            image: Input image
            annotation: Cornea annotation
            
        Returns:
            Tuple of (cornea_region, mask)
        """
        h, w = image.shape[:2]
        
        # Create elliptical mask
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Ellipse parameters
        center = (int(annotation.center_x), int(annotation.center_y))
        axes = (int(annotation.major_axis / 2), int(annotation.minor_axis / 2))
        angle = annotation.angle
        
        # Draw filled ellipse
        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
        
        # Extract cornea region
        cornea = cv2.bitwise_and(image, image, mask=mask)
        
        # Get bounding box of ellipse for cropping
        x, y, bbox_w, bbox_h = cv2.boundingRect(mask)
        cornea_cropped = cornea[y:y+bbox_h, x:x+bbox_w]
        mask_cropped = mask[y:y+bbox_h, x:x+bbox_w]
        
        return cornea_cropped, mask_cropped
    
    def paste_cornea(self, background: np.ndarray, 
                    cornea: np.ndarray,
                    mask: np.ndarray,
                    target_annotation: CorneaAnnotation) -> np.ndarray:
        """
        Paste cornea onto background at specified location.
        
        Args:
            background: Background image
            cornea: Cornea region to paste
            mask: Cornea mask
            target_annotation: Target location annotation
            
        Returns:
            Composite image
        """
        composite = background.copy()
        
        # Calculate scaling factors
        scale_x = target_annotation.major_axis / cornea.shape[1]
        scale_y = target_annotation.minor_axis / cornea.shape[0]
        
        # Resize cornea and mask
        new_size = (int(target_annotation.major_axis), int(target_annotation.minor_axis))
        cornea_resized = cv2.resize(cornea, new_size, interpolation=cv2.INTER_CUBIC)
        mask_resized = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)
        
        # Calculate paste position
        center_x = int(target_annotation.center_x)
        center_y = int(target_annotation.center_y)
        x1 = max(0, center_x - new_size[0] // 2)
        y1 = max(0, center_y - new_size[1] // 2)
        x2 = min(composite.shape[1], x1 + new_size[0])
        y2 = min(composite.shape[0], y1 + new_size[1])
        
        # Adjust for image boundaries
        cornea_x1 = 0 if x1 >= 0 else -x1
        cornea_y1 = 0 if y1 >= 0 else -y1
        cornea_x2 = cornea_x1 + (x2 - x1)
        cornea_y2 = cornea_y1 + (y2 - y1)
        
        # Create ROI
        roi = composite[y1:y2, x1:x2]
        cornea_roi = cornea_resized[cornea_y1:cornea_y2, cornea_x1:cornea_x2]
        mask_roi = mask_resized[cornea_y1:cornea_y2, cornea_x1:cornea_x2]
        
        # Paste using mask
        mask_inv = cv2.bitwise_not(mask_roi)
        bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg = cv2.bitwise_and(cornea_roi, cornea_roi, mask=mask_roi)
        composite[y1:y2, x1:x2] = cv2.add(bg, fg)
        
        return composite
    
    def select_backgrounds(self, image_paths: List[str], 
                          labels: List[int]) -> List[Dict]:
        """
        Select suitable background images based on criteria.
        
        Args:
            image_paths: List of candidate image paths
            labels: Ground truth labels for images
            
        Returns:
            List of background dictionaries
        """
        backgrounds = []
        
        for img_path, label in tqdm(zip(image_paths, labels), 
                                   desc="Selecting backgrounds"):
            # Run inference
            results = self.model.predict(img_path)
            
            if len(results['classes']) == 0:
                continue
                
            # Check criteria
            max_score = results['scores'].max()
            pred_class = results['classes'][results['scores'].argmax()]
            
            if (max_score >= self.background_criteria['min_confidence'] and
                pred_class == label):
                
                backgrounds.append({
                    'image_path': img_path,
                    'true_class': label,
                    'confidence': max_score,
                    'class_name': self.model.class_names[label]
                })
                
        logger.info(f"Selected {len(backgrounds)} background images")
        return backgrounds
    
    def validate_single_image(self, source_image_path: str,
                            source_annotation: CorneaAnnotation,
                            source_label: int,
                            backgrounds: List[Dict]) -> Dict:
        """
        Validate a single image against all backgrounds.
        
        Args:
            source_image_path: Path to source image
            source_annotation: Cornea annotation for source
            source_label: True class of source image
            backgrounds: List of background images
            
        Returns:
            Validation results dictionary
        """
        # Load source image
        source_img = cv2.imread(source_image_path)
        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
        
        # Extract cornea
        cornea, mask = self.extract_cornea(source_img, source_annotation)
        
        # Test on each background
        results = {
            'source_path': source_image_path,
            'source_label': source_label,
            'background_results': []
        }
        
        for bg_info in backgrounds:
            # Load background
            bg_img = cv2.imread(bg_info['image_path'])
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
            
            # Get background annotation
            bg_annotation = next((a for a in self.cornea_annotations 
                                if a.image_path == bg_info['image_path']), None)
            
            if bg_annotation is None:
                continue
                
            # Create composite
            composite = self.paste_cornea(bg_img, cornea, mask, bg_annotation)
            
            # Run inference on composite
            pred_results = self.model.predict(composite)
            
            if len(pred_results['classes']) > 0:
                pred_class = pred_results['classes'][pred_results['scores'].argmax()]
                pred_score = pred_results['scores'].max()
            else:
                pred_class = -1
                pred_score = 0.0
                
            results['background_results'].append({
                'background_class': bg_info['true_class'],
                'predicted_class': pred_class,
                'confidence': pred_score,
                'correct': pred_class == source_label
            })
            
        return results
    
    def run_validation(self, test_images: List[str],
                      test_labels: List[int],
                      output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Run complete cut-and-paste validation.
        
        Args:
            test_images: List of test image paths
            test_labels: List of ground truth labels
            output_path: Optional path to save results
            
        Returns:
            Results DataFrame
        """
        all_results = []
        
        # Process each test image
        for img_path, label in tqdm(zip(test_images, test_labels),
                                   desc="Running cut-and-paste validation"):
            # Find annotation for this image
            annotation = next((a for a in self.cornea_annotations 
                             if a.image_path == img_path), None)
            
            if annotation is None:
                logger.warning(f"No annotation found for {img_path}")
                continue
                
            # Validate against all backgrounds
            results = self.validate_single_image(
                img_path, annotation, label, self.backgrounds
            )
            
            # Aggregate results
            for bg_result in results['background_results']:
                all_results.append({
                    'source_image': img_path,
                    'source_class': label,
                    'source_class_name': self.model.class_names[label],
                    'background_class': bg_result['background_class'],
                    'background_class_name': self.model.class_names[bg_result['background_class']],
                    'predicted_class': bg_result['predicted_class'],
                    'predicted_class_name': self.model.class_names.get(bg_result['predicted_class'], 'Unknown'),
                    'confidence': bg_result['confidence'],
                    'correct': bg_result['correct']
                })
                
        # Convert to DataFrame
        df_results = pd.DataFrame(all_results)
        
        # Save if requested
        if output_path:
            df_results.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
            
        return df_results
    
    def compute_accuracy_matrix(self, results_df: pd.DataFrame) -> np.ndarray:
        """
        Compute accuracy matrix from validation results.
        
        Args:
            results_df: Results DataFrame
            
        Returns:
            Accuracy matrix (source_class x background_class)
        """
        n_classes = len(self.model.class_names)
        accuracy_matrix = np.zeros((n_classes, n_classes))
        
        for source_class in range(n_classes):
            for bg_class in range(n_classes):
                mask = ((results_df['source_class'] == source_class) & 
                       (results_df['background_class'] == bg_class))
                
                if mask.sum() > 0:
                    accuracy = results_df[mask]['correct'].mean()
                    accuracy_matrix[source_class, bg_class] = accuracy
                    
        return accuracy_matrix
    
    def analyze_context_dependency(self, results_df: pd.DataFrame) -> Dict:
        """
        Analyze context dependency for each disease class.
        
        Args:
            results_df: Results DataFrame
            
        Returns:
            Analysis results dictionary
        """
        analysis = {}
        
        for class_idx, class_name in self.model.class_names.items():
            class_results = results_df[results_df['source_class'] == class_idx]
            
            if len(class_results) == 0:
                continue
                
            # Accuracy on same background class
            same_bg = class_results[class_results['background_class'] == class_idx]
            same_bg_acc = same_bg['correct'].mean() if len(same_bg) > 0 else 0
            
            # Accuracy on different background classes  
            diff_bg = class_results[class_results['background_class'] != class_idx]
            diff_bg_acc = diff_bg['correct'].mean() if len(diff_bg) > 0 else 0
            
            # Context dependency score
            context_dependency = same_bg_acc - diff_bg_acc
            
            analysis[class_name] = {
                'same_background_accuracy': same_bg_acc,
                'different_background_accuracy': diff_bg_acc,
                'context_dependency_score': context_dependency,
                'n_samples': len(class_results)
            }
            
        return analysis