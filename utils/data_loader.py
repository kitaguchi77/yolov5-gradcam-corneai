"""
Data Loader Module

This module handles loading and organizing data for YOLOv5 GradCAM++ analysis,
including images, annotations, and metadata.
"""

import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import yaml
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


@dataclass
class ImageData:
    """Data class for image information."""
    image_path: str
    label: int
    class_name: str
    metadata: Optional[Dict] = None
    

class DataLoader:
    """
    Handles loading and organizing data for analysis.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize data loader.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.images = []
        self.annotations = {}
        
        # Disease categories
        self.class_names = {
            0: "Normal",
            1: "Infectious keratitis",
            2: "Non-infectious keratitis", 
            3: "Scar",
            4: "Tumor",
            5: "Deposit",
            6: "APAC",
            7: "Lens opacity",
            8: "Bullous keratopathy"
        }
        
        # Reverse mapping
        self.class_to_idx = {v: k for k, v in self.class_names.items()}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
            
    def load_image_list(self, list_file: str, 
                       base_path: Optional[str] = None) -> List[ImageData]:
        """
        Load image list from CSV or JSON file.
        
        Args:
            list_file: Path to list file
            base_path: Base path to prepend to image paths
            
        Returns:
            List of ImageData objects
        """
        file_path = Path(list_file)
        images = []
        
        if file_path.suffix == '.csv':
            df = pd.read_csv(list_file)
            
            for _, row in df.iterrows():
                img_path = row['image_path']
                if base_path:
                    img_path = os.path.join(base_path, img_path)
                    
                # Handle label as either index or name
                if 'label' in row:
                    label = row['label']
                elif 'class_name' in row:
                    label = self.class_to_idx.get(row['class_name'], -1)
                else:
                    label = -1
                    
                class_name = self.class_names.get(label, 'Unknown')
                
                # Extract metadata if present
                metadata = {}
                for col in df.columns:
                    if col not in ['image_path', 'label', 'class_name']:
                        metadata[col] = row[col]
                        
                images.append(ImageData(
                    image_path=img_path,
                    label=label,
                    class_name=class_name,
                    metadata=metadata if metadata else None
                ))
                
        elif file_path.suffix == '.json':
            with open(list_file, 'r') as f:
                data = json.load(f)
                
            for item in data:
                img_path = item['image_path']
                if base_path:
                    img_path = os.path.join(base_path, img_path)
                    
                label = item.get('label', -1)
                class_name = self.class_names.get(label, 'Unknown')
                
                images.append(ImageData(
                    image_path=img_path,
                    label=label,
                    class_name=class_name,
                    metadata=item.get('metadata')
                ))
                
        self.images = images
        logger.info(f"Loaded {len(images)} images from {list_file}")
        
        return images
    
    def load_annotations(self, annotation_file: str, 
                        annotation_type: str = 'cornea') -> Dict:
        """
        Load annotations (cornea regions, expert masks, etc.).
        
        Args:
            annotation_file: Path to annotation file
            annotation_type: Type of annotation ('cornea', 'lesion', etc.)
            
        Returns:
            Dictionary of annotations
        """
        annotations = {}
        
        with open(annotation_file, 'r') as f:
            data = json.load(f)
            
        for item in data:
            image_path = item['image_path']
            
            if annotation_type == 'cornea':
                annotations[image_path] = {
                    'center_x': item['center_x'],
                    'center_y': item['center_y'],
                    'major_axis': item['major_axis'],
                    'minor_axis': item['minor_axis'],
                    'angle': item.get('angle', 0)
                }
            elif annotation_type == 'lesion':
                annotations[image_path] = {
                    'mask_path': item.get('mask_path'),
                    'polygon': item.get('polygon'),
                    'bbox': item.get('bbox'),
                    'annotator': item.get('annotator', 'unknown')
                }
                
        self.annotations[annotation_type] = annotations
        logger.info(f"Loaded {len(annotations)} {annotation_type} annotations")
        
        return annotations
    
    def get_image_batch(self, batch_size: int = 32, 
                       shuffle: bool = True) -> List[List[ImageData]]:
        """
        Get images in batches.
        
        Args:
            batch_size: Size of each batch
            shuffle: Whether to shuffle data
            
        Returns:
            List of batches
        """
        images = self.images.copy()
        
        if shuffle:
            np.random.shuffle(images)
            
        batches = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batches.append(batch)
            
        return batches
    
    def load_image(self, image_path: str, 
                  resize: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Load and optionally resize an image.
        
        Args:
            image_path: Path to image
            resize: Optional (width, height) to resize to
            
        Returns:
            Image array
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if resize:
            img = cv2.resize(img, resize)
            
        return img
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get distribution of classes in loaded data.
        
        Returns:
            Dictionary of class counts
        """
        distribution = {}
        
        for img_data in self.images:
            class_name = img_data.class_name
            distribution[class_name] = distribution.get(class_name, 0) + 1
            
        return distribution
    
    def filter_by_class(self, class_names: List[str]) -> List[ImageData]:
        """
        Filter images by class names.
        
        Args:
            class_names: List of class names to include
            
        Returns:
            Filtered list of ImageData
        """
        filtered = []
        
        for img_data in self.images:
            if img_data.class_name in class_names:
                filtered.append(img_data)
                
        return filtered
    
    def split_data(self, test_ratio: float = 0.2, 
                  stratify: bool = True,
                  random_seed: int = 42) -> Tuple[List[ImageData], List[ImageData]]:
        """
        Split data into train and test sets.
        
        Args:
            test_ratio: Ratio of test data
            stratify: Whether to stratify by class
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, test_data)
        """
        np.random.seed(random_seed)
        
        if stratify:
            # Group by class
            class_groups = {}
            for img_data in self.images:
                label = img_data.label
                if label not in class_groups:
                    class_groups[label] = []
                class_groups[label].append(img_data)
                
            train_data = []
            test_data = []
            
            # Split each class
            for label, group in class_groups.items():
                np.random.shuffle(group)
                n_test = int(len(group) * test_ratio)
                test_data.extend(group[:n_test])
                train_data.extend(group[n_test:])
                
        else:
            # Simple random split
            images = self.images.copy()
            np.random.shuffle(images)
            n_test = int(len(images) * test_ratio)
            test_data = images[:n_test]
            train_data = images[n_test:]
            
        return train_data, test_data
    
    def create_background_pool(self, min_confidence: float = 0.9,
                             model=None) -> List[Dict]:
        """
        Create pool of background images for cut-and-paste.
        
        Args:
            min_confidence: Minimum confidence threshold
            model: YOLOv5Model instance for validation
            
        Returns:
            List of background image info
        """
        if model is None:
            logger.warning("No model provided for background validation")
            return []
            
        backgrounds = []
        
        for img_data in tqdm(self.images, desc="Selecting backgrounds"):
            # Run inference
            results = model.predict(img_data.image_path)
            
            if len(results['scores']) > 0:
                max_score = results['scores'].max()
                pred_class = results['classes'][results['scores'].argmax()]
                
                # Check if meets criteria
                if max_score >= min_confidence and pred_class == img_data.label:
                    backgrounds.append({
                        'image_path': img_data.image_path,
                        'label': img_data.label,
                        'class_name': img_data.class_name,
                        'confidence': float(max_score),
                        'metadata': img_data.metadata
                    })
                    
        logger.info(f"Selected {len(backgrounds)} background images")
        return backgrounds
    
    def save_data_split(self, train_data: List[ImageData],
                       test_data: List[ImageData],
                       output_dir: str):
        """
        Save data split information to files.
        
        Args:
            train_data: Training data
            test_data: Test data  
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as CSV
        train_df = pd.DataFrame([
            {
                'image_path': d.image_path,
                'label': d.label,
                'class_name': d.class_name
            }
            for d in train_data
        ])
        test_df = pd.DataFrame([
            {
                'image_path': d.image_path,
                'label': d.label,
                'class_name': d.class_name
            }
            for d in test_data
        ])
        
        train_df.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)
        
        # Save summary
        summary = {
            'n_train': len(train_data),
            'n_test': len(test_data),
            'train_distribution': self._get_distribution(train_data),
            'test_distribution': self._get_distribution(test_data)
        }
        
        with open(os.path.join(output_dir, 'data_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Saved data split to {output_dir}")
        
    def _get_distribution(self, data: List[ImageData]) -> Dict[str, int]:
        """Get class distribution for a dataset."""
        dist = {}
        for d in data:
            dist[d.class_name] = dist.get(d.class_name, 0) + 1
        return dist