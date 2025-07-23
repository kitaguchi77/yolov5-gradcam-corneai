"""
YOLOv5 Model Wrapper for GradCAM++ Analysis

This module provides a wrapper class for YOLOv5 models to facilitate
GradCAM++ analysis and inference for anterior segment disease classification.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import yaml
from typing import Dict, List, Tuple, Optional, Union
import cv2
import logging

# Add YOLOv5 to path if using local clone
sys.path.append(str(Path(__file__).parent.parent / 'yolov5'))

logger = logging.getLogger(__name__)


class YOLOv5Model:
    """
    Wrapper class for YOLOv5 model with support for intermediate layer access
    and GradCAM++ analysis.
    """
    
    def __init__(self, weights_path: str, device: str = 'cpu', config_path: Optional[str] = None):
        """
        Initialize YOLOv5 model wrapper.
        
        Args:
            weights_path: Path to YOLOv5 weights file
            device: Device to run model on ('cpu' or 'cuda')
            config_path: Optional path to configuration file
        """
        self.weights_path = weights_path
        self.device = torch.device(device)
        self.config = self._load_config(config_path) if config_path else {}
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Store model info
        self.img_size = self.config.get('img_size', 640)
        self.conf_threshold = self.config.get('conf_threshold', 0.25)
        self.iou_threshold = self.config.get('iou_threshold', 0.45)
        
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
        
        # Layer mapping for GradCAM++
        self.target_layers = {
            '17': 'model.17',
            '20': 'model.20',
            '23': 'model.23',
            '24_m_0': 'model.24.m.0',
            '24_m_1': 'model.24.m.1',
            '24_m_2': 'model.24.m.2'
        }
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('model', {})
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            return {}
    
    def _load_model(self):
        """Load YOLOv5 model from weights."""
        try:
            # Try to import YOLOv5 functions
            from models.experimental import attempt_load
            from utils.general import check_img_size
            
            # Load model
            model = attempt_load(self.weights_path, map_location=self.device)
            
            # Check and adjust image size
            if hasattr(self, 'img_size'):
                self.img_size = check_img_size(self.img_size, s=model.stride.max())
            
            return model
            
        except ImportError:
            # Fallback: load with torch.hub
            logger.info("Loading model with torch.hub")
            model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                 path=self.weights_path, device=self.device)
            return model.model
        
    def preprocess_image(self, image: Union[str, np.ndarray]) -> torch.Tensor:
        """
        Preprocess image for YOLOv5 inference.
        
        Args:
            image: Path to image or numpy array
            
        Returns:
            Preprocessed image tensor
        """
        # Load image if path provided
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Store original shape
        self.original_shape = image.shape[:2]
        
        # Letterbox resize
        from utils.augmentations import letterbox
        img = letterbox(image, self.img_size, stride=32, auto=True)[0]
        
        # Convert to tensor
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0  # 0-255 to 0.0-1.0
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        return img
    
    def forward(self, x: torch.Tensor, need_features: bool = False) -> Union[torch.Tensor, Tuple]:
        """
        Forward pass through model.
        
        Args:
            x: Input tensor
            need_features: Whether to return intermediate features
            
        Returns:
            Model predictions (and features if requested)
        """
        if need_features:
            features = {}
            
            # Register hooks to capture intermediate outputs
            handles = []
            for layer_name, module_path in self.target_layers.items():
                module = self._get_module_by_name(module_path)
                handle = module.register_forward_hook(
                    lambda m, i, o, name=layer_name: features.update({name: o})
                )
                handles.append(handle)
            
            # Forward pass
            with torch.no_grad():
                output = self.model(x)
            
            # Remove hooks
            for handle in handles:
                handle.remove()
                
            return output, features
        else:
            with torch.no_grad():
                return self.model(x)
    
    def _get_module_by_name(self, module_name: str) -> nn.Module:
        """Get module by dot-separated name."""
        module = self.model
        for name in module_name.split('.'):
            if name.isdigit():
                module = module[int(name)]
            else:
                module = getattr(module, name)
        return module
    
    def predict(self, image: Union[str, np.ndarray], 
                return_features: bool = False) -> Dict:
        """
        Run inference on image.
        
        Args:
            image: Path to image or numpy array
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing predictions and optionally features
        """
        # Preprocess image
        img_tensor = self.preprocess_image(image)
        
        # Run inference
        if return_features:
            pred, features = self.forward(img_tensor, need_features=True)
        else:
            pred = self.forward(img_tensor)
            features = None
        
        # Apply NMS
        from utils.general import non_max_suppression
        pred = non_max_suppression(pred, self.conf_threshold, self.iou_threshold,
                                 classes=None, agnostic=False, max_det=1000)
        
        # Process predictions
        results = self._process_predictions(pred[0] if pred[0] is not None else torch.tensor([]))
        
        if return_features:
            results['features'] = features
            
        return results
    
    def _process_predictions(self, predictions: torch.Tensor) -> Dict:
        """Process model predictions into readable format."""
        results = {
            'boxes': [],
            'scores': [],
            'classes': [],
            'class_names': []
        }
        
        if len(predictions) == 0:
            return results
        
        # Convert to CPU numpy
        predictions = predictions.cpu().numpy()
        
        for pred in predictions:
            x1, y1, x2, y2, conf, cls = pred
            
            results['boxes'].append([x1, y1, x2, y2])
            results['scores'].append(conf)
            results['classes'].append(int(cls))
            results['class_names'].append(self.class_names.get(int(cls), f"Class_{int(cls)}"))
        
        # Convert to numpy arrays
        results['boxes'] = np.array(results['boxes'])
        results['scores'] = np.array(results['scores'])
        results['classes'] = np.array(results['classes'])
        
        return results
    
    def get_layer_names(self) -> List[str]:
        """Get names of target layers for GradCAM++."""
        return list(self.target_layers.keys())
    
    def get_module_for_gradcam(self, layer_name: str) -> nn.Module:
        """
        Get module for GradCAM++ analysis.
        
        Args:
            layer_name: Name of target layer
            
        Returns:
            PyTorch module
        """
        if layer_name not in self.target_layers:
            raise ValueError(f"Layer {layer_name} not found in target layers")
        
        module_path = self.target_layers[layer_name]
        return self._get_module_by_name(module_path)
    
    def set_grad_enabled(self, enabled: bool = True):
        """Enable or disable gradient computation."""
        for param in self.model.parameters():
            param.requires_grad = enabled