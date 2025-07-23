"""
YOLOv5 Grad-CAM++ Implementation

This module implements Grad-CAM++ for YOLOv5 models to visualize
which regions the model focuses on for disease classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import logging

logger = logging.getLogger(__name__)


class YOLOv5GradCAMPlusPlus:
    """
    Grad-CAM++ implementation for YOLOv5 models.
    
    This class generates attention heatmaps showing which regions of the input
    image contribute most to the model's predictions.
    """
    
    def __init__(self, model, target_layers: List[str], use_cuda: bool = False):
        """
        Initialize Grad-CAM++ for YOLOv5.
        
        Args:
            model: YOLOv5Model instance
            target_layers: List of layer names to analyze
            use_cuda: Whether to use CUDA acceleration
        """
        self.model = model
        self.target_layers = target_layers
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.model.to(self.device)
        
        # Storage for gradients and activations
        self.gradients = {}
        self.activations = {}
        
        # Hooks
        self.handles = []
        
        # Enable gradients
        self.model.set_grad_enabled(True)
        
    def _register_hooks(self, target_layer: str):
        """Register forward and backward hooks for a specific layer."""
        module = self.model.get_module_for_gradcam(target_layer)
        
        # Forward hook
        def forward_hook(module, input, output):
            self.activations[target_layer] = output.detach()
        
        # Backward hook  
        def backward_hook(module, grad_input, grad_output):
            self.gradients[target_layer] = grad_output[0].detach()
        
        # Register hooks
        fhook = module.register_forward_hook(forward_hook)
        bhook = module.register_full_backward_hook(backward_hook)
        
        self.handles.extend([fhook, bhook])
        
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        
    def _compute_gradcam_plusplus(self, gradients: torch.Tensor, 
                                  activations: torch.Tensor) -> np.ndarray:
        """
        Compute Grad-CAM++ from gradients and activations.
        
        Args:
            gradients: Gradient tensor
            activations: Activation tensor
            
        Returns:
            Grad-CAM++ heatmap
        """
        # Compute second-order gradients
        grad_2 = gradients.pow(2)
        grad_3 = grad_2 * gradients
        
        # Global sum pooling
        sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)
        eps = 1e-7
        aij = grad_2 / (2 * grad_2 + sum_activations * grad_3 + eps)
        
        # Apply ReLU to aij
        aij = F.relu(aij)
        
        # Compute weights
        weights = torch.sum(aij * F.relu(gradients), dim=(2, 3), keepdim=True)
        
        # Compute weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam_min = cam.min()
        cam_max = cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + eps)
        
        return cam.squeeze().cpu().numpy()
    
    def generate_cam(self, image: Union[str, np.ndarray], 
                     class_idx: int,
                     target_layer: str) -> Tuple[np.ndarray, Dict]:
        """
        Generate Grad-CAM++ heatmap for a specific class and layer.
        
        Args:
            image: Input image path or array
            class_idx: Target class index
            target_layer: Target layer name
            
        Returns:
            Tuple of (cam_heatmap, metadata_dict)
        """
        # Clear previous data
        self.gradients.clear()
        self.activations.clear()
        self._remove_hooks()
        
        # Register hooks for target layer
        self._register_hooks(target_layer)
        
        # Preprocess image
        img_tensor = self.model.preprocess_image(image)
        img_tensor = img_tensor.to(self.device)
        img_tensor.requires_grad = True
        
        # Forward pass
        output = self.model.model(img_tensor)
        
        # Get prediction scores for target class
        # For YOLOv5, we need to handle the multi-scale output
        if isinstance(output, (list, tuple)):
            # Concatenate predictions from all scales
            predictions = []
            for out in output:
                if len(out.shape) == 3:  # [batch, anchors, classes+5]
                    pred = out[..., 5 + class_idx]  # Get class score
                    predictions.append(pred.view(-1))
            score = torch.cat(predictions).max()
        else:
            # Single output
            score = output[0, :, 5 + class_idx].max()
        
        # Backward pass
        self.model.model.zero_grad()
        score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[target_layer]
        activations = self.activations[target_layer]
        
        # Compute Grad-CAM++
        cam = self._compute_gradcam_plusplus(gradients, activations)
        
        # Resize CAM to original image size
        if hasattr(self.model, 'original_shape'):
            h, w = self.model.original_shape
            cam = cv2.resize(cam, (w, h))
        
        # Clean up
        self._remove_hooks()
        
        # Compute metadata
        metadata = {
            'target_layer': target_layer,
            'class_idx': class_idx,
            'class_name': self.model.class_names.get(class_idx, f'Class_{class_idx}'),
            'max_activation': float(cam.max()),
            'mean_activation': float(cam.mean()),
            'aoi_50': self._compute_aoi(cam, threshold=0.5)
        }
        
        return cam, metadata
    
    def _compute_aoi(self, cam: np.ndarray, threshold: float = 0.5) -> float:
        """
        Compute Area of Interest (AOI) metric.
        
        Args:
            cam: CAM heatmap
            threshold: Threshold for AOI calculation
            
        Returns:
            AOI value (proportion of pixels above threshold)
        """
        cam_normalized = cam / (cam.max() + 1e-7)
        aoi = np.sum(cam_normalized > threshold) / cam_normalized.size
        return float(aoi)
    
    def generate_all_cams(self, image: Union[str, np.ndarray], 
                         class_idx: int) -> Dict[str, Tuple[np.ndarray, Dict]]:
        """
        Generate Grad-CAM++ for all target layers.
        
        Args:
            image: Input image
            class_idx: Target class index
            
        Returns:
            Dictionary mapping layer names to (cam, metadata) tuples
        """
        results = {}
        
        for layer in self.target_layers:
            try:
                cam, metadata = self.generate_cam(image, class_idx, layer)
                results[layer] = (cam, metadata)
            except Exception as e:
                logger.error(f"Error generating CAM for layer {layer}: {e}")
                
        return results
    
    def visualize_cam(self, image: Union[str, np.ndarray], 
                     cam: np.ndarray,
                     alpha: float = 0.5,
                     colormap: str = 'jet',
                     save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize CAM heatmap overlaid on original image.
        
        Args:
            image: Original image
            cam: CAM heatmap
            alpha: Overlay transparency
            colormap: Colormap for heatmap
            save_path: Optional path to save visualization
            
        Returns:
            Visualization array
        """
        # Load image if path
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image.copy()
            
        # Ensure image is uint8
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
            
        # Resize CAM to match image
        cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
        
        # Apply colormap
        if colormap == 'custom':
            # Custom blue-to-red colormap as mentioned in the paper
            colors = ['blue', 'cyan', 'yellow', 'red']
            n_bins = 256
            cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
        else:
            cmap = plt.get_cmap(colormap)
            
        # Convert CAM to RGB
        cam_colored = cmap(cam_resized)
        cam_colored = (cam_colored[:, :, :3] * 255).astype(np.uint8)
        
        # Overlay
        overlay = cv2.addWeighted(img, 1 - alpha, cam_colored, alpha, 0)
        
        # Save if requested
        if save_path:
            plt.figure(figsize=(10, 8))
            plt.imshow(overlay)
            plt.axis('off')
            plt.title('Grad-CAM++ Visualization')
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        return overlay
    
    def compute_statistics(self, all_cams: Dict[str, Tuple[np.ndarray, Dict]]) -> Dict:
        """
        Compute statistics across all layers.
        
        Args:
            all_cams: Dictionary of CAMs for all layers
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'layer_aoi_50': {},
            'layer_max_activation': {},
            'layer_mean_activation': {}
        }
        
        for layer, (cam, metadata) in all_cams.items():
            stats['layer_aoi_50'][layer] = metadata['aoi_50']
            stats['layer_max_activation'][layer] = metadata['max_activation']
            stats['layer_mean_activation'][layer] = metadata['mean_activation']
            
        return stats