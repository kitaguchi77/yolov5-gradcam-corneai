"""
Image Preprocessing Module

This module provides preprocessing functions for YOLOv5 inference
and GradCAM++ analysis.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Union, List
import torch
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Handles image preprocessing for YOLOv5 models.
    """
    
    def __init__(self, img_size: int = 640, 
                device: str = 'cpu',
                stride: int = 32):
        """
        Initialize preprocessor.
        
        Args:
            img_size: Target image size
            device: Device for tensor operations
            stride: Model stride for size adjustment
        """
        self.img_size = img_size
        self.device = torch.device(device)
        self.stride = stride
        
    def letterbox(self, img: np.ndarray, 
                 new_shape: Union[int, Tuple[int, int]] = 640,
                 color: Tuple[int, int, int] = (114, 114, 114),
                 auto: bool = True,
                 scaleFill: bool = False,
                 scaleup: bool = True) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
        """
        Resize and pad image while meeting stride-multiple constraints.
        
        Args:
            img: Image to process
            new_shape: Target shape (height, width)
            color: Padding color
            auto: Minimum rectangle mode
            scaleFill: Stretch mode
            scaleup: Allow scaling up
            
        Returns:
            Tuple of (processed_image, scale_ratios, padding)
        """
        shape = img.shape[:2]  # Current shape [height, width]
        
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
            
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # Only scale down, do not scale up
            r = min(r, 1.0)
            
        # Compute padding
        ratio = r, r  # Width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        
        if auto:  # Minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
        elif scaleFill:  # Stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
            
        dw /= 2  # Divide padding into 2 sides
        dh /= 2
        
        if shape[::-1] != new_unpad:  # Resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        # Add border
        img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                               cv2.BORDER_CONSTANT, value=color)
        
        return img, ratio, (dw, dh)
    
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image],
                        augment: bool = False) -> torch.Tensor:
        """
        Preprocess image for YOLOv5 inference.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            augment: Whether to apply augmentation
            
        Returns:
            Preprocessed tensor
        """
        # Load image if path
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image: {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            img = np.array(image)
            if img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        else:
            img = image.copy()
            
        # Store original shape
        self.original_shape = img.shape[:2]
        
        # Letterbox
        img, self.ratio, self.pad = self.letterbox(img, self.img_size)
        
        # Convert to tensor
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img).to(self.device)
        img_tensor = img_tensor.float() / 255.0  # 0-255 to 0.0-1.0
        
        # Add batch dimension
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)
            
        return img_tensor
    
    def postprocess_bbox(self, bbox: np.ndarray) -> np.ndarray:
        """
        Convert bounding box from letterboxed to original coordinates.
        
        Args:
            bbox: Bounding box in letterboxed coordinates [x1, y1, x2, y2]
            
        Returns:
            Bounding box in original coordinates
        """
        # Remove padding
        bbox[0] -= self.pad[0]  # x1
        bbox[1] -= self.pad[1]  # y1
        bbox[2] -= self.pad[0]  # x2
        bbox[3] -= self.pad[1]  # y2
        
        # Scale to original size
        bbox /= self.ratio[0]
        
        # Clip to image boundaries
        bbox[0] = max(0, bbox[0])
        bbox[1] = max(0, bbox[1])
        bbox[2] = min(self.original_shape[1], bbox[2])
        bbox[3] = min(self.original_shape[0], bbox[3])
        
        return bbox
    
    def normalize_image(self, img: np.ndarray,
                       mean: Optional[List[float]] = None,
                       std: Optional[List[float]] = None) -> np.ndarray:
        """
        Normalize image using mean and std.
        
        Args:
            img: Input image
            mean: Mean values for each channel
            std: Standard deviation values for each channel
            
        Returns:
            Normalized image
        """
        if mean is None:
            mean = [0.485, 0.456, 0.406]  # ImageNet mean
        if std is None:
            std = [0.229, 0.224, 0.225]  # ImageNet std
            
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std
        
        return img
    
    def augment_image(self, img: np.ndarray,
                     augmentations: Optional[List[str]] = None) -> List[np.ndarray]:
        """
        Apply augmentations to image.
        
        Args:
            img: Input image
            augmentations: List of augmentation types
            
        Returns:
            List of augmented images
        """
        if augmentations is None:
            augmentations = ['flip_lr', 'flip_ud', 'rotate90']
            
        augmented = [img]  # Include original
        
        for aug in augmentations:
            if aug == 'flip_lr':
                augmented.append(cv2.flip(img, 1))
            elif aug == 'flip_ud':
                augmented.append(cv2.flip(img, 0))
            elif aug == 'rotate90':
                augmented.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
            elif aug == 'rotate180':
                augmented.append(cv2.rotate(img, cv2.ROTATE_180))
            elif aug == 'rotate270':
                augmented.append(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))
                
        return augmented
    
    def resize_cam(self, cam: np.ndarray,
                  target_shape: Tuple[int, int],
                  remove_padding: bool = True) -> np.ndarray:
        """
        Resize CAM heatmap to original image size.
        
        Args:
            cam: CAM heatmap
            target_shape: Target shape (height, width)
            remove_padding: Whether to remove letterbox padding
            
        Returns:
            Resized CAM
        """
        if remove_padding and hasattr(self, 'pad'):
            # Calculate unpadded size
            h_unpad = int(target_shape[0] * self.ratio[0])
            w_unpad = int(target_shape[1] * self.ratio[1])
            
            # Resize to unpadded size
            cam_resized = cv2.resize(cam, (w_unpad, h_unpad))
            
            # Create output array
            cam_output = np.zeros(target_shape, dtype=cam.dtype)
            
            # Calculate position in original image
            top = int(self.pad[1])
            left = int(self.pad[0])
            
            # Place resized CAM
            cam_output[:h_unpad, :w_unpad] = cam_resized
            
        else:
            # Simple resize
            cam_output = cv2.resize(cam, (target_shape[1], target_shape[0]))
            
        return cam_output
    
    def batch_preprocess(self, images: List[Union[str, np.ndarray]],
                        batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Preprocess multiple images into batches.
        
        Args:
            images: List of images
            batch_size: Maximum batch size
            
        Returns:
            Batched tensor
        """
        preprocessed = []
        
        for img in images:
            img_tensor = self.preprocess_image(img)
            preprocessed.append(img_tensor)
            
        # Stack into batch
        batch = torch.cat(preprocessed, dim=0)
        
        return batch
    
    def extract_roi(self, img: np.ndarray,
                   bbox: Union[List, np.ndarray],
                   padding: int = 0) -> np.ndarray:
        """
        Extract region of interest from image.
        
        Args:
            img: Input image
            bbox: Bounding box [x1, y1, x2, y2]
            padding: Additional padding around bbox
            
        Returns:
            Extracted ROI
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img.shape[1], x2 + padding)
        y2 = min(img.shape[0], y2 + padding)
        
        roi = img[y1:y2, x1:x2]
        
        return roi