import pandas as pd
import numpy as np
import cv2
import torch
import os
import re
from tqdm import tqdm
import warnings
import argparse
import sys
from pathlib import Path

# Add yolov5-gradcam directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5-gradcam'))

# YOLOv5 related imports
from utils.general import non_max_suppression as yolo_nms, xywh2xyxy
from utils.datasets import letterbox
from models.experimental import attempt_load
from deep_utils.utils.box_utils.boxes import Box

# Custom class imports
from models.yolo import Model, Detect
from models.common import Conv, Bottleneck, C3, SPPF, Concat
from torch.nn import Sequential, Conv2d, BatchNorm2d, SiLU, MaxPool2d, Upsample, ModuleList

warnings.filterwarnings("ignore", category=FutureWarning)

torch.serialization.add_safe_globals([
    Model, Detect, Sequential, ModuleList, Conv, Bottleneck, C3, SPPF, Concat,
    Conv2d, BatchNorm2d, SiLU, MaxPool2d, Upsample
])

from yolo_detector import YOLOV5TorchObjectDetector

def extract_ground_truth(basename):
    """Extracts the ground truth label from the filename using regex
    and maps it to the model's class name.
    Expected filename format: [class_name]rest_of_filename.jpg
    """
    pattern = r'\[([^\]]+)\]' # Regex to find content inside square brackets
    matches = re.findall(pattern, basename)

    if not matches:
        # If no bracketed class name found, try to use the whole basename (without extension)
        extracted_label = os.path.splitext(basename)[0]
    else:
        # Use the first match found inside brackets
        extracted_label = matches[0]

    # Convert extracted label to lowercase for consistent mapping key
    extracted_label_lower = extracted_label.lower()

    # Mapping from extracted label (lowercase) to the exact string in model's class_names list
    label_mapping = {
        'infection': 'infection',
        'normal': 'normal',
        'immun': 'non-infection', # 'immun' from filename maps to 'non-infection'
        'scar': 'scar',
        'tumor': 'tumor',
        'deposit': 'deposit',
        'apac': 'APAC', # 'apac' from filename maps to 'APAC' (model's case)
        'cat': 'lens opacity', # 'cat' from filename maps to 'lens opacity'
        'bullous': 'bullous'
    }

    # Get the mapped name. If not found in mapping, return the extracted label as is
    return label_mapping.get(extracted_label_lower, extracted_label)

def main(args):
    print(f"Reading images from: {args.image_dir}")
    print(f"Loading model from: {args.model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_names = ["infection", "normal", "non-infection", "scar", "tumor",
                   "deposit", "APAC", "lens opacity", "bullous"]

    model = YOLOV5TorchObjectDetector(
        args.model_path, device, img_size=(640, 640), names=class_names
    )

    image_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    results = []

    for filename in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(args.image_dir, filename)
        ground_truth = extract_ground_truth(filename)
        
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}")
                continue

            torch_img = model.preprocessing(img[..., ::-1])
            
            with torch.no_grad():
                [_, _, class_names_pred, confidences], _ = model(torch_img)

                if len(class_names_pred[0]) > 0:
                    confidences_list = confidences[0]
                    max_conf_idx = np.argmax(confidences_list)
                    predicted_class = class_names_pred[0][max_conf_idx]
                    likelihood = confidences_list[max_conf_idx]
                else:
                    predicted_class = None
                    likelihood = None
            
            results.append({
                'image_basename': filename, # Use the full filename including extension
                'GroundTruth': ground_truth,
                'Predict': predicted_class,
                'Likelihood': likelihood
            })

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    df = pd.DataFrame(results)
    
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nResults saved to: {output_path}")
    print("\n=== Statistics ===")
    print(f"Total images processed: {len(df)}")
    print(f"GroundTruth found for: {df['GroundTruth'].notna().sum()} images")
    print(f"Predictions made for: {df['Predict'].notna().sum()} images")
    
    if df['GroundTruth'].notna().sum() > 0 and df['Predict'].notna().sum() > 0:
        # Strict comparison
        accuracy = (df['GroundTruth'] == df['Predict']).mean()
        print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv5 inference and save predictions to a CSV file.")
    parser.add_argument('--image-dir', type=str, default='data/sample_images', help='Path to the directory containing images.')
    parser.add_argument('--model-path', type=str, default='models/last.pt', help='Path to the YOLOv5 model weights.')
    parser.add_argument('--output-csv', type=str, default='analysis_results.csv', help='Path to save the output CSV file.')
    
    args = parser.parse_args()
    main(args)