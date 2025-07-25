import os
import pandas as pd
import numpy as np
import cv2
import torch
import gc
import warnings
import sys
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
import argparse
import torch.nn as nn
import re

# --- UserWarningを抑制 ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- yolov5-gradcamへのパスを追加 ---
yolov5_gradcam_path = Path(__file__).parent / 'yolov5-gradcam'
if str(yolov5_gradcam_path) not in sys.path:
    sys.path.append(str(yolov5_gradcam_path))

# --- 外部のワーキングクラスをインポート ---
try:
    from yolo_detector import YOLOV5TorchObjectDetector
except ImportError:
    print("エラー: 'yolo_detector.py' が見つかりません。スクリプトと同じディレクトリに配置してください。")
    sys.exit(1)

# --- YOLOv5関連のインポート ---
from models.yolo import Model, Detect
from models.common import Conv, Bottleneck, C3, SPPF, Concat
from torch.nn import Sequential, ModuleList, Conv2d, BatchNorm2d, SiLU, MaxPool2d, Upsample

# --- 安全なクラスのリストを拡充 ---
torch.serialization.add_safe_globals([
    Model, Detect, Sequential, ModuleList, Conv, Bottleneck, C3, SPPF, Concat,
    Conv2d, BatchNorm2d, SiLU, MaxPool2d, Upsample
])

# --- グローバル変数 ---
IMG_SIZE = 640
SALIENCY_THRESHOLD = 0.5 # IoU計算のためのSaliency Mapの二値化閾値

# --- ▼▼▼ 2_calculate_aoi.pyから移植したコードブロック START ▼▼▼ ---
def find_yolo_layer(model, layer_name):
   """
   YOLOv5モデル内の特定のレイヤーを名前で検索します。
   "model_17_cv3_conv" のような名前を正しく解釈します。
   """
   hierarchy = layer_name.split("_")
   
   if hierarchy[0] == 'model' and len(hierarchy) > 1 and hierarchy[1].isdigit():
       layer_index = int(hierarchy[1])
       remaining_hierarchy = hierarchy[2:]
   else:
       raise ValueError(f"Unsupported layer name format: {layer_name}")

   yolo_model_module = model.model if hasattr(model, 'model') and isinstance(model.model, nn.Module) else model
   
   target_layer = yolo_model_module.model[layer_index]
   
   for h in remaining_hierarchy:
       if hasattr(target_layer, h):
           target_layer = getattr(target_layer, h)
       elif isinstance(target_layer, torch.nn.ModuleList) and h.isdigit():
           target_layer = target_layer[int(h)]
       else:
           target_layer = target_layer._modules[h]
           
   return target_layer

def set_model_gradients(model, layer_name, enable=True):
    """特定のレイヤーのみ勾配計算を有効化"""
    for param in model.parameters():
        param.requires_grad = False

    if enable:
        target_layer = find_yolo_layer(model, layer_name)
        for param in target_layer.parameters():
            param.requires_grad = True

class YOLOV5GradCAM:
    def __init__(self, model, layer_name, method="gradcampp"):
        self.model = model
        self.gradients = dict()
        self.activations = dict()
        self.method = method
        self.layer_name = layer_name
        self.forward_handle = None
        self.backward_handle = None
        self.target_layer = None

        def backward_hook(module, grad_input, grad_output):
            self.gradients["value"] = grad_output[0].detach().clone()
            return None

        def forward_hook(module, input, output):
            self.activations["value"] = output.detach().clone()
            return None

        self.target_layer = find_yolo_layer(self.model, layer_name)
        set_model_gradients(self.model, layer_name, enable=True)
        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_backward_hook(backward_hook)

    def cleanup(self):
        if self.forward_handle is not None: self.forward_handle.remove()
        if self.backward_handle is not None: self.backward_handle.remove()
        self.gradients.clear()
        self.activations.clear()
        set_model_gradients(self.model, self.layer_name, enable=False)
        self.target_layer = None

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.cleanup()
    def __del__(self):
        try: self.cleanup()
        except: pass

    def _gradcampp_weights(self, gradients, activations, score, b, k, u, v):
        alpha_num = gradients.pow(2)
        alpha_denom = gradients.pow(2).mul(2) + \
            activations.mul(gradients.pow(3)).view(b, k, u*v).sum(-1, keepdim=True).view(b, k, 1, 1)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alpha = alpha_num.div(alpha_denom + 1e-7)
        relu_grad = F.relu(score.exp() * gradients)
        weights = (relu_grad * alpha).view(b, k, u*v).sum(-1).view(b, k, 1, 1)
        return weights

    def __call__(self, input_img, class_idx):
        self.model.zero_grad()
        _, logits = self.model(input_img.clone())
        score = logits[0][0][class_idx]
        score.backward()

        if "value" not in self.gradients or "value" not in self.activations:
            return None

        gradients = self.gradients["value"]
        activations = self.activations["value"]
        b, k, u, v = gradients.size()

        if self.method == "gradcampp":
            weights = self._gradcampp_weights(gradients, activations, score, b, k, u, v)
        else: # gradcam
            weights = gradients.view(b, k, -1).mean(2).view(b, k, 1, 1)

        with torch.no_grad():
            saliency_map = (weights * activations).sum(1, keepdim=True)
            saliency_map = F.relu(saliency_map)
            saliency_map = F.interpolate(saliency_map, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
            saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
            saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min + 1e-8).data
        
        return saliency_map
# --- ▲▲▲ 2_calculate_aoi.pyから移植したコードブロック END ▲▲▲ ---


# --- ヘルパー関数 ---
def aggressive_memory_cleanup():
    """メモリを積極的に解放する"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def extract_class_prefix_from_filename(filename):
    """ファイル名から [prefix] 形式のプレフィックスを抽出する"""
    match = re.match(r'\[(.*?)\]', filename)
    if match:
        return match.group(1)
    return None

def run_pointing_game(saliency_map, expert_mask):
    """
    Pointing Gameを実行する。
    saliency_mapで最も値が高い点がexpert_maskに含まれていればTrueを返す。
    """
    if saliency_map is None or expert_mask is None:
        return False
    
    # 最も注目されている点の座標を見つける
    _, _, _, max_loc = cv2.minMaxLoc(saliency_map)
    
    # その点が専門家のマスク領域（赤色=[0,0,255] in BGR）に含まれているか確認
    if expert_mask[max_loc[1], max_loc[0]][2] == 255: # BGRのRedチャンネルを確認
        return True
        
    return False

def calculate_iou(saliency_map, expert_mask, threshold=0.5):
    """
    Saliency Mapと専門家マスク間のIoUを計算する。
    """
    if saliency_map is None or expert_mask is None:
        return 0.0

    # Saliency Mapを二値化
    binary_saliency_mask = (saliency_map >= threshold).astype(np.uint8)
    
    # 専門家マスクを二値化 (赤色領域を1、その他を0)
    binary_expert_mask = (expert_mask[:, :, 2] == 255).astype(np.uint8)

    # IntersectionとUnionを計算
    intersection = np.logical_and(binary_saliency_mask, binary_expert_mask).sum()
    union = np.logical_or(binary_saliency_mask, binary_expert_mask).sum()
    
    iou = intersection / union if union > 0 else 0.0
    return iou

# --- main関数 ---
def main(args):
    IMAGE_DIR = Path(args.image_dir)
    MASK_DIR = Path(args.mask_dir)
    MODEL_PATH = Path(args.model_path)
    OUTPUT_CSV_PATH = Path(args.output_csv)
    LAYER_NAME = args.layer_name
    
    DEVICE = "cpu"
    NAMES = ["infection","normal","non-infection","scar","deposit","APAC","lens opacity","bullous", "tumor"]

    label_mapping = {
        'infection': 'infection', 'normal': 'normal', 'immun': 'non-infection',
        'scar': 'scar', 'tumor': 'tumor', 'deposit': 'deposit', 'apac': 'APAC',
        'cat': 'lens opacity', 'bullous': 'bullous'
    }

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"エラー: {IMAGE_DIR} に画像ファイルが見つかりません。")
        return
    print(f"{len(image_files)} 件の画像を {IMAGE_DIR} から発見しました。")
    
    results_data = []

    print(f"モデルを読み込みます: {MODEL_PATH}")
    detector = YOLOV5TorchObjectDetector(MODEL_PATH, DEVICE, img_size=(IMG_SIZE, IMG_SIZE), names=NAMES)

    for img_filename in tqdm(image_files, desc=f"Metrics Calculation (Layer: {LAYER_NAME})"):
        img_path = IMAGE_DIR / img_filename
        mask_path = MASK_DIR / Path(img_filename).with_suffix('.png')

        if not img_path.exists() or not mask_path.exists():
            print(f"警告: 画像またはマスクファイルが見つかりません。スキップします: {img_filename}")
            continue
            
        img_bgr = cv2.imread(str(img_path))
        expert_mask = cv2.imread(str(mask_path))
        if img_bgr is None or expert_mask is None:
            print(f"警告: 画像またはマスクを読み込めません。スキップします: {img_filename}")
            continue

        original_h, original_w, _ = img_bgr.shape
        expert_mask = cv2.resize(expert_mask, (original_w, original_h))

        img_rgb = img_bgr[..., ::-1]
        torch_img = detector.preprocessing(img_rgb)
        
        predictions, _ = detector(torch_img.clone())
        
        file_prefix = extract_class_prefix_from_filename(img_filename)
        ground_truth_class = label_mapping.get(file_prefix, "unknown")
        
        result_row = {
            'image_basename': img_filename,
            'ground_truth_class': ground_truth_class,
            'predicted_class': 'N/A',
            'confidence': 0.0,
            'pointing_game_accuracy': False,
            'iou': 0.0
        }

        if not predictions[0]:
            results_data.append(result_row)
            aggressive_memory_cleanup()
            continue

        boxes, classes, _, confidences = predictions
        max_conf_idx = np.argmax(confidences[0])
        class_idx = classes[0][max_conf_idx]
        
        result_row['predicted_class'] = NAMES[class_idx]
        result_row['confidence'] = float(confidences[0][max_conf_idx])

        saliency_map_np = None
        try:
            with YOLOV5GradCAM(model=detector, layer_name=LAYER_NAME, method="gradcampp") as saliency_method:
                mask = saliency_method(torch_img, class_idx)
            
            if mask is not None:
                saliency_map_resized = F.interpolate(mask, size=(original_h, original_w), mode="bilinear", align_corners=False)
                saliency_map_np = saliency_map_resized.squeeze().cpu().numpy()
            else:
                raise Exception("Saliency Map generation failed")

        except Exception as e:
            print(f"Error generating Saliency Map for {img_filename}: {e}")
            results_data.append(result_row)
            aggressive_memory_cleanup()
            continue

        result_row['pointing_game_accuracy'] = run_pointing_game(saliency_map_np, expert_mask)
        result_row['iou'] = calculate_iou(saliency_map_np, expert_mask, threshold=SALIENCY_THRESHOLD)
        
        results_data.append(result_row)
        aggressive_memory_cleanup()

    df_results = pd.DataFrame(results_data)
    df_results.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"✅ Processing complete. Results saved to {OUTPUT_CSV_PATH}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate Pointing Game and IoU metrics using YOLOv5 Grad-CAM++.")
    parser.add_argument('--image-dir', type=str, required=True, help='Path to the directory containing images.')
    parser.add_argument('--mask-dir', type=str, required=True, help='Path to the directory containing expert annotation masks.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the YOLOv5 model weights file (.pt).')
    parser.add_argument('--output-csv', type=str, required=True, help='Path to save the output CSV file.')
    parser.add_argument('--layer-name', type=str, default='model_23_cv3_conv', help='The target layer for Grad-CAM++.')
    
    args = parser.parse_args()
    main(args)
