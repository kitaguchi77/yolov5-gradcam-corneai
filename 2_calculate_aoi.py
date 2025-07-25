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

# --- ヘルパー関数 ---
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

def aggressive_memory_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def extract_class_prefix_from_filename(filename):
    """ファイル名から [prefix] 形式のプレフィックスを抽出する"""
    match = re.match(r'\[(.*?)\]', filename)
    if match:
        return match.group(1)
    return None

# --- YOLOV5GradCAM クラス (変更なし) ---
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

# --- main関数 (CSV入力からディレクトリ走査へ変更) ---
def main(args):
    IMAGE_DIR = Path(args.image_dir)
    MODEL_PATH = Path(args.model_path)
    OUTPUT_CSV_PATH = Path(args.output_csv)
    
    DEVICE = "cpu"
    NAMES = ["infection","normal","non-infection","scar","deposit","APAC","lens opacity","bullous", "tumor"]
    THRESHOLD = 0.5

    # ▼▼▼ 追加: ファイル名のプレフィックスとクラス名のマッピング ▼▼▼
    label_mapping = {
        'infection': 'infection',
        'normal': 'normal',
        'immun': 'non-infection',
        'scar': 'scar',
        'tumor': 'tumor',
        'deposit': 'deposit',
        'apac': 'APAC',
        'cat': 'lens opacity',
        'bullous': 'bullous'
    }
    # ▲▲▲ 追加 ▲▲▲

    layers_to_process = [
        ("model_17_cv3_conv", f"AOI_{THRESHOLD}_layer17"),
        ("model_20_cv3_conv", f"AOI_{THRESHOLD}_layer20"),
        ("model_23_cv3_conv", f"AOI_{THRESHOLD}_layer23"),
        ("model_24_m_0", f"AOI_{THRESHOLD}_layer24_m_0"),
        ("model_24_m_1", f"AOI_{THRESHOLD}_layer24_m_1"),
        ("model_24_m_2", f"AOI_{THRESHOLD}_layer24_m_2")
    ]

    if args.save_maps:
        MAPS_OUTPUT_DIR = Path(args.maps_output_dir)
        MAPS_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        print(f"Saliency Mapの保存先: {MAPS_OUTPUT_DIR.resolve()}")

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"エラー: {IMAGE_DIR} に画像ファイルが見つかりません。")
        return
    print(f"{len(image_files)} 件の画像を {IMAGE_DIR} から発見しました。")
    
    results_data = []

    print(f"モデルを読み込みます: {MODEL_PATH}")
    detector = YOLOV5TorchObjectDetector(MODEL_PATH, DEVICE, img_size=(IMG_SIZE, IMG_SIZE), names=NAMES)

    for img_filename in tqdm(image_files, desc="AOI解析中"):
        img_path = IMAGE_DIR / img_filename
        
        if not img_path.exists():
            print(f"警告: ファイルが見つかりません {img_path}"); continue
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"警告: 画像を読み込めません {img_path}"); continue

        original_h, original_w, _ = img_bgr.shape
        img_rgb = img_bgr[..., ::-1]
        torch_img = detector.preprocessing(img_rgb)
        
        predictions, _ = detector(torch_img.clone())
        
        # ▼▼▼ 変更: クラス名のマッピングを適用 ▼▼▼
        file_prefix = extract_class_prefix_from_filename(img_filename)
        ground_truth_class = label_mapping.get(file_prefix, "unknown")
        
        result_row = {
            'image_basename': img_filename,
            'ground_truth_class': ground_truth_class,
            'predicted_class': 'N/A',
            'confidence': 0.0
        }
        # ▲▲▲ 変更 ▲▲▲

        if not predictions[0]:
            results_data.append(result_row)
            aggressive_memory_cleanup()
            continue

        boxes, classes, _, confidences = predictions
        max_conf_idx = np.argmax(confidences[0])
        class_idx = classes[0][max_conf_idx]
        box = boxes[0][max_conf_idx]
        
        result_row['predicted_class'] = NAMES[class_idx]
        result_row['confidence'] = float(confidences[0][max_conf_idx])

        for layer_name, col_name in layers_to_process:
            try:
                with YOLOV5GradCAM(model=detector, layer_name=layer_name, method="gradcampp") as saliency_method:
                    mask = saliency_method(torch_img, class_idx)
                
                if mask is not None:
                    saliency_map_resized = F.interpolate(mask, size=(original_h, original_w), mode="bilinear", align_corners=False)
                    saliency_map_np = saliency_map_resized.squeeze().cpu().numpy()
                    binary_saliency_mask = (saliency_map_np >= THRESHOLD).astype(np.uint8)
                    
                    x1, y1, x2, y2 = map(int, box)
                    bbox_mask = np.zeros((original_h, original_w), dtype=np.uint8)
                    cv2.rectangle(bbox_mask, (x1, y1), (x2, y2), 1, -1)

                    intersection = np.sum(binary_saliency_mask & bbox_mask)
                    bbox_area = np.sum(bbox_mask) # BBoxの面積を計算
                    aoi = intersection / bbox_area if bbox_area > 0 else 0 # 計算式を BBox面積基準に変更
                    result_row[col_name] = aoi

                    if args.save_maps:
                        heatmap = cv2.applyColorMap(np.uint8(255 * saliency_map_np), cv2.COLORMAP_JET)
                        blended_img = img_bgr.copy()
                        roi_slice = (slice(y1, y2), slice(x1, x2))
                        roi = blended_img[roi_slice]
                        heatmap_roi = heatmap[roi_slice]
                        
                        if roi.size > 0 and heatmap_roi.size > 0:
                            blended_roi = cv2.addWeighted(roi, 0.6, heatmap_roi, 0.4, 0)
                            blended_img[roi_slice] = blended_roi
                        
                        cv2.rectangle(blended_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        save_path = MAPS_OUTPUT_DIR / f"{img_path.stem}_{layer_name}.png"
                        cv2.imwrite(str(save_path), blended_img)
                else:
                    raise Exception("Mask generation failed")
            except Exception as e:
                print(f"Error processing {img_filename} for layer {layer_name}: {e}")
                continue
        
        results_data.append(result_row)
        aggressive_memory_cleanup()

    df_results = pd.DataFrame(results_data)
    df_results.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"✅ Processing complete. Results saved to {OUTPUT_CSV_PATH}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv5 Grad-CAM for AOI Analysis")
    parser.add_argument('--image-dir', type=str, required=True, help='Path to the directory containing images.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the YOLOv5 model weights file (.pt).')
    parser.add_argument('--output-csv', type=str, required=True, help='Path to save the output CSV file with AOI results.')
    parser.add_argument('--save-maps', action='store_true', help='Flag to save the blended saliency maps.')
    parser.add_argument('--maps-output-dir', type=str, default='saliency_maps', help='Directory to save the blended saliency maps.')
    
    args = parser.parse_args()
    
    main(args)
