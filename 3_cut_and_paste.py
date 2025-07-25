import torch
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import os
import argparse
import sys
from tqdm import tqdm
import unicodedata
import itertools

# --- Pythonにyolo_detector.pyの場所を教えるためのパス ---
# このパスが環境に合わせて正しいことを確認してください
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5-gradcam'))
from yolo_detector import YOLOV5TorchObjectDetector
# ----------------------------------------------------


TARGET_WIDTH = 640.0

def create_canonical_key(filename):
    if not isinstance(filename, str):
        return ""
    return unicodedata.normalize('NFC', filename).lower().strip()

def load_ellipse_data(xml_path):
    if not xml_path.exists():
        print(f"エラー: XMLファイルが見つかりません: {xml_path}", file=sys.stderr)
        return None
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ellipse_data = {}
    for image_node in root.findall('.//image'):
        name = image_node.get('name')
        original_width_str = image_node.get('width')
        ellipse = image_node.find('ellipse')
        if name and ellipse is not None and original_width_str:
            key = create_canonical_key(name)
            if not key: continue
            try:
                original_width = float(original_width_str)
                if original_width == 0:
                    print(f"警告: {name} のオリジナル幅が0です。スキップします。", file=sys.stderr)
                    continue
                scale_ratio = TARGET_WIDTH / original_width
                ellipse_data[key] = {
                    'cx': float(ellipse.get('cx')) * scale_ratio, 'cy': float(ellipse.get('cy')) * scale_ratio,
                    'rx': float(ellipse.get('rx')) * scale_ratio, 'ry': float(ellipse.get('ry')) * scale_ratio,
                    'rotation': float(ellipse.get('rotation', '0'))
                }
            except (ValueError, TypeError, AttributeError) as e:
                print(f"警告: {name} の情報が不正です。スキップします。エラー: {e}", file=sys.stderr)
                continue
    print(f"{len(ellipse_data)}件の楕円アノテーションをスケーリング・正規化して読み込みました。")
    return ellipse_data

def load_and_resize_image(path, target_width):
    img = cv2.imread(str(path))
    if img is None: return None
    h, w = img.shape[:2]
    if w == target_width: return img
    scale_ratio = target_width / w
    new_height = int(h * scale_ratio)
    return cv2.resize(img, (int(target_width), new_height), interpolation=cv2.INTER_AREA)

def get_ellipse_mean_brightness(image, ellipse_params):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    try:
        center = (int(ellipse_params['cx']), int(ellipse_params['cy']))
        axes = (int(ellipse_params['rx']), int(ellipse_params['ry']))
        rotation = ellipse_params['rotation']
        cv2.ellipse(mask, center, axes, rotation, 0, 360, 255, -1)
    except (KeyError, ValueError) as e:
        print(f"エラー: 楕円マスクの作成に失敗しました。パラメータ: {ellipse_params}, エラー: {e}", file=sys.stderr)
        return 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = cv2.mean(gray, mask=mask)[0]
    return mean_brightness

def create_composite_image(source_img, target_img, source_ellipse, target_ellipse):
    source_rgb = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
    target_rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    h, w = target_rgb.shape[:2]
    source_brightness = get_ellipse_mean_brightness(source_img, source_ellipse)
    target_brightness = get_ellipse_mean_brightness(target_img, target_ellipse)
    brightness_diff = source_brightness - target_brightness
    target_adjusted = np.clip(target_rgb.astype(np.float32) + brightness_diff, 0, 255).astype(np.uint8)
    
    # ターゲットの回転角度をソースの回転角度に合わせることで、向きを維持する
    target_ellipse['rotation'] = source_ellipse['rotation']
    
    src_angle_rad = np.deg2rad(source_ellipse['rotation'])
    src_cos, src_sin = np.cos(src_angle_rad), np.sin(src_angle_rad)
    src_pts = np.float32([[source_ellipse['cx'], source_ellipse['cy']], [source_ellipse['cx'] + source_ellipse['rx'] * src_cos, source_ellipse['cy'] + source_ellipse['rx'] * src_sin], [source_ellipse['cx'] - source_ellipse['ry'] * src_sin, source_ellipse['cy'] + source_ellipse['ry'] * src_cos]])
    tgt_angle_rad = np.deg2rad(target_ellipse['rotation'])
    tgt_cos, tgt_sin = np.cos(tgt_angle_rad), np.sin(tgt_angle_rad)
    tgt_pts = np.float32([[target_ellipse['cx'], target_ellipse['cy']], [target_ellipse['cx'] + target_ellipse['rx'] * tgt_cos, target_ellipse['cy'] + target_ellipse['rx'] * tgt_sin], [target_ellipse['cx'] - target_ellipse['ry'] * tgt_sin, target_ellipse['cy'] + target_ellipse['ry'] * tgt_cos]])
    M = cv2.getAffineTransform(src_pts, tgt_pts)
    source_warped = cv2.warpAffine(source_rgb, M, (w, h))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (int(target_ellipse['cx']), int(target_ellipse['cy'])), (int(target_ellipse['rx']), int(target_ellipse['ry'])), target_ellipse['rotation'], 0, 360, 255, -1)
    mask_blurred = cv2.GaussianBlur(mask, (21, 21), 10)
    mask_3ch = cv2.cvtColor(mask_blurred, cv2.COLOR_GRAY2BGR).astype('float32') / 255.0
    result_rgb = (mask_3ch * source_warped) + ((1 - mask_3ch) * target_adjusted)
    result_rgb = result_rgb.astype(np.uint8)
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

def draw_text_on_image(image, text, position=(10, 30), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, color=(255, 255, 255), thickness=2):
    img_with_text = image.copy()
    cv2.putText(img_with_text, text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    return img_with_text

def main(args):
    image_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    all_pairs = list(itertools.product(image_files, repeat=2))
    df_input = pd.DataFrame(all_pairs, columns=['source_image_basename', 'target_image_basename'])
    if df_input.empty:
        print(f"警告: {args.image_dir} に有効な画像ファイルが見つからないか、ペアが生成されませんでした。", file=sys.stderr)
        return
    results_data = []
    model = None
    if args.model_path:
        print("モデルをロード中...")
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            class_names = ["infection", "normal", "non-infection", "scar", "tumor",
                           "deposit", "APAC", "lens opacity", "bullous"]
            model = YOLOV5TorchObjectDetector(args.model_path, device, img_size=(640, 640), names=class_names)
            print(f"モデルロード完了。デバイス: {device}")
        except Exception as e:
            print(f"\nエラー: モデルの読み込み中に予期せぬ問題が発生しました: {e}")
            return
    composite_dir = Path(args.composite_output_dir)
    composite_dir.mkdir(parents=True, exist_ok=True)
    ellipse_data = load_ellipse_data(Path(args.annotations_xml))
    if ellipse_data is None:
        print("エラー: 楕円アノテーションの読み込みに失敗しました。", file=sys.stderr)
        return
    print(f"{len(df_input)} 件の画像ペアを処理します。")
    for index, row in tqdm(df_input.iterrows(), total=len(df_input), desc="Processing image pairs"):
        source_img_basename = row['source_image_basename']
        target_img_basename = row['target_image_basename']
        source_key = create_canonical_key(source_img_basename)
        target_key = create_canonical_key(target_img_basename)
        source_ellipse = ellipse_data.get(source_key)
        target_ellipse = ellipse_data.get(target_key)
        if source_ellipse is None or target_ellipse is None:
            continue
        source_img_path = Path(args.image_dir) / source_img_basename
        target_img_path = Path(args.image_dir) / target_img_basename
        source_img = load_and_resize_image(source_img_path, TARGET_WIDTH)
        target_img = load_and_resize_image(target_img_path, TARGET_WIDTH)
        if source_img is None or target_img is None:
            continue
        composite_image = create_composite_image(source_img, target_img, source_ellipse, target_ellipse)
        predicted_class = "N/A"
        likelihood = 0.0
        if model:
            torch_img = model.preprocessing(composite_image[..., ::-1])
            with torch.no_grad():
                # --- ▼▼▼ 修正箇所 ▼▼▼ ---
                # お客様のYOLOV5TorchObjectDetectorの独自の出力形式に合わせて、
                # 推論結果の受け取り方を元の形式に戻しました。
                [boxes, _, class_names_pred, confidences], _ = model(torch_img)
                # --- ▲▲▲ 修正箇所 ▲▲▲ ---

                if len(class_names_pred[0]) > 0:
                    max_conf_idx = np.argmax(confidences[0])
                    predicted_class = class_names_pred[0][max_conf_idx]
                    likelihood = float(confidences[0][max_conf_idx])

        display_text = f"Pred: {predicted_class} ({likelihood:.2f})"
        composite_image_with_text = draw_text_on_image(composite_image, display_text, position=(10, 30), color=(0, 255, 0))
        composite_filename = f"source_{Path(source_img_basename).stem}_on_target_{Path(target_img_basename).stem}.jpg"
        output_img_path = composite_dir / composite_filename
        cv2.imwrite(str(output_img_path), composite_image_with_text)
        results_data.append({
            'source_image_basename': source_img_basename, 'target_image_basename': target_img_basename,
            'composite_image_path': str(output_img_path), 'predicted_class': predicted_class,
            'likelihood': likelihood
        })
    df_results = pd.DataFrame(results_data)
    df_results.to_csv(args.output_csv, index=False)
    print(f"\n処理結果を保存しました: {args.output_csv}")
    print(f"合成画像を保存しました: {args.composite_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="複数の画像ペアに対してCut-and-Pasteを実行し、予測結果を保存します。")
    parser.add_argument('--image-dir', type=str, required=True, help='すべての画像が保存されているディレクトリへのパス')
    parser.add_argument('--annotations-xml', type=str, required=True, help='楕円情報を含むXMLファイルへのパス')
    parser.add_argument('--output-csv', type=str, default='batch_cut_and_paste_results.csv', 
                        help='処理結果を保存するCSVファイルへのパス')
    parser.add_argument('--composite-output-dir', type=str, default='composite_images_batch', 
                        help='合成画像を保存するディレクトリへのパス')
    parser.add_argument('--model-path', type=str, help='(オプション) 予測に使用するYOLOv5モデルの重みへのパス')
    args = parser.parse_args()
    main(args)