# YOLOv5 Grad-CAMによる角膜疾患分析

このリポジトリは、前眼部疾患分類のために訓練されたYOLOv5モデルに対し、Grad-CAM++による可視化を行うためのツールを提供します。これは、[pooya-mohammadi/yolov5-gradcam](https://github.com/pooya-mohammadi/yolov5-gradcam) の研究に基づいています。

## プロジェクト構成

```
/Code/
├── yolov5-gradcam/      # クローンしたリポジトリ
├── data/
│   ├── sample_images/   # テスト用のサンプル画像
│   └── annotations.xml  # Cut-and-Paste検証に必要
├── models/
│   └── last.pt          # YOLOv5モデルの重みファイル
├── 1_add_predictions.py
├── 2_calculate_aoi.py
├── 3_cut_and_paste.py          # 改善されたロジックでバッチCut-and-Paste検証を実行
└── README.md
└── README_ja.md
```

## セットアップ

1.  **リポジトリのクローンと移動:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name/Code
    ```

2.  **依存関係のインストール:**
    Python 3.8以上がインストールされていることを確認してください。その後、必要なパッケージをインストールします。
    ```bash
    pip install -r yolov5-gradcam/requirements.txt
    ```

3.  **必要なファイルの配置:**
    *   訓練済みのYOLOv5モデル (`last.pt`) を `models/` ディレクトリに配置してください。
    *   サンプル画像を `data/sample_images/` ディレクトリに配置してください。(`sample.py` を使用して生成することも可能です。)
    *   Cut-and-Paste検証を行うために、角膜のセグメンテーションデータが含まれる `annotations.xml` ファイルを `data/` ディレクトリに配置してください。

## 分析ワークフロー

分析は複数のステップに分かれています。`Code` ディレクトリ内から、以下の順序でスクリプトを実行してください。

### ステップ0: サンプル画像の準備 (オプション)

大規模なデータセットからサンプル画像を生成する必要がある場合は、`sample.py` を使用できます。このスクリプトは、指定されたCSVから各正解クラスごとに1枚の画像を抽出し、`data/sample_images` にコピーします。

```bash
python sample.py --seed 42 
```

### ステップ1: 予測結果の生成

このスクリプトは、指定されたディレクトリ内のすべての画像に対して推論を実行し、ファイル名から正解ラベルを抽出して、予測結果と信頼度をCSVファイルに保存します。

```bash
python 1_add_predictions.py \
    --image-dir data/sample_images \
    --model-path models/last.pt \
    --output-csv analysis_results.csv
```

### ステップ2: 注目領域(AOI)の計算

このスクリプトは、指定されたディレクトリ内のすべての画像ファイルを直接処理し、ネットワークの複数レイヤーに対してGrad-CAM++の注目領域(Area of Interest, AOI)を計算します。正解クラスは、ファイル名のプレフィックス（例: `[tumor]image.jpg`）から自動的に判定されます。AOIは、予測されたBBox（バウンディングボックス）の面積に対する、その内部にある注目領域（Saliency Mapでハイライトされた部分）の面積の割合 (`Intersection / BBox Area`) として計算され、モデルが対象物のどの程度に注目しているかをより直感的に評価できます。

```bash
python 2_calculate_aoi.py \
    --image-dir data/sample_images \
    --model-path models/last.pt \
    --output-csv analysis_results_with_aoi.csv \
    --save-maps \
    --maps-output-dir saliency_maps
```

### ステップ3: Cut-and-Paste検証の実行

このスクリプトは、画像ペアを生成し、バッチモードでCut-and-Paste検証を実行します。改善された合成画像生成ロジック（輝度マッチング、スムーズなブレンディング）に加え、アフィン変換は**貼り付けるオブジェクト（病変部など）の元の向き（回転）を維持する**ように改善されており、より現実的な合成画像を作成します。`annotations.xml` ファイルが必要です。


```bash
python 3_cut_and_paste.py \
    --image-dir data/sample_images \
    --annotations-xml data/annotations.xml \
    --output-csv batch_cut_and_paste_results.csv \
    --composite-output-dir data/composite_images \
    --model-path models/last.pt
```

これにより、合成画像は `composite_images_batch/` に保存され、各合成画像の予測結果を含むCSV (`batch_cut_and_paste_results.csv`) が出力されます。


### ステップ4: Pointing GameとIoUメトリクスの計算

このスクリプトは、生成されたGrad-CAM++のSaliency Mapを、専門家が作成した病変マスクと比較することで、モデルの注目の臨床的妥当性を評価します。以下の2つの主要なメトリクスを計算します。

*   **Pointing Game Accuracy**: Saliency Map上で最も注目度が高い点（"hottest" spot）が、専門家の病変アノテーション領域内に含まれているかどうかを判定します。
*   **Intersection over Union (IoU)**: Saliency Mapを50%の閾値で二値化した領域と、専門家のマスク領域との空間的な重複度を測定します。

この処理には、サンプル画像とファイル名が対応する専門家のアノテーションマスク（PNG形式）が格納されたディレクトリが必要です。

```bash
python 4_calculate_metrics.py \
    --image-dir data/sample_images \
    --mask-dir data/sample_masks \
    --model-path models/last.pt \
    --output-csv metrics_results.csv \
    --layer-name model_23_cv3_conv
```

```
