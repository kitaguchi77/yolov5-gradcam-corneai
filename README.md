# YOLOv5 Grad-CAM for Corneal Disease Analysis

This repository provides a tool to generate Grad-CAM++ visualizations for a YOLOv5 model trained for anterior segment disease classification. It is based on the work by [pooya-mohammadi/yolov5-gradcam](https://github.com/pooya-mohammadi/yolov5-gradcam).

## Project Structure

```
/Code/
├── yolov5-gradcam/      # Cloned repository
├── data/
│   ├── sample_images/   # Sample images for testing
│   └── annotations.xml  # Required for Cut-and-Paste
├── models/
│   └── last.pt          # YOLOv5 model weights
├── 1_add_predictions.py
├── 2_calculate_aoi.py
├── 3_cut_and_paste.py          # Performs batch Cut-and-Paste validation with improved logic
└── README.md
```

## Setup

1.  **Clone the repository and navigate to the `Code` directory:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name/Code
    ```

2.  **Install dependencies:**
    Make sure you have Python 3.8+ installed. Then, install the required packages:
    ```bash
    pip install -r yolov5-gradcam/requirements.txt
    ```

3.  **Place necessary files:**
    *   Place your trained YOLOv5 model (`last.pt`) in the `models/` directory.
    *   Place your sample images in the `data/sample_images/` directory. (You can use `sample.py` to generate these.)
    *   For the Cut-and-Paste analysis, place the `annotations.xml` file (containing cornea segmentation data) in the `data/` directory.

## Analysis Workflow

The analysis is divided into several steps. Run the scripts in the following order from within the `Code` directory.

### Step 1: Generate Predictions

This script runs inference on all images in the specified directory, extracts the ground truth label from the filename, and saves the predictions and likelihoods to a CSV file.

```bash
python 1_add_predictions.py \
    --image-dir data/sample_images \
    --model-path models/last.pt \
    --output-csv analysis_results.csv
```

### Step 2: Calculate Area of Interest (AOI)

This script processes all image files in a directory to calculate the Grad-CAM++ Area of Interest (AOI). The ground truth class is automatically determined from the filename prefix (e.g., `[tumor]image.jpg`). The AOI is calculated as the ratio of the highlighted saliency area within the predicted bounding box to the total area of the bounding box (`Intersection / BBox Area`). This provides a more intuitive measure of how much of the object of interest is being focused on by the model.

```bash
python 2_calculate_aoi.py     --image-dir data/sample_images     --model-path models/last.pt     --output-csv analysis_results_with_aoi.csv     --save-maps     --maps-output-dir output/saliency_maps
```

### Step 3: Perform Cut-and-Paste Validation

This script generates image pairs and performs Cut-and-Paste validation in batch mode. It incorporates improved composite image generation logic, including brightness matching, smooth blending, and an affine transformation that **preserves the original orientation (rotation) of the source object**, ensuring a more realistic composite. It requires the `annotations.xml` file.


```bash
python 3_cut_and_paste.py     --image-dir data/sample_images     --annotations-xml data/annotations.xml     --output-csv output/batch_cut_and_paste_results.csv     --composite-output-dir output/composite_images     --model-path models/last.pt
```

This will save the composite images to `composite_images_batch/` and output a CSV (`batch_cut_and_paste_results.csv`) with prediction results for each composite image.


### Step 4: Calculate Pointing Game and IoU Metrics

This script evaluates the clinical relevance of the model's attention by comparing the generated Grad-CAM++ saliency maps against expert-annotated lesion masks. It calculates two key metrics:

*   **Pointing Game Accuracy**: Checks if the "hottest" point (maximum activation) of the saliency map falls within the expert's annotated lesion area.
*   **Intersection over Union (IoU)**: Measures the spatial overlap between the binarized saliency map (thresholded at 50%) and the expert's mask.

This requires a directory of expert annotation masks (as PNG files) where filenames correspond to the sample images.

```bash
python 4_calculate_metrics.py \
    --image-dir data/sample_images \
    --mask-dir data/sample_masks \
    --model-path models/last.pt \
    --output-csv output/metrics_results.csv \
    --layer-name model_23_cv3_conv
```


