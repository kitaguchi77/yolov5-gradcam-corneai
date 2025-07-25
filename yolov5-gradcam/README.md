# YOLOv5 Grad-CAM for Corneal Disease Analysis

This repository provides a tool to generate Grad-CAM++ visualizations for a YOLOv5 model trained for anterior segment disease classification. It is based on the work by [pooya-mohammadi/yolov5-gradcam](https://github.com/pooya-mohammadi/yolov5-gradcam).

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To generate a Grad-CAM++ visualization, use the `run_gradcam.py` script. 

### Arguments

*   `--model-path`: Path to the YOLOv5 model weights (`.pt` file).
*   `--img-path`: Path to the input image.
*   `--output-dir`: Directory to save the output images (default: `outputs`).
*   `--layer-name`: The name of the layer to visualize (default: `model_23_cv3_conv`).
*   `--img-size`: Image size for the model (default: 640).
*   `--method`: Visualization method, `gradcam` or `gradcampp` (default: `gradcampp`).
*   `--device`: `cpu` or `cuda` (default: `cpu`).

### Example

```bash
python run_gradcam.py \
    --model-path ../models/last.pt \
    --img-path ../data/sample_images/Infectious_keratitis.jpg \
    --output-dir outputs \
    --layer-name model_23_cv3_conv \
    --device cpu
```

This will generate a heatmap visualization and save it in the `outputs` directory.