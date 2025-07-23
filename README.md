# YOLOv5 Attention Analysis for Anterior Eye Disease Classification

This repository contains the implementation code for the paper "YOLOv5 Attention Analysis for Anterior Eye Disease Classification: Grad-CAM++ Feature Importance and Cut-and-Paste Validation".

## Overview

This codebase provides tools for analyzing the explainability of YOLOv5 models in anterior segment disease classification through:

1. **Grad-CAM++ Visualization**: Visualize which regions the model focuses on across different network layers
2. **Cut-and-Paste Validation**: Assess the model's reliance on contextual information vs. primary pathological features
3. **Expert Annotation Comparison**: Validate model attention against expert-annotated lesion areas

## Features

- YOLOv5 model wrapper with intermediate layer access
- Grad-CAM++ implementation optimized for YOLOv5 architecture
- Cut-and-paste validation framework for context dependency analysis
- Statistical analysis tools (Kruskal-Wallis, Dunn's test, Mann-Whitney U)
- Comprehensive visualization suite
- Metrics calculation (IoU, Pointing Game Accuracy, AOI_50)

## Installation

### Requirements

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/kitaguchi77/yolov5-gradcam-corneai.git
cd yolov5-gradcam-corneai
```

2. Clone the YOLOv5 repository (forked from CorneAI):
```bash
git clone https://github.com/modafone/corneaai.git yolov5
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

### 1. Image Data

Prepare your test images in a directory structure:
```
data/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── annotations/
    ├── image_list.csv
    ├── cornea_annotations.json
    └── expert_annotations.json
```

### 2. Image List Format

Create a CSV file (`image_list.csv`) with the following columns:
```csv
image_path,label,class_name
images/image1.jpg,0,Normal
images/image2.jpg,1,Infectious keratitis
...
```

### 3. Cornea Annotations

For cut-and-paste validation, provide cornea region annotations in JSON format:
```json
[
  {
    "image_path": "images/image1.jpg",
    "center_x": 320,
    "center_y": 240,
    "major_axis": 200,
    "minor_axis": 180,
    "angle": 0
  },
  ...
]
```

### 4. Expert Annotations (Optional)

For expert comparison, provide lesion annotations:
```json
[
  {
    "image_path": "images/image1.jpg",
    "lesion_polygon": [[x1,y1], [x2,y2], ...],
    "diagnosis": "Infectious keratitis",
    "annotator": "Expert1"
  },
  ...
]
```

## Usage

### Basic Usage

Run the complete analysis pipeline:

```bash
python main.py \
  --config configs/config.yaml \
  --weights path/to/yolov5_weights.pt \
  --test-data data/image_list.csv \
  --output-dir results \
  --all
```

### Individual Analyses

#### Grad-CAM++ Analysis Only
```bash
python main.py \
  --config configs/config.yaml \
  --weights path/to/yolov5_weights.pt \
  --test-data data/image_list.csv \
  --gradcam \
  --save-visualizations
```

#### Cut-and-Paste Validation Only
```bash
python main.py \
  --config configs/config.yaml \
  --weights path/to/yolov5_weights.pt \
  --test-data data/image_list.csv \
  --cut-paste
```

#### Expert Comparison
```bash
python main.py \
  --config configs/config.yaml \
  --weights path/to/yolov5_weights.pt \
  --test-data data/image_list.csv \
  --gradcam \
  --expert-comparison
```

### Advanced Options

- `--device`: Choose device ('cpu' or 'cuda')
- `--target-layers`: Specify which layers to analyze (default: ['17', '20', '23', '24_m_0', '24_m_1', '24_m_2'])
- `--batch-size`: Set batch size for processing
- `--save-visualizations`: Save all generated visualizations

## Configuration

Edit `configs/config.yaml` to customize:

```yaml
model:
  img_size: 640
  conf_threshold: 0.25
  device: cpu  # or cuda

gradcam:
  target_layers: [17, 20, 23, "24_m_0", "24_m_1", "24_m_2"]
  normalize_cam: true

cut_paste:
  num_backgrounds: 341
  min_confidence: 0.9

analysis:
  aoi_threshold: 0.5
  significance_level: 0.05

paths:
  cornea_annotations: "data/cornea_annotations.json"
  expert_annotations: "data/expert_annotations.json"
```

## Output Structure

```
results/
├── gradcam/
│   ├── image1_layers.png
│   └── ...
├── visualizations/
│   ├── aoi_boxplot.png
│   ├── cutpaste_matrix.png
│   └── statistics.png
├── gradcam_aoi_results.csv
├── cut_paste_results.csv
├── cut_paste_accuracy_matrix.npy
├── context_dependency_analysis.json
├── expert_comparison_results.csv
├── clinical_relevance_analysis.json
├── statistical_analysis_results.json
└── analysis_summary.json
```

## Key Metrics

### Area of Interest (AOI_50)
Proportion of pixels in the attention map with activation values exceeding 50% of the maximum.

### Intersection over Union (IoU)
Overlap between model attention and expert-annotated lesion areas.

### Pointing Game Accuracy
Whether the point of maximum activation falls within the expert-annotated region.

### Context Dependency Score
Difference in accuracy between same-background and different-background conditions in cut-and-paste validation.

## Disease Categories

The model classifies anterior segment images into 9 categories:

1. Normal
2. Infectious keratitis
3. Non-infectious keratitis
4. Scar
5. Tumor
6. Deposit
7. Acute primary angle closure (APAC)
8. Lens opacity
9. Bullous keratopathy

## Citation

If you use this code in your research, please cite:

```bibtex
@article{kitaguchi2024yolov5,
  title={YOLOv5 Attention Analysis for Anterior Eye Disease Classification: Grad-CAM++ Feature Importance and Cut-and-Paste Validation},
  author={Kitaguchi, Yoshiyuki and Ueno, Yuta and Yamaguchi, Takefumi and others},
  journal={Journal Name},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv5 implementation: https://github.com/ultralytics/yolov5
- CorneAI fork: https://github.com/modafone/corneaai
- Japan Anterior Segment Artificial Intelligence Research Group
- Japan Ocular Imaging Registry

## Contact

For questions or issues, please contact:
- Yoshiyuki Kitaguchi: kitaguchi@ophthal.med.osaka-u.ac.jp

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU mode
2. **Import errors**: Ensure YOLOv5 is cloned in the correct directory
3. **Missing annotations**: Check file paths in config.yaml

### Debug Mode

For detailed logging, set the environment variable:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python -u main.py --config configs/config.yaml ...
```