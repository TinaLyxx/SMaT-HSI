# SMaT-HSI: Spatial-Spectral Mamba Transformer for Hyperspectral Image Classification

## Overview

SMaT-HSI is a Mamba architecture-based hyperspectral image classification framework specifically designed for processing hyperspectral remote sensing image classification tasks. This project implements various Mamba variant models that can effectively extract spatial-spectral features from hyperspectral images.

This project implements a Spatial-Spectral Mamba Transformer (SMaT) for hyperspectral image (HSI) classification. It leverages the Mamba architecture to effectively model both spatial and spectral features in hyperspectral remote sensing data.

## Key Features

- **Multiple Mamba Model Architectures**: Implements 1D, 2D, and spatial-spectral fusion Mamba models
- **Efficient State Space Modeling**: Utilizes Mamba's selective state space mechanism for sequence data processing
- **Structure-Aware Processing**: Includes minimum spanning tree-based sequence generation and deep fusion mechanisms
- **Multi-Dataset Support**: Supports multiple standard hyperspectral datasets
- **Flexible Configuration**: Easy adjustment of model and training configurations through command-line parameters

## Supported Datasets

- **IP**: Indian Pines (16 classes)
- **PU**: Pavia University (9 classes)  
- **HU2013**: Houston 2013 (15 classes)
- **WHU-HC**: WHU-HanChuan (16 classes)
- **WHU-HH**: WHU-HongHu (22 classes)

## Requirements

### System Requirements
- Python 3.9+
- CUDA 11.7+ (GPU recommended)
- Linux/Unix system (recommended)



## Installation

1. **Clone the Repository**
```bash
git clone <repository-url>
cd SMaT-HSI
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. **Install Dependencies**
```bash
# Install basic dependencies
pip install -r requirements.txt

# Install custom CUDA kernels
cd kernels/selective_scan && pip install .
cd ../dwconv2d && python setup.py install
cd ../..
```

4. **Data Preparation**
Place dataset files in the `datasets/` directory following this structure:
```
datasets/
├── IP/
│   ├── Indian_pines_corrected.mat
│   └── Indian_pines_gt.mat
├── PU/
│   ├── PaviaU.mat
│   └── PaviaU_gt.mat
└── ...
```

## Usage

### Basic Run
```bash
python main.py --dataset IP --model_id 4 --train_num 20 --epoch 200
```


### Example Commands

```bash
# Train on Pavia University dataset
python main.py \
    --dataset PU \
    --train_num 50 \
    --batch_size 64 \
    --epoch 200 \
    --windowsize 27 \
    --model_id 4

```

## Model Architecture

### Mamba Spatial Model (model_id=4)

This model is the core of the project, implementing joint spatial-spectral processing:

1. **Dimension Reduction**: Uses convolutional layers to reduce high-dimensional spectral data
2. **Spatial Branch**: Processes spatial neighborhood information
3. **Spectral Branch**: Processes spectral features of center pixels
4. **Fusion Mechanism**: Fuses spatial and spectral features through gating mechanisms
5. **Classification Head**: Outputs final classification results

### Key Components

- **StructureAwareSSM**: Structure-aware state space model
- **StateFusion**: Multi-scale spatial feature fusion
- **PatchEmbed**: Image patch embedding
- **spectral_spatial_block**: Spatial-spectral fusion block

## Experimental Results

The model has been evaluated on multiple standard hyperspectral datasets, with output metrics including:
- Overall Accuracy (OA)
- Average Accuracy (AA)
- Kappa coefficient
- Per-class accuracy
- Training and testing time

Result files are automatically saved as PNG format classification maps.



## Contributing

We welcome Issues and Pull Requests to improve the project.

## License

This project is released under an open source license. Please see the LICENSE file for specific license information.