# ReScale4DL: Balancing Pixel and Contextual Information for Enhanced Bioimage Segmentation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<img src="https://raw.githubusercontent.com/HenriquesLab/ReScale4DL/refs/heads/main/.github/logo.png?token=GHSAT0AAAAAAC5PYSXBD65GVB3HMVLFOUDGZ7S7V3A" align="right" width="200"/>

A systematic approach for determining optimal image resolution in deep learning-based microscopy segmentation, balancing accuracy with acquisition/storage costs.

## Key Features
- **Resolution simulation**: Rescale images and their respective annotations (upsample and downsample)
- **Segmentation evaluation**: Compare performance across resolutions using:
  - Mean Intersection-over-Union (IoU)
  - Morphological features
  - Potential throughput
  - Personalised metrics
- **Visualization tools**: Generate comparative plots and sample outputs

## Installation
Manual installation of this repository (until its release)
```terminal
git clone https://github.com/HenriquesLab/ReScale4DL.git
cd rescale4dl
conda create -n rescale4dl "python=>3.10"
conda activate rescale4dl
pip install -r requirements.txt`
```


```terminal
pip install rescale4dl
```


## Usage

### 1. Image Rescaling
Notebook: `rescale_images.ipynb`

### 2. Segmentation Analysis 
Notebook: `evaluate_segmentation.ipynb`


## Key Parameters


## Contributing
We welcome contributions through:
- [Issue reporting](https://github.com/HenriquesLab/ReScale4D/issues)
- [Pull requests](https://github.com/HenriquesLab/ReScale4D/pulls)

## License
MIT License - See [LICENSE](LICENSE) for details

## Citation
If using this work in research, please cite:






