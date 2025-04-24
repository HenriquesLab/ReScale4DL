# ReScale4DL: Balancing Pixel and Contextual Information for Enhanced Bioimage Segmentation

<img src="https://raw.githubusercontent.com/HenriquesLab/ReScale4DL/refs/heads/main/.github/logo.png" align="right" width="200"/>

[![Python 3.8+](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A systematic approach for determining optimal image resolution in deep learning-based microscopy segmentation, balancing accuracy with acquisition/storage costs. Following this approach, researchers can improve the sustainability and cost-effectiveness of bioimaging studies by reducing data and computing
needs while optimising microscopy techniques.

## Key Features
- **Resolution simulation**: Rescale images and their respective annotations (upsample and downsample)
- **Segmentation evaluation**: Compare performance across resolutions using:
  - Mean Intersection-over-Union (IoU)
  - Morphological features
  - Potential throughput
  - Personalised metrics
- **Visualization tools**: Generate comparative plots and sample outputs

## Installation

ReScale4DL is available as a Python package through pip. 
Activate your conda environment or create one:
```terminal
conda create -n rescale4dl "python<=3.12"
conda activate rescale4dl
```

Install ReScale4DL with `pip`:

```terminal
pip install rescale4dl
```

### Manual installation
Manual installation using the GitHub repository
```terminal
git clone https://github.com/HenriquesLab/ReScale4DL.git
cd rescale4dl
conda create -n rescale4dl "python<=3.12"
conda activate rescale4dl
python -m pip install .
```


## Usage

### 1. Image Rescaling
Notebook: [`Rescale_Images.ipynb`](https://github.com/HenriquesLab/ReScale4DL/blob/main/Notebooks/Rescale_Images.ipynb)

### 2. Segmentation Analysis 
Notebook: [`Evaluate_Segmentation.ipynb`](https://github.com/HenriquesLab/ReScale4DL/blob/main/Notebooks/Evaluate_Segmentation.ipynb)

### 3. Rescale and crop 
Notebook: [`Rescale_Foundation_Models.ipynb`](https://github.com/HenriquesLab/ReScale4DL/blob/main/Notebooks/Rescale_Foundation_Models.ipynb)


## Contributing
We welcome contributions through:
- [Issue reporting](https://github.com/HenriquesLab/ReScale4D/issues)
- [Pull requests](https://github.com/HenriquesLab/ReScale4D/pulls)

## License
MIT License - See [LICENSE](LICENSE) for details

## How to cite this work

Ferreira, M.G., Saraiva, B.M., Brito, A.D., Pinho, M.G., Henriques, R. and Gómez-de-Mariscal, E., *ReScale4DL: Balancing Pixel and Contextual Information for Enhanced Bioimage Segmentation.* bioRxiv, pp.2025-04, (2025) [https://doi.org/10.1101/2025.04.09.647871](https://doi.org/10.1101/2025.04.09.647871)

[![ReScale4DL-preprint](https://raw.githubusercontent.com/HenriquesLab/ReScale4DL/refs/heads/main/.github/rescale4dl-biorxiv.png)](https://www.biorxiv.org/content/10.1101/2025.04.09.647871v1)

```
@article{ferreira2025rescale4dl,
  title={ReScale4DL: Balancing Pixel and Contextual Information for Enhanced Bioimage Segmentation},
  author={Ferreira, Mariana G and Saraiva, Bruno M and Brito, Ant{\'o}nio D and Pinho, Mariana G and Henriques, Ricardo and G{\'o}mez-de-Mariscal, Estibaliz},
  journal={bioRxiv},
  pages={2025--04},
  year={2025},
  publisher={Cold Spring Harbor Laboratory},
  URL = https://doi.org/10.1101/2025.04.09.647871
}
```




