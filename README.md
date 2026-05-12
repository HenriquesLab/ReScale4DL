# ReScale4DL: Balancing Pixel and Contextual Information for Enhanced Bioimage Segmentation

<img src="https://raw.githubusercontent.com/HenriquesLab/ReScale4DL/refs/heads/main/.github/logo.png" align="right" width="200"/>

[![PyPI](https://img.shields.io/pypi/v/rescale4dl.svg)](https://pypi.org/project/rescale4dl/)
[![Python 3.10-3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/downloads/)
[![Oncall Tests](https://github.com/HenriquesLab/ReScale4DL/actions/workflows/oncall_test.yml/badge.svg)](https://github.com/HenriquesLab/ReScale4DL/actions/workflows/oncall_test.yml)
[![Nightly](https://github.com/HenriquesLab/ReScale4DL/actions/workflows/nightly_coverage.yml/badge.svg)](https://github.com/HenriquesLab/ReScale4DL/actions/workflows/nightly_coverage.yml)
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

#### Additional DL resources for microscopy:
The deep learning networks presented in the ReScale4DL paper were trained using the following platforms:

* **[ZeroCostDL4Mic](https://github.com/HenriquesLab/ZeroCostDL4Mic)**: A Google Colab-based no-cost toolbox to explore Deep Learning in Microscopy
* **[DL4MicEverywhere](https://github.com/HenriquesLab/DL4MicEverywhere)**: Docker-based implementation bringing the ZeroCostDL4Mic experience for local deployment

For detailed hyperparameter settings and training configurations, please refer to **Table 1** in our [bioRxiv preprint](https://doi.org/10.1101/2025.04.09.647871).

## Scripts
ReScale4DL provides a set of functionalities to quickly analyse your images and find an optimal pixel size: 
1. Rescaling the images in the path by donwsampling with a factor of 2 and 3, and by upsampling with a factor of 2:
```
rescale4dl.batch.process_all_datasets(“/path/data”, [2,3], [2], [1], modes=[“mean”])
```
2. Analyse the segmentation results for different scaling factors in 2D:
```
rescale4dl.analyse(“/path/data”) 
```
3. Analyse the segmentation results for different scaling factors in 3D:
```
rescale4dl.analyse(“/path/data”,
        is_3d=True,
        run_per_object_stats = False, # True for Instance Segmentation, False for Semantic or Binary Segmentation
        save_images = False, # True to save images of the segmentation examples and data distributions, False to skip saving images and saving some memory
        sampling_dir_list = None)
```

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



