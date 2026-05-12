from __future__ import annotations

import matplotlib
import numpy as np
import pytest
from tifffile import imwrite


matplotlib.use("Agg")


@pytest.fixture
def label_image_2d() -> np.ndarray:
    image = np.zeros((6, 6), dtype=np.uint16)
    image[1:3, 1:3] = 1
    image[3:5, 3:5] = 2
    return image


@pytest.fixture
def prediction_image_2d() -> np.ndarray:
    image = np.zeros((6, 6), dtype=np.uint16)
    image[1:3, 1:3] = 10
    image[3:5, 4:6] = 20
    return image


@pytest.fixture
def volume_3d() -> np.ndarray:
    volume = np.zeros((3, 5, 5), dtype=np.uint16)
    volume[0:2, 1:3, 1:3] = 1
    volume[1:3, 3:5, 3:5] = 2
    return volume


@pytest.fixture
def segmentation_dataset(tmp_path, label_image_2d, prediction_image_2d):
    dataset = tmp_path / "Dataset"
    for sampling in ("OG", "downsampling_2"):
        gt_dir = dataset / sampling / "GT"
        pred_dir = dataset / sampling / "Prediction"
        gt_dir.mkdir(parents=True)
        pred_dir.mkdir(parents=True)
        imwrite(gt_dir / "sample.tif", label_image_2d)
        imwrite(pred_dir / "sample.tif", prediction_image_2d)
    return dataset


@pytest.fixture
def segmentation_volume_dataset(tmp_path, volume_3d):
    dataset = tmp_path / "Dataset3D"
    pred = volume_3d.copy()
    pred[:, 3:5, 3:5] = 3
    for sampling in ("OG", "downsampling_z_2", "upsampling_xyz_2"):
        gt_dir = dataset / sampling / "GT"
        pred_dir = dataset / sampling / "Prediction"
        gt_dir.mkdir(parents=True)
        pred_dir.mkdir(parents=True)
        imwrite(gt_dir / "volume.tif", volume_3d)
        imwrite(pred_dir / "volume.tif", pred)
    return dataset
