import os
import numpy as np
from typing import List
from tkinter import filedialog as fd
from tifffile import imread, imwrite


def downsample_batch(input_folder_path: str, downsampling_factor: int, keep_dims: bool = False, mode: str = "sum"):
    """Downsamples a batch of images by a given factor. The last two dimensions of the array are binned.
    Creates new folders outside input_folder to store the results.
    :param input_folder_path: path to folder containing an "images" folder and a "labels" folder. Images inside both folders should have the same name.
    :param downsampling_factor: factor used to bin dimensions
    :para keep_dims: whether to keep the original dimensions or just blur the image (defaults to False)
    :param mode: can be either sum, max or mean, defaults to sum if not specified or not valid mode
    """

    if keep_dims:
        new_dataset_path = os.path.join(input_folder_path, f"downsampled_{downsampling_factor}_mode_{mode}_same_dims")
    else:
        new_dataset_path = os.path.join(input_folder_path, f"downsampled_{downsampling_factor}_mode_{mode}_diff_dims")

    new_images_path = os.path.join(new_dataset_path, "Images")
    new_labels_path = os.path.join(new_dataset_path, "Labels")

    if not os.path.exists(new_dataset_path):
        os.mkdir(new_dataset_path)
        os.mkdir(new_images_path)
        os.mkdir(new_labels_path)

    for img_name in os.path.join(input_folder_path, "Images"):
        img = imread(os.path.join(input_folder_path, "Images", img_name))
        lbl = imread(os.path.join(input_folder_path, "Labels", img_name))
        imwrite(os.path.join(new_images_path, img_name), binning_img(img, downsampling_factor, keep_dims=keep_dims, mode=mode))
        imwrite(os.path.join(new_labels_path, img_name), binning_lbl(lbl, downsampling_factor, keep_dims=keep_dims, mode=mode))


def upsample_batch(input_folder_path: str, magnification: int, keep_dims: bool = False):
    """Upsamples a batch of images by the magnification param using Catmull-rom interpolation and labels using Nearest-neighbor.
    Creates new folders outside input_folder to store the results.
    :param input_folder_path: path to folder containing an "images" folder and a "labels" folder. Images inside both folders should have the same name.
    :param magnification: upscaling factor
    :para keep_dims: whether to keep the original dimensions or just blur the image (defaults to False)
    """

    if keep_dims:
        new_dataset_path = os.path.join(input_folder_path, f"upsampled_{magnification}_same_dims")
    else:
        new_dataset_path = os.path.join(input_folder_path, f"upsampled_{magnification}_diff_dims")

    new_images_path = os.path.join(new_dataset_path, "Images")
    new_labels_path = os.path.join(new_dataset_path, "Labels")

    if not os.path.exists(new_dataset_path):
        os.mkdir(new_dataset_path)
        os.mkdir(new_images_path)
        os.mkdir(new_labels_path)

    for img_name in os.path.join(input_folder_path, "Images"):
        img = imread(os.path.join(input_folder_path, "Images", img_name))
        lbl = imread(os.path.join(input_folder_path, "Labels", img_name))
        imwrite(os.path.join(new_images_path, img_name), upsample_img(imread(img), magnification, keep_dims=keep_dims))
        imwrite(os.path.join(new_labels_path, img_name), upsample_labels(imread(lbl), magnification, keep_dims=keep_dims))


def process_batch(input_folder_path: str, magnifications: List[int], downsampling_factors: List[int], modes: List[str] = ["sum", "mean"]):
    """Performs all downstream preprocessing on a single dataset"""

    for mag in magnifications:
        upsample_batch(
            input_folder_path,
            mag
            )
        upsample_batch(
            input_folder_path,
            mag,
            keep_dims=True
            )

    for mode in modes:
        for dsf in downsampling_factors:
            downsample_batch(
                input_folder_path,
                dsf,
                keep_dims=True,
                mode=mode
                )
            downsample_batch(
                input_folder_path,
                dsf,
                keep_dims=False,
                mode=mode
                )


def process_all_datasets(datasets_path: str, downsampling_factor: List[int], magnification: List[int], modes: List[str] = ["sum", "mean"]):
    """Performs all downstream preprocessing on all datasets in a folder"""

    if datasets_path is None:
        datasets_path = fd.askdirectory()

    for fld in os.listdir(datasets_path):
        if os.path.isdir(os.path.join(datasets_path, fld)):
            process_batch(
                os.path.join(datasets_path, fld),
                downsampling_factor,
                magnification,
                modes
                )
