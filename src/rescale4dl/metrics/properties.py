# Imports
import ast
import numpy as np
import pandas as pd
from scipy import ndimage
import skimage as ski
from ..utils import get_csv_dict

from typing import List, Tuple, Dict, Optional

## Miscellaneous functions


def pixel_coverage_percent(img_array: np.ndarray) -> float:
    """
    Calculate the pixel coverage percentage of the input image array, how much of the whole object is covered by a single pixel.

    Args:
        img_array (np.ndarray): A numpy array image array with a single object label.

    Returns:
        pixel_coverage_percent (float): The percentage of the object that each pixel covers, as a float.

    """

    # Calculate the percentage of the object that each pixel covers
    pixel_coverage = (1 / np.count_nonzero(img_array)) * 100

    return pixel_coverage


def object_diameter(image_array: np.array) -> Tuple[float, float, float, float]:
    """
    Calculate the diameter of the object in the image array.

    Args:
        image_array: A numpy/dataframe image array with a single object

    Returns:
        min_diameter: The minimum diameter of the object in the image array.
        max_diameter: The maximum diameter of the object in the image array.
        mean_diameter: The mean diameter of the object in the image array.
        median_diameter: The median diameter of the object in the image array.
    """
    # Calculate the object skeleton and Euclidean distance transform
    obj_skeleton = ski.morphology.skeletonize(image_array)
    obj_edt = ndimage.distance_transform_edt(image_array)

    # Get the EDT values for the object skeleton
    obj_skeleton_edt = obj_skeleton * obj_edt

    # Calculate the min, max, mean, and median radius excluding the zero values of the background, multiply by 2 for diameter
    min_diameter = np.min(obj_skeleton_edt[np.nonzero(obj_skeleton_edt)]) * 2
    max_diameter = np.max(obj_skeleton_edt[np.nonzero(obj_skeleton_edt)]) * 2
    mean_diameter = np.mean(obj_skeleton_edt[np.nonzero(obj_skeleton_edt)]) * 2
    median_diameter = np.median(obj_skeleton_edt[np.nonzero(obj_skeleton_edt)]) * 2

    return min_diameter, max_diameter, mean_diameter, median_diameter


