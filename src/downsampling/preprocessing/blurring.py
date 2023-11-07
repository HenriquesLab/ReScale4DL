import numpy as np
from skimage.filters import gaussian


def gaussian_blur(img: np.array, sigma: float):
    """
    Apply Gaussian blur to an image.
    Parameters:
        img (ndarray): The input image.
        sigma (float): The standard deviation for Gaussian kernel.
    Returns:
        ndarray: The blurred image.
    """
    return gaussian(img, sigma=sigma)
