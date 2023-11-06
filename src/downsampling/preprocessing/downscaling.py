import numpy as np

from skimage.transform import rescale
from nanopyx.core.transform.binning import rebin_2d
from nanopyx.core.transform._le_convolution import Convolution



def binning_downsize(img: np.array, downsampling_factor: int, mode: str = "sum"):
    """Bins a 2D array by a given factor. The last two dimensions of the array are binned.
    :param arr: numpy array with any shape as long as last two dimensions are y, x (example: time, channel, z, y, x)
    :param bin_factor: factor used to bin dimensions
    :param mode: can be either sum or mean, defaults to sum if not specified or not valid mode
    :return: binned array
    """
    return rebin_2d(img, downsampling_factor, mode=mode)


def binning_blur(img: np.array, downsampling_factor: int, mode: str = "sum"):
    """Blurs a 2D array by a given factor using binning. The last two dimensions of the array are binned.
    :param arr: numpy array with any shape as long as last two dimensions are y, x (example: time, channel, z, y, x)
    :param bin_factor: factor used to bin dimensions
    :param mode: can be either sum or mean, defaults to sum if not specified or not valid mode
    :return: binned array
    """
    conv = Convolution()

    if mode not in ["sum", "mean"]:
        mode = "sum"

    if mode == "sum":
        kernel = np.ones((downsampling_factor, downsampling_factor), dtype=np.float32)
        return np.asarray(
            conv.run(img.astype(np.float32), kernel),
            dtype=np.float32
            )

    elif mode == "mean":
        kernel = np.ones((
                downsampling_factor, downsampling_factor),
                dtype=np.float32) / (
                downsampling_factor**2
                )
        return np.asarray(
            conv.run(img.astype(np.float32), kernel),
            dtype=np.float32
            )


def binning_img(img: np.array, downsampling_factor: int, keep_dims: bool = False, mode: str = "sum"):
    """Bins a 2D array by a given factor. The last two dimensions of the array are binned.
    :param arr: numpy array with any shape as long as last two dimensions are y, x (example: time, channel, z, y, x)
    :param bin_factor: factor used to bin dimensions
    :param keep_dims: whether to keep the original dimensions or just blur the image (defaults to False)
    :param mode: can be either sum or mean, defaults to sum if not specified or not valid mode
    :return: binned array
    """
    if keep_dims:
        return binning_blur(img, downsampling_factor, mode=mode)
    else:
        return binning_downsize(img, downsampling_factor, mode=mode)

def binning_label(img: np.array, downsampling_factor: int):
    """Bins a 2D array by a given factor using Nearest-neighbor.
    :params img: Input image, should be a 2-d np array.
    :params downsampling_factor: factor used to bin dimensions"""
    return rescale(img, downsampling_factor, anti_aliasing=False, order=0)
