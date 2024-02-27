import numpy as np

from nanopyx.core.transform._le_interpolation_catmull_rom import ShiftAndMagnify as interp_cr
from nanopyx.core.transform._le_interpolation_catmull_rom import ShiftScaleRotate as magnify_cr
from nanopyx.core.transform._le_interpolation_nearest_neighbor import ShiftAndMagnify as interp_nn
from nanopyx.core.transform._le_interpolation_nearest_neighbor import ShiftScaleRotate as magnify_nn


def upsample_img(img: np.array, magnification: int, keep_dims: bool = False):
    """Upscale an image by the magnification param using Catmull-Rom interpolation.
    :param img: 3D image array
    :param magnification: upscaling factor
    :param keep_dims: whether to keep the original dimensions or just upscale the image (defaults to False)
    :return: upscaled array
    """
    if keep_dims:
        cr_upscale = magnify_cr()
        return np.asarray(
            cr_upscale.run(
                img.astype(np.float32),
                0,
                0,
                magnification,
                magnification,
                0
                ),
            dtype=np.float32
            )
    else:
        cr_upscale = interp_cr()
        return np.asarray(
            cr_upscale.run(
                img.astype(np.float32),
                0,
                0,
                magnification,
                magnification
                ),
            dtype=np.float32
            )


def upsample_labels(labels: np.array, magnification: int, keep_dims: bool = False):
    """Upscale a labels image by the magnification param using Nearest-neighbor interpolation.
    :param img: 3D image array
    :param magnification: upscaling factor
    :param keep_dims: whether to keep the original dimensions or just upscale the image (defaults to False)
    :return: upscaled array
    """
    if keep_dims:
        nn_upscale = magnify_nn()
        return np.asarray(
            nn_upscale.run(
                labels.astype(np.float32),
                0,
                0,
                magnification,
                magnification,
                0
                ),
            dtype=np.float32
            )
    else:
        nn_upscale = interp_nn()
        return np.asarray(
            nn_upscale.run(
                labels.astype(np.float32),
                0,
                0,
                magnification,
                magnification
                ),
            dtype=np.float32
            )
