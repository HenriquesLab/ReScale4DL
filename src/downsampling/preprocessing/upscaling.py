import numpy as np
from nanopyx.core.transform._le_interpolation_nearest_neighbor import ShiftAndMagnify as interp_nn
from nanopyx.core.transform._le_interpolation_catmull_rom import ShiftAndMagnify as interp_cr


def upscale_img(img, magnification):
    """Upscale an image by the magnification param using Catmull-Rom interpolation.
    :param img: 3D image array
    :param magnification: upscaling factor
    :return: upscaled array
    """
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


def upscale_labels(labels, magnification):
    """Upscale a labels image by the magnification param using Nearest-neighbor interpolation.
    :param img: 3D image array
    :param magnification: upscaling factor
    :return: upscaled array
    """
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
