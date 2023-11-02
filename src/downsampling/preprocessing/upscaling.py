from nanopyx.core.transform._le_interpolation_nearest_neighbor import ShiftAndMagnify as interp_nn
from nanopyx.core.transform._le_interpolation_catmull_rom import ShiftAndMagnify as interp_cr


def upscale_img(img, magnification):
    cr_upscale = interp_cr()
    return cr_upscale.run(img, 0, 0, magnification, magnification)


def upscale_labels(labels, magnification):
    nn_upscale = interp_nn()
    return nn_upscale.run(labels, 0, 0, magnification, magnification)
