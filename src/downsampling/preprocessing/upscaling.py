import numpy as np
from nanopyx.core.transform._le_interpolation_nearest_neighbor import ShiftAndMagnify as interp_nn
from nanopyx.core.transform._le_interpolation_catmull_rom import ShiftAndMagnify as interp_cr

def upscale(img, labels, magnification):
    nn_upscale = interp_nn()
    cr_upscale = interp_cr()

    img_out = cr_upscale.run(img, 0, 0, magnification, magnification)
    labels_out = nn_upscale.run(labels, 0, 0, magnification, magnification)

    # labels_out = np.zeros((labels.shape[0]*magnification, labels.shape[1]*magnification))

    # for i in range(1, np.max(labels)):
    #     tmp = labels == i
    #     filtered_img = tmp * img
    #     labels_out += (nn_upscale.run(filtered_img, 0, 0, magnification, magnification) > 0)[0] * i

    return img_out, labels_out
