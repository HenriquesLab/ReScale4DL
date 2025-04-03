from rescale4dl.blurring import gaussian_blur
import numpy as np


def test_gaussian_blur():
    # Test with a simple 2D array
    img = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32
    )
    sigma = 1.0
    blurred_img = gaussian_blur(img, sigma)

    # Check the shape of the output
    assert (
        blurred_img.shape == img.shape
    ), "Output shape does not match input shape"
