def check_crop_img(arr, bin_factor):
    """Crops the image if any of the dims is not divisible by bin factor)"""
    r, c = arr.shape

    if c % bin_factor != 0:
        c = int(c / bin_factor) * bin_factor
    if r % bin_factor != 0:
        r = int(r / bin_factor) * bin_factor

    return arr[:r, :c]