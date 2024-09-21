import cv2
import numpy as np


def norm_unsharp_mask(img: np.ndarray):
    """
    Normalized unsharp mask\n
    S = (I + N {I − G ∗ I }) / 2\n
    where\n
    - G ∗ I is the Gaussian filtered image and\n
    - N is the linear normalization (histogram stretching) operator.

    :param img: image to process
    :return: S
    """
    gaussian_filtered = cv2.GaussianBlur(img, (5, 5), 0)

    # lin_normalized = cv2.equalizeHist((255 * img).astype(np.uint8) - (255 * gaussian_filtered).astype(np.uint8))
    # lin_normalized = lin_normalized.astype(np.float32) / 255

    return gaussian_filtered # (img + lin_normalized) / 2
