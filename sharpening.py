import cv2
import numpy as np


def norm_unsharp_mask(img: np.ndarray):
    """
    Normalized unsharp mask\n
    S = (I + N {I − G ∗ I}) / 2\n
    where\n
    - G ∗ I is the Gaussian-filtered image and\n
    - N is the linear normalization (histogram stretching) operator.

    :param img: image to process
    :return: S
    """

    gaussian_filtered = cv2.GaussianBlur(img, (5, 5), 0)
    difference = img - gaussian_filtered

    difference_yuv = cv2.cvtColor((255 * difference).astype(np.uint8), cv2.COLOR_BGR2YUV)
    difference_yuv[:, :, 0] = cv2.equalizeHist(difference_yuv[:, :, 0])

    lin_normalized = cv2.cvtColor(difference_yuv, cv2.COLOR_YUV2BGR).astype(np.float32) / 255

    return (img + lin_normalized) / 2
