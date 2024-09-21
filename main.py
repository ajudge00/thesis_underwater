import cv2
from wb_comps import comp_for_channel
from gamma_comps import gamma_correction
from sharpening import norm_unsharp_mask


if __name__ == '__main__':
    img = cv2.imread('imgs_from_pdf/278.jpg')
    assert img is not None, "A kép megnyitása nem sikerült."

    img_norm = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # I. WHITE BALANCE
    img_comp_r = comp_for_channel(channel='red', img=img_norm)
    img_comp_rb = comp_for_channel(channel='blue', img=img_comp_r)

    cv2.imshow('Original', img_norm)
    cv2.imshow('RED comp.', img_comp_r)
    cv2.imshow('RED + BLUE comp.', img_comp_rb)

    # II.A. GAMMA CORRECTION
    img_gammad = gamma_correction(img_comp_rb, 2.2)
    cv2.imshow('Gamma corrected', img_gammad)

    # II.A. EDGE SHARPENING
    img_sharpened = norm_unsharp_mask(img_comp_rb)
    cv2.imshow('Sharpened', img_sharpened)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
