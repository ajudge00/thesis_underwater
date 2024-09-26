import cv2
from wb_comps import comp_for_channel
from gamma_comps import gamma_correction
from sharpening import norm_unsharp_mask
from weights import laplacian_contrast_weight, saliency_weight, saturation_weight

FILENAME = 'harang'

if __name__ == '__main__':
    original = cv2.imread(f'imgs_from_pdf/{FILENAME}.jpg', cv2.IMREAD_COLOR)
    original = cv2.normalize(original, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # WHITE BALANCE
    wb_comp_red = comp_for_channel(channel='red', img=original)
    wb_comp_blue = comp_for_channel(channel='blue', img=original)
    wb_comp_red_blue = comp_for_channel(channel='blue', img=wb_comp_red)

    # GAMMA CORRECTION and EDGE SHARPENING
    gamma_corr = gamma_correction(img=wb_comp_red_blue, gamma=2.0)
    sharpened = norm_unsharp_mask(img=wb_comp_red_blue)

    # WEIGHTS
    gamma_lap = laplacian_contrast_weight(img=gamma_corr)
    gamma_sali = saliency_weight(img=gamma_corr)
    gamma_satur = saturation_weight(img=gamma_corr)

    sharpened_lap = laplacian_contrast_weight(img=sharpened)
    sharpened_sali = saliency_weight(img=sharpened)
    sharpened_satur = saturation_weight(img=sharpened)

    # MULTI-SCALE FUSION
    # TO-DO

    outputs = {
        "original": original,
        "wb_comp_red": wb_comp_red,
        "wb_comp_blue": wb_comp_blue,
        "wb_comp_red_blue": wb_comp_red_blue,
        "gamma_corr": gamma_corr,
        "sharpened": sharpened,
        # "gamma_lap": gamma_lap,
        # "gamma_sali": gamma_sali,
        # "gamma_satur": gamma_satur,
        # "sharpened_lap": sharpened_lap,
        # "sharpened_sali": sharpened_sali,
        # "sharpened_satur": sharpened_satur
    }

    z = '0'
    i = 1
    for key in outputs.keys():
        cv2.imwrite(f'exports/{z}{i}_{FILENAME}_{key}.jpg', (outputs[key] * 255).astype('uint8'))
        i += 1
        z = '' if i > 9 else '0'
