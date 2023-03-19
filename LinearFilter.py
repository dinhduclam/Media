import BasicProcessor
import numpy as np

def sobel_edge_detect(img):
    maskX = np.array(([-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]), dtype="float")
    maskY = np.array(([-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]), dtype="float")

    sobelX = BasicProcessor.convolve(img, maskX)
    sobelY = BasicProcessor.convolve(img, maskY)
    sobelXY = sobelX + sobelY

    return sobelXY

def laplace_edge_detect(img):
    mask = np.array(([0, 1, 0],
                   [1, -4, 1],
                   [0, 1, 0]), dtype="float")

    laplace_img = BasicProcessor.convolve(img, mask)
    return laplace_img

def robert_cross_edge_detect(img):
    mask_Robert_Cross1 = np.array(([0, 0, 0],
                                   [0, -1, 0],
                                   [0, 0, 1]), dtype="float")

    mask_Robert_Cross2 = np.array(([0, 0, 0],
                                   [0, 0, -1],
                                   [0, 1, 0]), dtype="float")

    robert_Cross1 = BasicProcessor.convolve(img, mask_Robert_Cross1)
    robert_Cross2 = BasicProcessor.convolve(img, mask_Robert_Cross2)
    robert_Cross = robert_Cross1 + robert_Cross2

    return robert_Cross

def prewitt_edge_detect(img):
    maskX = np.array(([-1, -1, -1],
                      [0, 0, 0],
                      [1, 1, 1]), dtype="float")
    maskY = np.array(([-1, 0, 1],
                      [-1, 0, 1],
                      [-1, 0, 1]), dtype="float")

    prewittX = BasicProcessor.convolve(img, maskX)
    prewittY = BasicProcessor.convolve(img, maskY)
    prewittXY = prewittX + prewittY

    return prewittXY

def gaussian_blur(img):
    mask = np.array([
        [1/16, 1/8, 1/16],
        [1/8, 1/4, 1/8],
        [1/16, 1/8, 1/16]
    ], dtype="float")

    gaussian_blur_img = BasicProcessor.convolve(img, mask)
    return gaussian_blur_img

def mean_filter(img):
    mask = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]])
    mask = mask / 9

    new_img = BasicProcessor.convolve(img, mask)
    return new_img