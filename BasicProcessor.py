import numpy as np

def convolve(img, mask):
    m, n = img.shape
    img_new = np.zeros([m, n])

    for i in range(1, m-1):
        for j in range(1, n-1):
            temp   =  img[i-1, j-1]    * mask[0, 0]\
                   +  img[i, j-1]      * mask[0, 1]\
                   +  img[i+1, j - 1]  * mask[0, 2]\
                   +  img[i-1, j]      * mask[1, 0]\
                   +  img[i, j]        * mask[1, 1]\
                   +  img[i+1, j]      * mask[1, 2]\
                   +  img[i - 1, j+1]  * mask[2, 0]\
                   +  img[i, j + 1]    * mask[2, 1]\
                   +  img[i + 1, j + 1]* mask[2, 2]
            img_new[i, j] = temp
    img_new = img_new.astype(np.uint8)
    return img_new

def equalize_histogram(img):
    hist = calculate_histogram(img)

    cumsum = np.zeros(256)
    cumsum[0] = hist[0]

    for i in range(1, 256):
        cumsum[i] = cumsum[i-1] + hist[i]

    new_img = np.zeros(img.shape, dtype="int")
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img[i][j] = round(255 * cumsum[img[i][j]] / cumsum[255])

    return new_img

def calculate_histogram(img):
    histogram = np.zeros(256)
    for row in img:
        for pixel in row:
            histogram[pixel] += 1

    return histogram

def gamma(img, gamma, c):
    return float(c) * pow(img, float(gamma))

def logarit(img, c):
    return float(c) * np.log(1.0 + img)

def binary(img, th=128):
    return img > th

def dao_anh(img):
    return 255-img