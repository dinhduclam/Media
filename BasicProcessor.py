import numpy as np
import matplotlib.pyplot as plt

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

    new_img = np.zeros(img.shape, dtype="i4")
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img[i][j] = round(255 * cumsum[int(img[i][j])] / cumsum[255])

    return new_img

def calculate_histogram(img):
    histogram = np.zeros(256)
    for row in img:
        for pixel in row:
            histogram[int(pixel)] += 1

    return histogram

def gamma(img, gamma, c):
    return float(c) * pow(img, float(gamma))

def logarit(img, c):
    return float(c) * np.log(1.0 + img)

def binary(img, th=128):
    return img > th

def dao_anh(img):
    return 255-img

def cat_mp_bit(img):
    lst = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            lst.append(np.binary_repr(img[i][j], width=8))

    mp_bit_8 = []
    for i in lst:
        mp_bit_8.append(int(i[0]))

    mp_bit_7 = []
    for i in lst:
        mp_bit_7.append(int(i[1]))

    mp_bit_6 = []
    for i in lst:
        mp_bit_6.append(int(i[2]))

    mp_bit_5 = []
    for i in lst:
        mp_bit_5.append(int(i[3]))

    mp_bit_4 = []
    for i in lst:
        mp_bit_4.append(int(i[4]))

    mp_bit_3 = []
    for i in lst:
        mp_bit_3.append(int(i[5]))

    mp_bit_2 = []
    for i in lst:
        mp_bit_2.append(int(i[6]))

    mp_bit_1 = []
    for i in lst:
        mp_bit_1.append(int(i[7]))

    image_bit_8 = (np.array(mp_bit_8, dtype='uint8') * 128).reshape(img.shape[0], img.shape[1])
    image_bit_7 = (np.array(mp_bit_7, dtype='uint8') * 64).reshape(img.shape[0], img.shape[1])
    image_bit_6 = (np.array(mp_bit_6, dtype='uint8') * 32).reshape(img.shape[0], img.shape[1])
    image_bit_5 = (np.array(mp_bit_5, dtype='uint8') * 16).reshape(img.shape[0], img.shape[1])
    image_bit_4 = (np.array(mp_bit_4, dtype='uint8') * 8).reshape(img.shape[0], img.shape[1])
    image_bit_3 = (np.array(mp_bit_3, dtype='uint8') * 4).reshape(img.shape[0], img.shape[1])
    image_bit_2 = (np.array(mp_bit_2, dtype='uint8') * 2).reshape(img.shape[0], img.shape[1])
    image_bit_1 = (np.array(mp_bit_1, dtype='uint8') * 1).reshape(img.shape[0], img.shape[1])

    fig = plt.figure(figsize=(16, 9))
    (ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9) = fig.subplots(3, 3)

    ax1.imshow(img, cmap='gray')
    ax1.set_title("Ảnh gốc")

    ax2.imshow(image_bit_8, cmap='gray')
    ax2.set_title("Mặt phẳng bit 8")

    ax3.imshow(image_bit_7, cmap='gray')
    ax3.set_title("Mặt phẳng bit 7")

    ax4.imshow(image_bit_6, cmap='gray')
    ax4.set_title("Mặt phẳng bit 6")

    ax5.imshow(image_bit_5, cmap='gray')
    ax5.set_title("Mặt phẳng bit 5")

    ax6.imshow(image_bit_4, cmap='gray')
    ax6.set_title("Mặt phẳng bit 4")

    ax7.imshow(image_bit_3, cmap='gray')
    ax7.set_title("Mặt phẳng bit 3")

    ax8.imshow(image_bit_2, cmap='gray')
    ax8.set_title("Mặt phẳng bit 2")

    ax9.imshow(image_bit_1, cmap='gray')
    ax9.set_title("Mặt phẳng bit 1")
    plt.show()

    fig1 = plt.figure(figsize=(16, 9))
    (bx1, bx2), (bx3, bx4) = fig1.subplots(2, 2)

    bx1.imshow(img, cmap='gray')
    bx1.set_title("ảnh gốc")

    bx2.imshow(image_bit_8 + image_bit_7, cmap='gray')
    bx2.set_title("Mặt phẳng bit 8 và bit 7")

    bx3.imshow(image_bit_8 + image_bit_7 + image_bit_6, cmap='gray')
    bx3.set_title("Mặt phẳng bit 8, bit 7 và bit 6")

    bx4.imshow(image_bit_8 + image_bit_7 + image_bit_6 + image_bit_5, cmap='gray')
    bx4.set_title("Mặt phẳng bit 8, bit 7, bit 6 và bit 5")

    plt.show()