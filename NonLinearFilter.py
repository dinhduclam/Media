import numpy as np

def median_filter(img):
    m, n = img.shape
    img_new = np.zeros([m, n])
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            temp = [img[i - 1, j - 1],
                    img[i - 1, j],
                    img[i - 1, j + 1],
                    img[i, j - 1],
                    img[i, j],
                    img[i, j + 1],
                    img[i + 1, j - 1],
                    img[i + 1, j],
                    img[i + 1, j + 1]]

            temp.sort()
            img_new[i, j] = temp[4]

    return img_new

def max_filter(img):
    m, n = img.shape
    img_new = np.zeros([m, n])
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            max_value = max(img[i - 1, j - 1],
                    img[i - 1, j],
                    img[i - 1, j + 1],
                    img[i, j - 1],
                    img[i, j],
                    img[i, j + 1],
                    img[i + 1, j - 1],
                    img[i + 1, j],
                    img[i + 1, j + 1])

            img_new[i, j] = max_value

    return img_new

def min_filter(img):
    m, n = img.shape
    img_new = np.zeros([m, n])
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            min_value = min(img[i - 1, j - 1],
                            img[i - 1, j],
                            img[i - 1, j + 1],
                            img[i, j - 1],
                            img[i, j],
                            img[i, j + 1],
                            img[i + 1, j - 1],
                            img[i + 1, j],
                            img[i + 1, j + 1])

            img_new[i, j] = min_value

    return img_new