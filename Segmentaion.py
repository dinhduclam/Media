import math

import numpy as np

def cat_nguong_toan_cuc(img):
    t = np.mean(img)
    g1 = []
    g2 = []
    m, n = img.shape

    while (True):
        for i in range(m):
            for j in range(n):
                if img[i, j] < t:
                    g1.append(img[i, j])
                else:
                    g2.append(img[i, j])

        mg1 = np.mean(g1)
        mg2 = np.mean(g2)

        t = (mg1 + mg2)/2
        t0 = t
        delta_t = abs(t-t0)
        if delta_t < 1:
            break

    new_img = np.zeros([m, n])

    for i in range(m):
        for j in range(n):

            if img[i, j] < t:
                new_img[i, j] = 0
            else:
                new_img[i, j] = 225

    return new_img

def phan_doan_kmeans(img, k):
    g = []
    for i in range(k):
        g.append([])
    t = [0] * k

    for i in range(k):
        t[i] = int((256/k) * i + (128/k))

    m, n = img.shape

    while (True):
        for i in range(m):
            for j in range(n):
                if img[i, j] <= t[0]:
                    g[0].append(img[i, j])
                elif img[i, j] >= t[k-1]:
                    g[k-1].append(img[i, j])
                else:
                    for x in range(1, k):
                        if img[i, j] < t[x]:
                            if t[x] - img[i, j] < img[i, j] - t[x - 1]:
                                g[x].append(img[i, j])
                            else:
                                g[x - 1].append(img[i, j])
                            break



        mg = [0] * k
        max_change = 0
        for i in range(k):
            sum = 0
            for v in g[i]:
                sum += v

            if len(g[i]) == 0:
                mg[i] = t[i]
            else:
                mg[i] = sum / len(g[i])

            max_change = max(max_change, abs(mg[i] - t[i]))
            t[i] = mg[i]

        if max_change < 1:
            break

    for i in range(k):
        t[i] = round(t[i])

    new_img = np.zeros([m, n])

    for i in range(m):
        for j in range(n):
            if img[i, j] <= t[0]:
                new_img[i, j] = t[0]
            elif img[i, j] >= t[k - 1]:
                new_img[i, j] = t[k - 1]
            else:
                for x in range(1, k):
                    if img[i, j] < t[x]:
                        if (t[x] - img[i, j]) < (img[i, j] - t[x - 1]):
                            new_img[i, j] = t[x]
                        else:
                            new_img[i, j] = t[x - 1]
                        break

    return new_img