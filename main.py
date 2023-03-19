import BasicProcessor
import LinearFilter
import NonLinearFilter
import Segmentaion
import cv2
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(16, 9))
(ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12) = fig.subplots(4, 3)

image = cv2.imread('test1.tif', 0)
ax1.imshow(image, cmap='gray')
ax1.set_title("Ảnh gốc")

# histogram = BasicProcessor.calculate_histogram(image)
# ax2.plot(histogram)
# ax2.set_title("Histogram")
#
# linear = LinearFilter.gaussian_blur(image)
# ax3.imshow(linear, cmap='gray')
# ax3.set_title("Gauss Blur")
#
# linear = LinearFilter.mean_filter(image)
# ax4.imshow(linear, cmap='gray')
# ax4.set_title("Mean Filter")
#
# linear = LinearFilter.sobel_edge_detect(image)
# ax5.imshow(linear, cmap='gray')
# ax5.set_title("Sobel")
#
# linear = LinearFilter.laplace_edge_detect(image)
# ax6.imshow(linear, cmap='gray')
# ax6.set_title("Laplace")
#
# linear = NonLinearFilter.median_filter(image)
# ax7.imshow(linear, cmap='gray')
# ax7.set_title("Median Filter")
#
# linear = NonLinearFilter.max_filter(image)
# ax8.imshow(linear, cmap='gray')
# ax8.set_title("Max Filter")
#
# linear = NonLinearFilter.min_filter(image)
# ax9.imshow(linear, cmap='gray')
# ax9.set_title("Min Filter")
#
# linear = BasicProcessor.equalize_histogram(image)
# ax10.imshow(linear, cmap='gray')
# ax10.set_title("Equalize")
#
# histogram = BasicProcessor.calculate_histogram(linear)
# ax11.plot(histogram)
# ax11.set_title("Histogram")

linear = Segmentaion.phan_doan_kmeans(image, 3)
ax12.imshow(linear, cmap='gray')
ax12.set_title("Robert")
# Hiển thị vùng vẽ
plt.show()