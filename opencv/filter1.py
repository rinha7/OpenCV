# Filtering

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('ara.jpg').astype(np.float32)/255

noised = (image + 0.2 *
          np.random.rand(*image.shape).astype(np.float32))
noised = noised.clip(0,1)
plt.imshow(noised[:,:,[2,1,0]])
plt.xlabel('noise')
plt.show()

gaus_blur = cv2.GaussianBlur(noised,(7,7),0)
plt.imshow(gaus_blur[:,:,[2,1,0]])
plt.xlabel('gausian')
plt.show()

median_blur = cv2.medianBlur((noised * 255).astype(np.uint8),7)
plt.imshow(median_blur[:,:,[2,1,0]])
plt.show()

bilat = cv2.bilateralFilter(noised, -1, 0.3, 10)
plt.imshow(bilat[:,:,[2,1,0]])
plt.show()