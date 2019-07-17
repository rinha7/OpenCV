# 이산푸리에 변환의 연산
# cv2.dft는 이산 푸리에변환을

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('ara.jpg',0).astype(np.float32) / 255

fft = cv2.dft(image,flags=cv2.DFT_COMPLEX_OUTPUT)

shifted = np.fft.fftshift(fft, axes=[0,1])
magnitude = cv2.magnitude(shifted[:,:,0],shifted[:,:,1])
magnitude = np.log(magnitude)

plt.axis('off')
plt.imshow(magnitude, cmap='gray')
plt.tight_layout()
plt.show()
restored = cv2.idft(fft,flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
plt.imshow(restored,cmap='gray')
plt.tight_layout()
plt.show()
