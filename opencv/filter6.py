# 주파수 조작을 통한 이미지 필터링
#

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('ara.jpg',0).astype(np.float32) / 255

fft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)

fft_shift = np.fft.fftshift(fft,axes=[0,1])

sz = 25
mask = np.zeros(fft_shift.shape,np.uint8)
mask[mask.shape[0]//2-sz:mask.shape[0]//2+sz,mask.shape[1]//2-sz:mask.shape[1]//2+sz,:] = 1
fft_shift *= mask

fft = np.fft.ifftshift(fft_shift,axes=[0,1])
filtered = cv2.idft(fft,flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

plt.figure()
plt.subplot(121)
plt.axis('off')
plt.title('original')
plt.imshow(image,cmap='gray')
plt.subplot(122)
plt.axis('off')
plt.title('no high frequencies')
plt.imshow(filtered, cmap='gray')
plt.tight_layout()
plt.show()
