# edge detect 1
# 이진 이미지에서 외부 및 내부 구분, 윤곽선 찾

import cv2, numpy as np, matplotlib.pyplot as plt

image = cv2.imread('ara.jpg',cv2.IMREAD_GRAYSCALE)
otsu_thres, otsu_mask = cv2.threshold(image,-1,1,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# 외부 및 내부의 윤곽선을 찾아 2단계의 계층으로 조직화합니다.
# findContours는 윤곽을 찾아주는 함수입니다. 인자로 image, RETR_CCOMP는 mode로, 2단계로 계층을
# 표현합니다.  CHAIN_APPROX_SIMPLE은 수평, 수직, 대각선이 엣지에 포함됩니다.

contours, hierarchy = cv2.findContours(otsu_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

eximage = np.zeros(image.shape, image.dtype)
for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(eximage, contours, i, 255, -1)

inimage = np.zeros(image.shape, image.dtype)
for i in range(len(contours)):
    if hierarchy[0][1][3] != -1:
        cv2.drawContours(inimage, contours, i, 255, -1)

# 3개짜리 figure 실행
plt.figure(figsize=(10,3))
plt.subplot(131)
plt.axis('off')
plt.title('original')
plt.imshow(image, cmap='gray')
plt.subplot(132)
plt.axis('off')
plt.title('external image')
plt.imshow(eximage,cmap='gray')
plt.subplot(133)
plt.axis('off')
plt.title('internal image')
plt.imshow(inimage, cmap='gray')
plt.tight_layout()
plt.show()
