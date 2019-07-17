# otsu 의 바법을 이용한 그레이스케일 이미지 이진화
# 클래스를 추출하는데 otsu의 방법을 사용하여 그레이스케일 이미지를 이진 이미지로 변환한다

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 이미지를 흑백으로 읽어옵니다.
image = cv2.imread('ara.jpg',cv2.IMREAD_GRAYSCALE)

otsu_thres, otsu_mask = cv2.threshold(image,-1,1,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print("측정된 임계값 : ", otsu_thres)

plt.figure()
plt.subplot(121)
plt.axis('off')
plt.title('original')
plt.imshow(image,cmap='gray')
plt.subplot(122)
plt.axis('off')
plt.title('otsu')
plt.imshow(otsu_mask, cmap='gray')
plt.tight_layout()
plt.show()