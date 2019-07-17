#filter3 자체 필터 만들기

import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('ara.jpg')

KSIZE = 11 # filter의 size
alpha = 2

kernel = cv2.getGaussianKernel(KSIZE,0)

kernel = -alpha * kernel @ kernel.T
kernel[KSIZE//2,KSIZE//2] += 1+alpha

filtered = cv2.filter2D(image,-1,kernel)
# filter2d 함수는 매개변수로 이미지, 출력 결과 데이터 타입, opencv id(입력 이미지 데이터 유형을 유지하려는 경우에는 -1 사용)
# 필터 커널을 입력받음. 이 함수는 이미지를 선형적으로 필터링한다.

plt.figure(figsize=(8,4))
plt.subplot(121)
plt.axis('off')
plt.title('image')
plt.imshow(image[:,:,[2,1,0]])
plt.subplot(122)
plt.axis('off')
plt.title('filtered')
plt.imshow(filtered[:,:,[2,1,0]])
plt.tight_layout(True)
plt.show()