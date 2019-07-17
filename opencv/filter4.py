# float 값으로 된 가버 필터로 이미지 처리하기

import math
import  cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('ara.jpg').astype(np.float32)/255

kernel = cv2.getGaborKernel((21,21), 5,1,10,1,0,cv2.CV_32F)
kernel /= math.sqrt((kernel*kernel).sum())

filtered = cv2.filter2D(image,-1,kernel)
#image를 받아다 -1 속성으로 kernel을 입힘.

plt.figure(figsize=(8,3))
plt.subplot(131)
plt.axis('off')
plt.title('image')
# 원본 이미지를 plt에 출력

plt.imshow(image, cmap='gray')
plt.subplot(132)
plt.title('kernel')
plt.imshow(kernel,cmap='gray')
# kernel이 어떻게 생겼는지에 대해 표시합니다.

plt.subplot(133)
plt.axis('off')
plt.title('filtered')
plt.imshow(filtered,cmap='gray')
plt.tight_layout()
plt.show()
# filter를 적용한 이후의 모습을 출력합니다.

# 가버필터는 코사인파로 변조된 2D 가우시안 커널을 갖는 선형필터입니다.
# 가버필터는 이미지의 방향을 아는 경우 에지를 검출하는 데에 유용한 필터이다.