# kmeans clustering
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('lena.jpg').astype(np.float32) / 255.
# 이미지를 RGB가 아닌 LAB 형식의 대이터로 치환함.
# l은 밝기를(1~100), a는 빨강과 초록을, b는 파랑과 노랑을 표현

image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

#이미지를 벡터의 형태로 변환한다.
data = image_lab.reshape((-1, 3))

num_classes = 50

# 분할 처리를 위한 클러스터의 수를 정의(criteria)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,50,0.1)

#cv2의 내장함수 kmeans를 이용한 kmeans clustering
_, labels, centers = cv2.kmeans(data, num_classes, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

segmented_lab = centers[labels.flatten()].reshape(image.shape)
segmented = cv2.cvtColor(segmented_lab, cv2.COLOR_Lab2LBGR)

plt.subplot(121)
plt.axis('off')
plt.title('original')
plt.imshow(image[:,:,[2,1,0]])
plt.subplot(122)
plt.title('segmented')
plt.imshow(segmented)
plt.show()