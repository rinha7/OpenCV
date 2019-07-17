# 이미지 히스토그램 평활화 하기
# 히스토그램 평활화란, 이미지 히스토그램으로 표현된 이미지의 강도를
# 평활화를 통해 균일하게 만듬으로써 명암구분을 늘려 이미지를 선명하게 하는 효과가 있다.
# OpenCV에서는 cv2.equalizeHist를 통해 평활화를 진행한다.

import cv2
import numpy as np
import matplotlib.pyplot as plt

greyimage = cv2.imread('child.png', 0) # grey image 로 이미지를 읽어옵니다.
cv2.imshow('original grey', greyimage) # 히스토그램 평탄화 하기 전의 이미지를 화면에 띄웁니다.

grey_eq = cv2.equalizeHist(greyimage) # 히스토그램 평탄화를 실행

hist,bins = np.histogram(grey_eq, 256, [0,255]) # 평탄화 한 이미지의  histogram을 hist에 저장합니다.
# histogram 출력
plt.fill_between(range(256),hist,0)
plt.xlabel('pixel value')
plt.show()

cv2.imshow('eq grey', grey_eq)

# color = cv2.imread('ara.jpg')
# hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV) # 뭐임 이거
# cv2.imshow('original color', color)
#
# hsv[..., 2] = cv2.equalizeHist(hsv[..., 2])
# color_eq = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
# cv2.imshow('original color',color)
#
#
# cv2. imshow('equalized color', color_eq)

cv2.waitKey(0)
cv2.destroyAllWindows()