import cv2 ,numpy as np, matplotlib.pyplot as plt

greyimage = cv2.imread('child.png',0) # 이미지를 흑백으로 불러옵니다.

hist, bins = np.histogram(greyimage,256,[0,256]) # numpy를 통해 histogram화

cv2.imshow("original",greyimage)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 히스토그램을 화면에 표시
plt.fill(hist)
plt.xlabel('pixel 값')
plt.show()