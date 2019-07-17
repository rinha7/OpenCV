import cv2, numpy as np
import matplotlib.pyplot as plt

grey = cv2.imread('ara.jpg',cv2.IMREAD_GRAYSCALE)

# cv2.imshow("img",grey)

hist, bins = np.histogram(grey,256,[0,255])

plt.fill(hist)
plt.xlabel('pixel value')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()