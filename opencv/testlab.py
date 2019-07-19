import cv2, numpy as np

image = cv2.imread('child.png',cv2.IMREAD_GRAYSCALE)

row, col = image.shape

print(row, col)

cv2.circle()