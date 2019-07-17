import cv2

src = cv2.imread("ara.jpg",cv2.IMREAD_COLOR)

dst = src.copy()
dst = src[400:1200, 100:700]

cv2.imshow("dst",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()