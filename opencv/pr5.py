# image rotation

import cv2

src = cv2.imread("ara.jpg",cv2.IMREAD_COLOR)

height,width,channel = src.shape
matrix = cv2.getRotationMatrix2D((width/2,height/2),90,1)
print(src)
dst = cv2.warpAffine(src,matrix,(width,height))

cv2.imshow("src",src)
cv2.imshow("dst",dst)
cv2.waitKey(0)
cv2.destroyAllWindows()