import cv2

src = cv2.imread("ara.jpg",cv2.IMREAD_COLOR)

height, width, channel = src.shape

# 확대
dst = cv2.pyrUp(src,dstsize=(width*2,height*2),borderType=cv2.BORDER_DEFAULT)
# 축소
dst2 = cv2.pyrDown(src);

cv2.imshow("dst",dst)
cv2.imshow("dst2",dst2)

cv2.waitKey(0) # time마다 키 입력상태를 받아옵니다. 0일경우, 여기를 넘어가지 않음.
cv2.destroyAllWindows() # 모든 윈도우창 닫