import cv2, numpy as np

image = np.full((480,640,3),(0,0,255),np.uint8)
cv2.imshow('red',image)
cv2.waitKey(0)
cv2.destroyAllWindows()