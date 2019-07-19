# 분할 시드를 이용하여 이미지를 분할하는 워터쉐드 알고리즘
# 초기에 분할되는 점이 있고, 주변 영역을 같은 클래스로 채우는 방법을 사용한다.
# 랜덤으로 값을 주고, 주변 값을 채워나간다.

import cv2
import numpy as np, random

from random import randint

image = cv2.imread('ara.jpg')

# 분할 결과를 저장할 사본이미지
show_img = np.copy(image)

# 초기에 분할되는점인 시드를 만든다.
seeds = np.full(image.shape[0:2],0,np.int32)
segmentation = np.full(image.shape, 0, np.uint8)

n_seeds = 9 # seed case

color=[] # seed color

for m in range(n_seeds):
    color.append((255* m / n_seeds, randint(0,255), randint(0,255)))

mouse_pressed = False
current_seed = 1
seed_update = False

def mouse_callback(event, x,y,flags,param):
    global mouse_pressed, seed_update

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        cv2.circle(seeds,(x,y),5,(current_seed),cv2.FILLED)
        cv2.circle(show_img,(x,y),5,color[current_seed-1],cv2.FILLED)
        seed_update = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            cv2.circle(seeds,(x,y),5,(current_seed), cv2.FILLED)
            cv2.circle(show_img, (x,y), 5, color[current_seed-1],cv2.FILLED)
            seed_update = True
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False


cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)

while True:
    cv2.imshow('segmentation',segmentation)
    cv2.imshow('image', show_img)
    k = cv2.waitKey(1)

    if k == 27:
        break
    elif k == ord('c'):
        show_img = np.copy(image)
        seeds = np.full(image.shape[0:2], 0, np.int32)
        segmentation = np.full(image.shape, 0, np.uint8)
    elif k > 0 and chr(k).isdigit():
        n = int(chr(k))
        if 1 <= n <= n_seeds and not mouse_pressed:
            current_seed = n
    if seed_update and not mouse_pressed:
        seeds_copy = np.copy(seeds)
        cv2.watershed(image, seeds_copy)
        segmentation = np.full(image.shape,0,np.uint8)
        for m in range(n_seeds):
            segmentation[seeds_copy == (m + 1)] == color[m]
        seed_update = False
cv2.destroyAllWindows()