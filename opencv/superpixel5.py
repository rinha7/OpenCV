import cv2
import numpy as np

# 현재 center들의
def calculate_centers():
    centers = []
    for i in range(S, width - int(S / 2), S):
        for j in range(S, height - int(S / 2), S):
            # 각 center들에 대해 lowest_gradient를 찾아서 center 갱신
            nc = lowest_gradient_position(center=(i,j))
            color = labimg[nc[1],nc[0]]
            temp_cetner = [color[0], color[1], color[2], nc[0], nc[1]]
            centers.append(temp_cetner)

    return centers

def lowest_gradient_position(center):
    min_grad = 1
    local_min = center

    # center의 주변 3x3에 대하여 lowest gradients를 계산함
    for i in range(center[0] - 1, center[0] + 2):
        for j in range(center[1] - 1, center[1] + 2):
            c1 = labimg[j + 1, i]
            c2 = labimg[j, i + 1]
            c3 = labimg[j, i]
            if ((c1[0] - c3[0]) ** 2) ** 0.5 + ((c2[0] - c3[0]) ** 2) ** 0.5 < min_grad:
                min_grad = abs(c1[0] - c3[0]) + abs(c2[0] - c3[0])
                local_min = [i, j]
    return local_min

def generate_pixels():
    # ind_np = numpy.mgrid[0:height, 0:width].swapaxes(0, 2).swapaxes(0, 1)

    for i in range(iterations):
        image_distances = 1* np.ones(image.shape[:2]) # 각 pixel들의 center와의 거리 값들 담고 비교할 행렬 변수
        for j in range(SLIC_centers.shape[0]):
            x_low, x_high = int(SLIC_centers[j][3] - S), int(SLIC_centers[j][3] + S)
            y_low, y_high = int(SLIC_centers[j][4] - S), int(SLIC_centers[j][4] + S)

            if x_low <= 0:
                x_low = 0
            if x_high > width:
                x_high = width
            if y_low <= 0:
                y_low = 0
            if y_high > height:
                y_high = height

            local_image = labimg[y_low:y_high, x_low:x_high] # superpixel로 자른 픽셀 이미지를 저장
            color_diff = local_image - labimg[int(SLIC_centers[j][4]), int(SLIC_centers[j][3])] # lab image에서 center 사이의 차이값을 구합니다
            color_distance = np.sqrt(np.sum(color_diff ** 2 , axies = 2)) #color_diff를 이용하여 local_image의 각 픽셀에 대하여 center와의 거리를 구합니다

            yy, xx = np.ogrid[y_low: y_high, x_low: x_high]
            pix_dist = ((yy - SLIC_centers[j][4]) ** 2 + (xx - SLIC_centers[j][3]) ** 2) ** 0.5 # local_image의 각 픽셀에 대하여 center와의 물리거리를 구합니다.

            dist = color_distance + (m / S) * pix_dist

            distance_crop = image_distances[y_low : y_high, x_low : x_high]
            idx = dist < distance_crop
            distance_crop[idx] = dist[idx]
            image_distances[y_low:y_high, x_low:x_high] = distance_crop
            all_clusters[y_low:y_high, x_low:x_high][idx] = j


image = cv2.imread('lenacolor.png')
K = 1000
S = int((image.shape[0] * image.shape[1] / K) ** 0.5)
m = 40
iterations = 4

height, width = image.shape[:2]
labimg = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float64)

all_distance = 1 * np.ones(image.shape[:2])
all_clusters = -1 * all_distance
center_counts = np.zeros(len(calculate_centers()))
SLIC_centers = np.array(calculate_centers()) # 각 센터들의 5차원 정보값을 담고 있는 배열