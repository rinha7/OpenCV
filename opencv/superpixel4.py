import numpy
import cv2


def generate_pixels():
    # iteration 만큼 반복하면서 수행
    for i in range(iterations):
        current_distances = 1 * numpy.ones(image.shape[:2])
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

            crop_img = labimg[y_low: y_high, x_low: x_high]
            color_diff = crop_img - labimg[int(SLIC_centers[j][4]), int(SLIC_centers[j][3])] # center와 각 픽셀의 차잇값을 담은 배열 생성
            color_distance = numpy.sqrt(numpy.sum(color_diff**2,axis=2)) # 각 픽셀들의 color_diff에 대한 center와의 거리를 구합니다.

            yy, xx = numpy.ogrid[y_low: y_high, x_low: x_high]
            pix_dist = ((yy - SLIC_centers[j][4]) ** 2 + (xx - SLIC_centers[j][3]) ** 2) ** 0.5

            dist = ((color_distance / m) ** 2 + (pix_dist / S) ** 2) ** 0.5
            # dist = color_distance + (m/S)*pix_dist


            # if j < SLIC_centers.shape[0]-1:
            #     x_low2, x_high2 = int(SLIC_centers[j+1][3] - S), int(SLIC_centers[j+1][3] + S)
            #     y_low2, y_high2 = int(SLIC_centers[j+1][4] - S), int(SLIC_centers[j+1][4] + S)
            #
            #     if x_low2 <= 0:
            #         x_low2 = 0
            #     if x_high2 > width:
            #         x_high2 = width
            #     if y_low2 <= 0:
            #         y_low2 = 0
            #     if y_high2 > height:
            #         y_high2 = height
            #
            #     crop_img2 = labimg[y_low2: y_high2, x_low2: x_high2]
            #     color_diff2 = crop_img2 - labimg[int(SLIC_centers[j+1][4]), int(SLIC_centers[j+1][3])]  # center와 각 픽셀의 차잇값을 담은 배열 생성
            #     color_distance2 = numpy.sqrt(numpy.sum(color_diff2 ** 2, axis=2))  # 각 픽셀들의 color_diff에 대한 center와의 거리를 구합니다.
            #
            #     yy2, xx2 = numpy.ogrid[y_low2: y_high2, x_low2: x_high2]
            #     pix_dist2 = ((yy2 - SLIC_centers[j+1][4]) ** 2 + (xx2 - SLIC_centers[j+1][3]) ** 2) ** 0.5
            #
            #     dist2 = color_distance2 + (m / S) * pix_dist2


            distance_crop = current_distances[y_low: y_high, x_low: x_high]
            idx = dist < distance_crop
            distance_crop[idx] = dist[idx]
            current_distances[y_low: y_high, x_low: x_high] = distance_crop
            SLIC_clusters[y_low: y_high, x_low: x_high][idx] = j

def create_connectivity():
    label = 0
    adj_label = 0
    lims = int(width * height / SLIC_centers.shape[0])

    new_clusters = -1 * numpy.ones(image.shape[:2]).astype(numpy.int64)
    elements = []
    for i in range(width):
        for j in range(height):
            if new_clusters[j, i] == -1:
                elements = []
                elements.append((j, i))
                for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                    x = elements[0][1] + dx
                    y = elements[0][0] + dy
                    if (x >= 0 and x < width and
                            y >= 0 and y < height and
                            new_clusters[y, x] >= 0):
                        adj_label = new_clusters[y, x]

            count = 1
            counter = 0
            while counter < count:
                for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                    x = elements[counter][1] + dx
                    y = elements[counter][0] + dy

                    if (x >= 0 and x < width and y >= 0 and y < height):
                        if new_clusters[y, x] == -1 and SLIC_clusters[j, i] == SLIC_clusters[y, x]:
                            elements.append((y, x))
                            new_clusters[y, x] = label
                            count += 1

                counter += 1
            if (count <= lims >> 2):
                for counter in range(count):
                    new_clusters[elements[counter]] = adj_label

                label -= 1

            label += 1

def display_contours(color):
    is_taken = numpy.zeros(image.shape[:2], numpy.bool)
    contours = []

    for i in range(width):
        for j in range(height):
            nr_p = 0
            for dx, dy in [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]:
                x = i + dx
                y = j + dy
                if x >= 0 and x < width and y >= 0 and y < height:
                    if is_taken[y, x] == False and SLIC_clusters[j, i] != SLIC_clusters[y, x]:
                        nr_p += 1

            if nr_p >= 2:
                is_taken[j, i] = True
                contours.append([j, i])

    for i in range(len(contours)):
        image[contours[i][0], contours[i][1]] = color

# def compare_distance(center):



#find local minimu in 3x3 neighborhood
#center를 기준으로 근처 3x3에서 최소값 탐
def find_local_minimum(center):
    min_grad = 1
    loc_min = center
    for i in range(center[0] - 1, center[0] + 2):
        for j in range(center[1] - 1, center[1] + 2):
            c1 = labimg[j + 1, i]
            c2 = labimg[j, i + 1]
            c3 = labimg[j, i]
            if ((c1[0] - c3[0]) ** 2) ** 0.5 + ((c2[0] - c3[0]) ** 2) ** 0.5 < min_grad:
                min_grad = abs(c1[0] - c3[0]) + abs(c2[0] - c3[0])
                loc_min = [i, j]

    return loc_min



def calculate_centers():
    centers = []
    for i in range(S, width - int(S / 2), S):
        for j in range(S, height - int(S / 2), S):
            nc = find_local_minimum(center=(i, j))
            color = labimg[nc[1], nc[0]]
            center = [color[0], color[1], color[2], nc[0], nc[1]]
            centers.append(center)

    return centers
def calculate_center_again():
    centers = []
    for i in range(SLIC_centers.shape[0]):
        nc = find_local_minimum(center=(SLIC_centers[i][3],SLIC_centers[i][4]))
        color = labimg[nc[1], nc[0]]
        center = [color[0], color[1], color[2], nc[0], nc[1]]
        centers.append(center)

    return centers


image = cv2.imread('lenacolor.png')
K = 1000
S = int((image.shape[0] * image.shape[1] / K) ** 0.5)
m = 40
iterations = 4
# mK = 40

height, width = image.shape[:2]
labimg = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(numpy.float64)
global_distance = 1 * numpy.ones(image.shape[:2])
SLIC_clusters = -1 * global_distance
center_counts = numpy.zeros(len(calculate_centers()))
SLIC_centers = numpy.array(calculate_centers())


generate_pixels()
create_connectivity()
centers = calculate_centers()
SLIC_centers = numpy.asarray(centers)
# cetner2 = calculate_center_again()
display_contours([0.0, 0.0, 0.0])
cv2.imshow("superpixels", image)
cv2.waitKey(0)
cv2.destroyAllWindows()