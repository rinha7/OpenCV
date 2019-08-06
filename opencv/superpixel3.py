import numpy
import cv2

def generate_pix():
    # numpy mgrid : 매트랩의 meshgrid와 같음.
    # x방향, y방향 mgrid가 indnp에 생성됨.
    indnp = numpy.mgrid[0:height,0:width].swapaxes(0,2).swapaxes(0,1)

    for i in tqdm.tqdm(range(SLIC_INTERACTIONS)):
        distances = 1 * numpy.ones(image.shape[:2])
        for j in range(global_centers.shape[0]):
            x_low, x_high = int(global_centers[j][3] - step), int(global_centers[j][3] + step)
            y_low, y_high = int(global_centers[j][4] - step), int(global_centers[j][4] + step)

            # 영역을 벗어나는 경우 원래 위치로 되돌림.
            if x_low <= 0:
                x_low = 0
            if x_high > width:
                x_high = width

            if y_low <=0:
                y_low =0
            if y_high > height:
                y_high = height

            crop_img = labimg[y_low:y_high,x_low:x_high]
            color_diff = crop_img - labimg[int(global_centers[j][4]),int(global_centers[j][3])]
            color_distance = numpy.sqrt(numpy.sum(numpy.square(color_diff),axis=2))

            yy, xx = numpy.ogrid[y_low : y_high , x_low : x_high]
            pix_dist = ((yy-global_centers[j][4]) ** 2 + (xx-global_centers[j][3])**2)**0.5

            dist = ((color_distance/SLIC_m)**2 + (pix_dist/step)**2)**0.5

            distance_crop = distances[y_low : y_high , x_low :x_high]

            idx = dist < distance_crop
            distance_crop[idx] = dist[idx]

            distances[y_low : y_high, x_low : x_high] = distance_crop
            clusters[y_low : y_high, x_low : x_high][idx] = j

        for k in range(len(global_centers)):
            idx = (clusters == k)
            color_np = labimg[idx]
            dist_np = indnp[idx]

            global_centers[k][0:3] = numpy.sum(color_np, axis=0)
            sumy, sumx = numpy.sum(dist_np, axis=0)
            global_centers[k][3:] = sumx, sumy
            global_centers[k] /= numpy.sum(idx)

# 화면에 표시하기 위한 connectivity 작성
def create_connectivity():
    label = 0
    adj_label = 0
    lims = int(width * height / global_centers.shape[0])

    new_clusters = -1 * numpy.ones(image.shape[:2]).astype(numpy.int64)
    elements = []


    for i in range(width):
        for j in range(height):
            if new_clusters[j, i] == -1:
                elements = []
                elements.append((j,i))
                for dx, dy in [(-1,0), (0,-1), (1,0), (0,1)]:
                    x = elements[0][1] + dx
                    y = elements[0][0] + dy
                    if x >= 0 and x < width and y >= 0 and y < height and new_clusters[y,x] >= 0:
                        adj_label = new_clusters[y,x]
            count = 1
            counter = 0

            while counter < count:
                for dx, dy in [(-1,0), (0,-1), (1,0), (0,1)]:
                    x = elements[counter][1] + dx
                    y = elements[counter][0] + dy

                    if x >= 0 and x < width and y >= 0 and y < height:
                        if new_clusters[y,x] == -1 and clusters[j,i] == clusters[y,x]:
                            elements.append((y, x))
                            clusters[y,x] = label
                            count += 1
                counter += 1
            if count <= lims >> 2:
                for counter in range(count):
                    new_clusters[elements[counter]] = adj_label
                label -=1
            label +=1
    # SLIC_new_clusters = new_clusters

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
                    if is_taken[y, x] == False and clusters[j, i] != clusters[y, x]:
                        nr_p += 1
            if nr_p >= 2:
                is_taken[j,i] = True
                contours.append([j,i])
    for i in range(len(contours)):
        image[contours[i][0], contours[i][1]] = color

def find_local_minimum(center):
    min_grad = 1
    loc_min = center

    for i in range(center[0] - 1, center[0] + 2):
        for j in range(center[1] - 1, center[1] + 2):
            c1 = labimg[j+1, i]
            c2 = labimg[j, i+1]
            c3 = labimg[j,i]

            if ((c1[0]-c3[0]) ** 2) ** 0.5 + ((c2[0] - c3[0]) ** 2) ** 0.5 < min_grad:
                min_grad = abs(c1[0] - c3[0]) + abs(c2[0] - c3[0])
                loc_min = [i,j]
    return loc_min


def caculate_centers():
    centers = []
    for x in range(step, width - int(step/2), step):
        for y in range(step, height - int(step/2), step):
            nc = find_local_minimum(center=(x,y))
            color = labimg[nc[1],nc[0]]
            center = [color[0],color[1],color[2],nc[0],nc[1]]
            centers.append(center)
    return centers

image = cv2.imread('lenacolor.png')

height,width = image.shape[:2]
step = int((image.shape[0]*image.shape[1]/ int(1000))**0.5)
SLIC_m = int(40)
SLIC_INTERACTIONS = 4

# lab image로 convert
labimg = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(numpy.float64)
distances = 1 * numpy.ones(image.shape[:2])
clusters = -1 * distances
center_counts = numpy.zeros(len(caculate_centers()))
global_centers = numpy.array(caculate_centers())

# run

generate_pix()
create_connectivity()
caculate_centers()
display_contours([0.0,0.0,0.0])
cv2.imshow("superpixels",image)
cv2.waitKey(0)
cv2.imwrite("sliclena.jpg",image)