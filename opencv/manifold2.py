import scipy as sp
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.linalg import inv

from skimage.segmentation import slic

alpha = 0.99
delta = 0.1
segs = 200
compactness = 10
max_iter = 10
sigma = 1
spacing = None
multichannel = True
convert2lab = None
enfoce_connectivity = False
min_size_factor = 0.5
max_size_factor = 3
slic_zero = False
binary_thres = None


# image를 불러들여와 lab 영상으로 바꾸어 줍니다.
def read_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(float) / 255
    h = 100
    w = int(float(h) / float(img.shape[0]) * float(img.shape[1]))
    return cv2.resize(img, (w, h))


def superpixel_label(img):
    return slic(img, segs, compactness, max_iter, sigma, spacing, multichannel, convert2lab, enfoce_connectivity,
                min_size_factor, max_size_factor, slic_zero)


def superpixel_mean_vector(img, labels):
    s = sp.amax(labels) + 1
    vec = sp.zeros((s,3)).astype(float)
    for i in range(s):
        mask = labels == i
        super_v = img[mask].astype(float)
        mean = sp.mean(super_v,0)
        vec[i] = mean
    return vec


def adj_loop(labels):
    s = sp.amax(labels) + 1

    adj = np.ones((s, s), np.bool)

    for i in range(labels.shape[0] - 1):
        for j in range(labels.shape[1] - 1):
            if labels[i, j] != labels[i + 1, j]:
                adj[labels[i, j], labels[i + 1, j]] = False
                adj[labels[i + 1, j], labels[i, j]] = False
            if labels[i, j] != labels[i, j + 1]:
                adj[labels[i, j], labels[i, j + 1]] = False
                adj[labels[i, j + 1], labels[i, j]] = False
            if labels[i, j] != labels[i + 1, j + 1]:
                adj[labels[i, j], labels[i + 1, j + 1]] = False
                adj[labels[i + 1, j + 1], labels[i, j]] = False
            if labels[i + 1, j] != labels[i, j + 1]:
                adj[labels[i + 1, j], labels[i, j + 1]] = False
                adj[labels[i, j + 1], labels[i + 1, j]] = False

    upper_ids = sp.unique(labels[0, :]).astype(int)
    right_ids = sp.unique(labels[:, labels.shape[1] - 1]).astype(int)
    low_ids = sp.unique(labels[labels.shape[0] - 1, :]).astype(int)
    left_ids = sp.unique(labels[:, 0]).astype(int)

    bd = np.append(upper_ids, right_ids)
    bd = np.append(bd, low_ids)
    bd = sp.unique(np.append(bd, left_ids))

    for i in range(len(bd)):
        for j in range(i + 1, len(bd)):
            adj[bd[i], bd[j]] = False
            adj[bd[j], bd[i]] = False

    return adj


def build_matrix(img, labels):
    s = sp.amax(labels) + 1
    vect = superpixel_mean_vector(img, labels)

    adj = adj_loop(labels)

    W = sp.spatial.distance.squareform(sp.spatial.distance.pdist(vect))

    W = sp.exp(-1 * W / delta)
    W[adj.astype(np.bool)] = 0

    D = sp.zeros((s, s)).astype(float)
    for i in range(s):
        D[i, i] = sp.sum(W[i])

    return W, D


# manifold ranking
def affinity_matrix(img, labels):
    W, D = build_matrix(img, labels)
    aff = inv(D - (alpha * W))
    aff[sp.eye(sp.amax(labels) + 1).astype(bool)] = 0.0  # 대각행렬을 0으로 만들어줍니다.
    return aff

def boundary_indictor(labels):
    s = sp.amax(labels) + 1

    up_indictor = (sp.ones((s,1))).astype(float)
    right_indictor = (sp.ones((s,1))).astype(float)
    low_indictor = (sp.ones((s, 1))).astype(float)
    left_indictor = (sp.ones((s, 1))).astype(float)

    upper_idx = sp.unique(labels[0, :]).astype(int)
    right_idx = sp.unique(labels[:, labels.shape[1] - 1]).astype(int)
    lower_idx = sp.unique(labels[labels.shape[0] - 1, :]).astype(int)
    left_idx = sp.unique(labels[:, 0]).astype(int)

    up_indictor[upper_idx] = 0.0
    right_indictor[right_idx] = 0.0
    low_indictor[lower_idx] = 0.0
    left_indictor[left_idx] = 0.0

    return up_indictor,right_indictor,low_indictor,left_indictor

def fill_superpixel_with_saliency(labels, sal):
    sal_img = labels.copy().astype(float)

    for i in range(sp.amax(labels) + 1):
        mask = labels == i
        sal_img[mask] = sal[i]
    return cv2.normalize(sal_img,None,0,255,cv2.NORM_MINMAX)

def first_stage(aff, labels):
    up, right, low, left = boundary_indictor(labels)

    up_sal = 1-MR_saliency(aff, up)
    up_img = fill_superpixel_with_saliency(labels,up_sal)

    right_sal = 1 - MR_saliency(aff, right)
    right_img = fill_superpixel_with_saliency(labels, right_sal)

    low_sal = 1 - MR_saliency(aff, low)
    low_img = fill_superpixel_with_saliency(labels, low_sal)

    left_sal = 1 - MR_saliency(aff, left)
    left_img = fill_superpixel_with_saliency(labels, left_sal)

    return 1 - up_img * right_img * left_img * low_img

def final_stage(integrated_sal, labels, aff):
    if binary_thres == None:
        thres = sp.median(integrated_sal .astype(float))

    mask = integrated_sal > thres

    ind = second_stage_indicator(mask,labels)

    return MR_saliency(aff, ind)

def second_stage_indicator(mask, labels):
    s = sp.amax(labels) + 1

    ids = sp.unique(labels[mask]).astype(int)

    indictor = sp.zeros((s,1)).astype(float)
    indictor[ids] = 1.0
    return indictor

def MR_saliency(aff, indictor):
    return sp.dot(aff, indictor)


def saliency(img):
    img = read_img(img)
    labels = superpixel_label(img)
    aff = affinity_matrix(img, labels)
    first_saliency = first_stage(aff, labels)
    final_saliency = final_stage(first_saliency, labels, aff)

    return fill_superpixel_with_saliency(labels, final_saliency)

sal = saliency(cv2.imread('ara.jpg'))

plt.imshow(sal)
plt.show()