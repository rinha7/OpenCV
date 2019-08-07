import scipy as sp
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.linalg import inv

from skimage.segmentation import slic


vect = [[0, 10], [10, 10], [20, 20], [30, 30]]
# W = sp.spatial.distance.squareform(sp.spatial.distance.pdist(x))
# W = sp.spatial.distance.pdist(x)
vect2=[]
for i in range(len(vect)):
    for j in range(i+1, len(vect), 1):
        vect2.append(((vect[i][0] - vect[j][0]) ** 2 + (vect[i][1] - vect[j][1]) ** 2) ** 0.5)
vector = sp.spatial.distance.squareform(vect2)
print(vector)