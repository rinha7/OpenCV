from scipy.optimize import curve_fit
import numpy as np

def find_zero(array):
    count = 0
    arr = []
    for x in range(array.shape[0]):
        for y in range(array.shape[1]):
            count += 1
            if array[y][x] != 0:
                arr.append(count)


    return arr



a = np.ones([3,3],dtype=np.int32)
a[0][1] = 0
a[1][0] = 0
a[2][0] = 0
a[2][1] = 0

print(a)

f = np.transpose(np.nonzero(a))
zero = find_zero(a)
print(np.asarray(zero))