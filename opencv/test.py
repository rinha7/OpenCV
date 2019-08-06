from . import manifold1
import matplotlib.pyplot as plt

if __name__ == '__main__':
    mr = manifold1.MR_saliency()
    sal = mr.saliency('lenacolor.png')

    plt.imshow(sal)
    plt.show()
