import numpy as np
from sklearn.neighbors import NearestNeighbors
from helpers.load_cifar import get_cifar10_data, get_cifar100_data
from skimage import color
from scipy.ndimage import gaussian_filter
from matplotlib.image import NonUniformImage
import matplotlib.pyplot as plt
import cv2

# gets ab gamut distribution


def get_color_histogram_ab():
    cifar10 = get_cifar10_data()
    cifar10_ab = color.rgb2lab(cifar10)[1:]
    cifar10_ab = np.reshape(cifar10_ab, (-1, 2))
    a = cifar10_ab[:, 0]
    b = cifar10_ab[:, 1]
    H, xedges, yedges = np.histogram2d(a, b, bins=9)
    return H, xedges, yedges

# get uv gamut distribution


def get_color_histogram_uv():
    cifar10 = get_cifar10_data()
    cifar10_ab = color.rgb2yuv(cifar10)[1:]
    cifar10_ab = np.reshape(cifar10_ab, (-1, 2))
    a = cifar10_ab[:, 0]
    b = cifar10_ab[:, 1]
    H, xedges, yedges = np.histogram2d(a, b, bins=9)
    return H, xedges, yedges


# rebalance
def rebalance(H):
    H_smooth = gaussian_filter(H, sigma=3)
    H_smooth /= np.sum(H_smooth)
    H_mix = 0.8 * H_smooth + 0.2/81
    w = 1 / H_mix
    H = w / np.sum(w*H_smooth)
    return H, H_smooth


H1, xab, yab = get_color_histogram_ab()
probab, Hab = rebalance(H1)
H2, xuv, yuv = get_color_histogram_uv()
probuv, Huv = rebalance(H2)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(231, title='(a)', aspect='equal',
                     xlim=xab[[0, -1]], ylim=yab[[0, -1]])
im = NonUniformImage(ax, interpolation='bilinear')
ax.set_xlabel('b')
ax.set_ylabel('a')
xcenters = (xab[:-1] + xab[1:]) / 2
ycenters = (yab[:-1] + yab[1:]) / 2
im.set_data(xcenters, ycenters, H1)
ax.images.append(im)


ax = fig.add_subplot(232, title='(b)', aspect='equal',
                     xlim=xab[[0, -1]], ylim=yab[[0, -1]])
im = NonUniformImage(ax, interpolation='bilinear')
ax.set_xlabel('b')
ax.set_ylabel('a')
im.set_data(xcenters, ycenters, Hab)
ax.images.append(im)


ax = fig.add_subplot(233, title='(c)', aspect='equal',
                     xlim=xab[[0, -1]], ylim=yab[[0, -1]])
ax.set_xlabel('b')
ax.set_ylabel('a')
im = NonUniformImage(ax, interpolation='bilinear')
im.set_data(xcenters, ycenters, probab)
ax.images.append(im)

ax = fig.add_subplot(234, title='(d)', aspect='equal',
                     xlim=xuv[[0, -1]], ylim=yuv[[0, -1]])
im = NonUniformImage(ax, interpolation='bilinear')
ax.set_xlabel('v')
ax.set_ylabel('u')
xcenters = (xuv[:-1] + xuv[1:]) / 2
ycenters = (yuv[:-1] + yuv[1:]) / 2
im.set_data(xcenters, ycenters, H2)
ax.images.append(im)


ax = fig.add_subplot(235, title='(e)',
                     aspect='equal', xlim=xuv[[0, -1]], ylim=yuv[[0, -1]])
im = NonUniformImage(ax, interpolation='bilinear')
ax.set_xlabel('v')
ax.set_ylabel('u')
im.set_data(xcenters, ycenters, Huv)
ax.images.append(im)

ax = fig.add_subplot(236, title='(f)',
                     aspect='equal', xlim=xuv[[0, -1]], ylim=yuv[[0, -1]])
im = NonUniformImage(ax, interpolation='bilinear')
ax.set_xlabel('v')
ax.set_ylabel('u')
im.set_data(xcenters, ycenters, probuv)
ax.images.append(im)
plt.subplots_adjust(hspace=.3, wspace=.3)
plt.show()

# print(H.shape, H)
