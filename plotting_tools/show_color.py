import numpy as np
from skimage import color
from helpers.load_cifar import get_cifar100_data
import cv2
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.image as mpimg


def lab():
    L = np.ones((11, 11)) * 30
    # print(list(np.load('weight81.npy')))
    a = np.repeat(np.arange(-110, 110, 20), 11, axis=-1)
    a = np.reshape(a, (11, 11))

    b = a.T
    lab = np.dstack((L, a, b))
    rgb = color.lab2rgb(lab)
    rgb = cv2.resize(rgb, (110, 110))

    plt.title('(b)')
    plt.xlabel('b')
    plt.ylabel('a')
    plt.imshow(rgb, extent=[-110, 110, 110, -110])


def yuv():
    Y = np.ones((10, 10)) * 0.5
    u = np.repeat(np.arange(-0.5, 0.5, 0.1), 10, axis=-1)
    u = np.reshape(u, (10, 10))
    v = u.T
    yuv = np.dstack((Y, u, v))
    rgb = color.yuv2rgb(yuv)
    rgb = cv2.resize(rgb, (110, 110))
    plt.title('(c)')
    plt.xlabel('v')
    plt.ylabel('u')
    plt.imshow(rgb, extent=[-0.5, 0.5, 0.5, -0.5])


def rgb():
    rgb = mpimg.imread(
        '/Users/pipy/Downloads/429-image-colorization/M99Iz.jpg')
    rgb = cv2.resize(rgb, (110, 110))
    plt.title('(a)')
    plt.axis('off')
    plt.imshow(rgb)


def plot_colors():
    fig = plt.figure(figsize=(24, 8))
    columns = 3
    rows = 1
    fig.add_subplot(1, 3, 1)
    rgb()
    fig.add_subplot(1, 3, 2)
    lab()
    fig.add_subplot(1, 3, 3)
    yuv()

    plt.subplots_adjust(wspace=0.5)
    plt.show()


def plot_annealed_colors():
    fig = plt.figure(figsize=(60, 10))
    fig.add_subplot(1, 6, 1)
    plt.title('T = 0')
    plt.axis('off')
    im = mpimg.imread('temp/1.png')
    plt.imshow(im)

    fig.add_subplot(1, 6, 2)
    plt.title('T = 0.25')
    plt.axis('off')
    im = mpimg.imread('temp/25.png')
    plt.imshow(im)

    fig.add_subplot(1, 6, 3)
    plt.title('T = 0.38')
    plt.axis('off')
    im = mpimg.imread('temp/38.png')
    plt.imshow(im)

    fig.add_subplot(1, 6, 4)
    plt.title('T = 0.50')
    plt.axis('off')
    im = mpimg.imread('temp/50.png')
    plt.imshow(im)

    fig.add_subplot(1, 6, 5)
    plt.title('T = 0.75')
    plt.axis('off')
    im = mpimg.imread('temp/75.png')
    plt.imshow(im)

    fig.add_subplot(1, 6, 6)
    plt.title('T = 1')
    plt.axis('off')
    im = mpimg.imread('temp/100.png')
    plt.imshow(im)


plot_annealed_colors()
plt.show()
