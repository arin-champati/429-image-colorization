import cv2
import numpy as np
from skimage import color
from sklearn.neighbors import NearestNeighbors

gamut = np.load('Data/pts_in_hull.npy', allow_pickle=True)
nn = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(gamut)

def load_img(path):
    # im = np.asarray(Image.open(path))
    # if im.ndim == 2:
    #     return np.tile(im[:, :, None], 3)

    # reverse im_color from BGR to RGB
    im = cv2.imread(path)
    im = im[:,:,::-1]

    return im


def resize_img(img, size, resample=3):
	# return np.asarray(Image.fromarray(img).resize((size, size), resample=resample))
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

def get_ab(img):
    return color.rgb2lab(img)[:, :, 1:3]

def get_qab(img_ab, sigma=5, bins=313):

    # gets num_nb nearest neighbors
    h, w, _ = img_ab.shape
    a = np.ravel(img_ab[:, :, 0])
    b = np.ravel(img_ab[:, :, 1])
    ab = np.vstack((a, b)).T # now dim (H*W*2)
    distances, idx = nn.kneighbors(ab)

    # smoothen
    gaussian = np.exp(-distances**2 / (2*sigma**2))
    gaussian /= np.sum(gaussian, axis=1)[:, np.newaxis]
    soft_encoding = np.zeros((ab.shape[0], bins))
    pts = np.arange(ab.shape[0])[:, np.newaxis]
    soft_encoding[pts, idx] = gaussian
    return soft_encoding.reshape(h, w, bins)


def get_data(path, size=64):
    img = resize_img(load_img(path), size)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_ab = get_ab(img)
    return img_gray, img_ab
