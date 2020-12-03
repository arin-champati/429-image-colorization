import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import cv2
import scipy.io
from skimage.transform import resize
from skimage import color


DATASET_PATH = './Data/Stanford Dog Dataset'
PICKLE_PATH = './Data/Preprocessed Data'


def load_img(path):
    im = cv2.imread(path)
    im = im[:, :, ::-1]

    return im


def resize_img(img, size, resample=3):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def get_ab(img):
    return color.rgb2lab(img)[:, :, 1:3]


def get_data(path, size=64):
    img = load_img(path)
    img = resize_img(img, 256)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).reshape(256, 256, 1)
    img = resize_img(img, 64)
    img_ab = get_ab(img)
    return img_gray, img_ab


def __loop_filetree(image_path, file_list, size):
    """
    Helper to loop over filetree
    """
    gray_images = np.zeros((len(file_list), 256, 256, 1))
    labels_ab = np.zeros((len(file_list), size, size, 2))

    for i, file_name in enumerate(file_list):
        print(i)
        file_name = file_name[0][0]
        full_filename = os.path.join(image_path, file_name)
        im_gray, im_color_ab = get_data(full_filename, size)

        gray_images[i] = im_gray
        labels_ab[i] = im_color_ab

    np.savez('train_full.npz', gray_images, labels_ab)
    print('done')

    return gray_images, labels_ab


def convert_images(image_path, label_path, size=64):
    """
    image_path: string - path to image files
    label_path: path to .mat needed to label every image

    summary: Creates list of tuples of images and labels
    """

    mat_file = scipy.io.loadmat(label_path)
    gray, ab = __loop_filetree(image_path, mat_file['file_list'], size)

    data = {'gray_images': gray, 'ab_images': ab}

    return data


def save_data(images, save_path):
    """
    images: list of numpy arrays
    save_path: exact saving location

    summary: saves numpy_images to save_path as numpy array
    """

    with open(save_path, 'wb') as f:
        pickle.dump(images, f)


def load_data(save_path):
    """
    save_path: exact location of numpy data

    summary: loads numpy file from save_path
    """

    with open(save_path, 'rb') as f:
        images = pickle.load(f)

    return images


if __name__ == "__main__":
    train_data = convert_images(os.path.join(DATASET_PATH, 'Images'), os.path.join(
        DATASET_PATH, 'lists/train_list.mat'))

    save_data(train_data, os.path.join(PICKLE_PATH, 'train_data.pkl'))

    train_data = load_data(os.path.join(PICKLE_PATH, 'train_data.pkl'))

    print(train_data['gray_images'].shape)
    print(train_data['ab_images'].shape)

    test_data = convert_images(os.path.join(DATASET_PATH, 'Images'), os.path.join(
        DATASET_PATH, 'lists/test_list.mat'))

    save_data(test_data, os.path.join(PICKLE_PATH, 'test_data.pkl'))

    test_data = load_data(os.path.join(PICKLE_PATH, 'test_data.pkl'))

    print(test_data['gray_images'].shape)
    print(test_data['ab_images'].shape)
