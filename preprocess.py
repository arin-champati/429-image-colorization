import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import cv2
import scipy.io
from skimage import color
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from skimage.transform import resize

from utils import *

DATASET_PATH = './Data/Stanford Dog Dataset'
PICKLE_PATH = './Data/Preprocessed Data'

def __loop_filetree(image_path, file_list, size):
    """
    Helper to loop over filetree
    """
    gray_images = np.zeros((len(file_list), size, size))
    labels_ab = np.zeros((len(file_list), size, size, 2))

    for i, file_name in enumerate(file_list):
        print(i)
        file_name = file_name[0][0]
        full_filename = os.path.join(image_path, file_name)
        im_gray, im_color_ab = get_data(full_filename, size)

        gray_images[i] = im_gray
        labels_ab[i] = im_color_ab
    
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
    train_data = convert_images(os.path.join(DATASET_PATH, 'Images'), os.path.join(DATASET_PATH,'lists/train_list.mat'))

    save_data(train_data, os.path.join(PICKLE_PATH, 'train_data.pkl'))

    train_data = load_data(os.path.join(PICKLE_PATH, 'train_data.pkl'))

    print(train_data['gray_images'].shape)
    print(train_data['ab_images'].shape)

    test_data = convert_images(os.path.join(DATASET_PATH, 'Images'), os.path.join(DATASET_PATH,'lists/test_list.mat'))

    save_data(test_data, os.path.join(PICKLE_PATH, 'test_data.pkl'))

    test_data = load_data(os.path.join(PICKLE_PATH, 'test_data.pkl'))

    print(test_data['gray_images'].shape)
    print(test_data['ab_images'].shape)