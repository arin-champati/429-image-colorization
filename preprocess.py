# Arin's data file path
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

import cielab

DATASET_PATH = './Data/Stanford Dog Dataset'
PICKLE_PATH = './Data/Preprocessed Data'

def __convert_images_helper(full_filename, size):
    """
    Gets lab, colored, and grayscale images from one filename
    """
    im_color = cv2.imread(full_filename)

    # reverse im_color from BGR to RGB
    im_color = im_color[:,:,::-1]
    im_gray = cv2.imread(full_filename, 0)

    im_color = cv2.resize(im_color, (size, size), interpolation=cv2.INTER_AREA)
    im_gray = cv2.resize(im_gray, (size, size), interpolation=cv2.INTER_AREA)

    im_color_ab, im_quantized = cielab.preprocess(full_filename, size)

    # add extra dimension to front for num images
    im_color = im_color.reshape((1, size, size, 3))
    im_gray = im_gray.reshape((1, size, size, 1))

    im_quantized = im_quantized.reshape((1, size, size, 313))
    im_color_ab = im_color_ab.reshape((1, size, size, 2))

    return im_color, im_gray, im_color_ab, im_quantized

def __loop_filetree(image_path, file_list, size):
    """
    Helper to loop over filetree
    """
    color_images = np.zeros((len(file_list), size, size, 3))
    gray_images = np.zeros((len(file_list), size, size, 1))
    labels_ab = np.zeros((len(file_list), size, size, 2))
    labels_quantized_ab = np.zeros((len(file_list), size, size, 313))

    for i, file_name in enumerate(file_list):
        file_name = file_name[0][0]
        full_filename = os.path.join(image_path, file_name)
        im_color, im_gray, im_color_ab, im_quantized = __convert_images_helper(full_filename, size)

        color_images[i,:,:,:] = im_color
        gray_images[i,:,:,:] = im_gray
        labels_ab[i,:,:,:] = im_color_ab
        labels_quantized_ab[i,:,:,:] = im_quantized 
    
    print('finished')
    
    return color_images, gray_images, labels_ab, labels_quantized_ab


def convert_images(image_path, label_path, size=64):
    """
    image_path: string - path to image files
    label_path: path to .mat needed to label every image

    summary: Creates list of tuples of images and labels
    """

    mat_file = scipy.io.loadmat(label_path)      
    color, gray, ab, quantized = __loop_filetree(image_path, mat_file['file_list'], size)

    data = {'rgb_images': color, 'gray_images': gray, 'ab_images': ab, 'quantized_images': quantized}

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

    # train_data = load_data(os.path.join(PICKLE_PATH, 'train_data.pkl'))

    # print(train_data['rgb_images'].shape)
    # print(train_data['gray_images'].shape)
    # print(train_data['ab_images'].shape)
    # print(train_data['quantized_images'].shape)

    # test_data = convert_images(os.path.join(DATASET_PATH, 'Images'), os.path.join(DATASET_PATH,'lists/test_list.mat'))

    # save_data(test_data, os.path.join(PICKLE_PATH, 'test_data.pkl'))

    # test_data = load_data(os.path.join(PICKLE_PATH, 'test_data.pkl'))

    # print(test_data['rgb_images'].shape)
    # print(test_data['gray_images'].shape)
    # print(test_data['ab_images'].shape)
    # print(test_data['quantized_images'].shape)