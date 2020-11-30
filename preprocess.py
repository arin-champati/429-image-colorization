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

DATASET_PATH = './Data/Stanford Dog Dataset'
PICKLE_PATH = './Data/Preprocessed Data'

def __convert_images_helper(full_filename, size):
    """
    Gets lab, colored, and grayscale images from one filename
    """
    im_color = cv2.imread(full_filename)

    # reverse im_color from BGR to RGB to be passed to rgbtolab
    im_color = im_color[:,:,::-1]
    im_gray = cv2.imread(full_filename, 0)

    im_color = cv2.resize(im_color, (size, size), interpolation=cv2.INTER_AREA)
    im_gray = cv2.resize(im_gray, (size, size), interpolation=cv2.INTER_AREA)

    # convert from RGB to LAB color space
    im_color_lab = color.rgb2lab(im_color)

    # add extra dimension to front for num images
    im_color_lab = im_color_lab.reshape((1, size, size, 3))
    im_color = im_color.reshape((1, size, size, 3))
    im_gray = im_gray.reshape((1, size, size, 1))

    return im_color_lab, im_color, im_gray

def __loop_filetree(image_path, file_list, size):
    """
    Helper to loop over filetree
    """
    images = []
    i = 0
    for file_name in file_list:
        file_name = file_name[0][0]
        full_filename = os.path.join(image_path, file_name)
        im_color_lab, img_color, im_gray = __convert_images_helper(full_filename, size)

        images.append({'rgb': img_color, 'lab': im_color_lab, 'gray': im_gray})

    return images

def convert_images(image_path, train_label_path, test_label_path, size=64):
    """
    image_path: string - path to image files
    label_path: path to .mat needed to label every image

    summary: Creates list of tuples of images and labels
    """

    test_mat_file = scipy.io.loadmat(test_label_path)
    test_images = __loop_filetree(image_path, test_mat_file['file_list'], size)
    
    print('testing done')
    
    train_mat_file = scipy.io.loadmat(train_label_path)      
    train_images = __loop_filetree(image_path, train_mat_file['file_list'], size)

    print('training done')

    return train_images, test_images

def save_images(images, save_path):
    """
    images: list of numpy arrays
    save_path: exact saving location

    summary: saves numpy_images to save_path as numpy array
    """
    
    with open(save_path, 'wb') as f:
        pickle.dump(images, f)

def load_images(save_path):
    """
    save_path: exact location of numpy data

    summary: loads numpy file from save_path
    """
    with open(save_path, 'rb') as f:
        images = pickle.load(f)
    
    return images

if __name__ == "__main__":
    train_images, test_images = convert_images(os.path.join(DATASET_PATH, 'Images'), os.path.join(DATASET_PATH,'lists/train_list.mat'), os.path.join(DATASET_PATH,'lists/test_list.mat'))
    print(len(train_images), len(test_images)) 

    save_images(train_images, os.path.join(PICKLE_PATH, 'train.pkl'))
    save_images(test_images, os.path.join(PICKLE_PATH, 'test.pkl'))