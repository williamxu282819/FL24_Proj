import numpy as np
import struct

import matplotlib.pyplot as plt
# import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

import os

def load_images(filename):
    with open(filename, 'rb') as f:
        # Read the magic number, number of images, rows, and columns
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        # Read the image data and reshape it into [number of images, rows, cols]
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)
    return images

def load_labels(filename):
    with open(filename, 'rb') as f:
        # Read the magic number and number of labels
        magic, num_labels = struct.unpack(">II", f.read(8))
        # Read the label data
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

def preprocess_images(images):
    # Normalize pixel values to [0, 1] range
    images = images.astype(np.float32) / 255.0
    # Expand dimensions to [number of images, rows, cols, channels]
    images = np.expand_dims(images, axis=-1)
    return images

def load_n_images(n_of_images, random_state=42, test_size=0.25):

    directory = './MNIST/'
    train_image_directory = directory + 'train-images.idx3-ubyte'
    train_label_directory = directory + 'train-labels.idx1-ubyte'
    # test_image_directory = directory + 't10k-images.idx3-ubyte'
    # test_label_directory = directory + 't10k-labels.idx1-ubyte'
    # does the file exist?
    if not os.path.exists(train_image_directory):
        print('File not found: ' + train_image_directory)
    if not os.path.exists(train_label_directory):
        print('File not found: ' + train_label_directory)

    # if not os.path.exists(test_image_directory):
    #     print('File not found: ' + test_image_directory)

    # if not os.path.exists(test_label_directory):
    #     print('File not found: ' + test_label_directory)

    images = load_images(train_image_directory)
    labels = load_labels(train_label_directory)
    # test_images = load_images(test_image_directory)
    # test_labels = load_labels(test_label_directory)
    # train_images = preprocess_images(train_images)
    # test_images = preprocess_images(test_images)

    # randomly sample 1200 images from my dataset
    sample_images = images[:n_of_images]
    sample_labels = labels[:n_of_images]

    # use the last 20% of the data for testing, and the first 80% for training
    train_images, test_images, train_labels, test_labels = train_test_split(sample_images, sample_labels, test_size=test_size, random_state=random_state)    
    # for train_images, train_labels, split the data into 75% training and 25% validation
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.25, random_state=random_state)
    return train_images, train_labels, test_images, test_labels, val_images, val_labels