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
    images = images.astype(np.float32) / 255.0
    images = np.expand_dims(images, axis=-1)
    return images

def load_n_images(n_of_images, random_state=42, test_size=0.25):

    directory = './MNIST/'
    image_directory = directory + 'train-images.idx3-ubyte'
    label_directory = directory + 'train-labels.idx1-ubyte'
    if not os.path.exists(image_directory):
        print('File not found: ' + image_directory)
    if not os.path.exists(label_directory):
        print('File not found: ' + label_directory)
    
    images = load_images(image_directory)
    labels = load_labels(label_directory)

    # Create an array of indices for tracking
    indices = np.arange(len(images))
    sampled_indices = indices[:n_of_images]
    sample_images = images[sampled_indices]
    sample_labels = labels[sampled_indices]

    # use the last 20% of the data for testing, and the first 80% for training
    images, test_images, labels, test_labels, indices, test_indices = train_test_split(sample_images, sample_labels, sampled_indices, test_size=test_size, random_state=random_state)    
    return images, labels, indices, test_images, test_labels, test_indices