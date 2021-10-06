import cv2
import json
import numpy as np
import os
import PIL.Image as Image
import sys

## This way the input, validation, and testing data are shuffled the same way 
# every time.
np.random.seed(1)

## Get the one-hot label vectors
def get_label_vector(directory):
    digit = int(directory)
    label = np.array([0 for i in range(10)])
    label[digit] = 1
    return label

## Load all images. Assumes 90 clockwise rotation must be performed
def load_from_images():
    dirpath = "data/greyscale/"
    image_vectors = []
    label_vectors = []
    for directory in os.listdir(dirpath):
        imgfiles = os.listdir(dirpath + directory)
        imgdir = dirpath + directory + "/"
        for imgfile in imgfiles:
            with Image.open(imgdir + imgfile) as img:
                greyscale_matrix = np.array(img.convert("L"))
                greyscale_matrix_rotated = cv2.rotate(greyscale_matrix, cv2.ROTATE_90_CLOCKWISE)
                greyscale_vector = greyscale_matrix_rotated.flatten()
                greyscale_vector_norm = greyscale_vector / 255
                image_vectors.append(greyscale_vector_norm)
                label_vectors.append(get_label_vector(directory))
    return image_vectors, label_vectors

## Split images into training, testing and validation
# the training data should be around 80% of the whole data set.
# Need to shuffle the images in the beginning since everything is initially
# ordered by digit. The testing and validation images will be approximately
# split in half.
def get_tvt(tr_frac = 0.8):
    inputs, labels = load_from_images()
    zipped = list(zip(inputs, labels))
    np.random.shuffle(zipped)
    inputs, outputs = zip(*zipped)
    inputs = list(inputs)
    outputs = list(outputs)
    tr_end_seg = int(tr_frac * len(inputs))
    val_end_seg = int((tr_end_seg + len(inputs)) / 2)
    tr_inputs = inputs[0 : tr_end_seg]
    tr_outputs = outputs[0 : tr_end_seg]
    val_inputs = inputs[tr_end_seg : val_end_seg]
    val_outputs = outputs[tr_end_seg : val_end_seg]
    te_inputs = inputs[val_end_seg : len(inputs)]
    te_outputs = outputs[val_end_seg : len(outputs)]
    return tr_inputs, tr_outputs, val_inputs, val_outputs, te_inputs, te_outputs

## Combine the makeshift and MNIST datasets (Experimental)
def get_combination(mnist_test = False):
    tr_inputs, tr_outputs, val_inputs, val_outputs, te_inputs, te_outputs = get_tvt()
    print(type(tr_inputs))
    tr_mnist_inputs, tr_mnist_outputs, val_mnist_inputs, val_mnist_outputs = get_training_and_validation()
    tr_inputs.extend(tr_mnist_inputs)
    tr_outputs.extend(tr_mnist_outputs)
    val_inputs.extend(val_mnist_inputs)
    val_outputs.extend(val_mnist_outputs)
    if mnist_test:
        te_mnist_inputs, te_mnist_outputs = get_testing_images()
        te_inputs.extend(te_mnist_inputs)
        te_outputs.extend(te_mnist_outputs)
    return tr_inputs, tr_outputs, val_inputs, val_outputs, te_inputs, te_outputs

"""
Functions for retrieving the images and labels from the MNIST binary files dataset.
"""

# Image training data files
directory = "data/"
training_image_filename = "train-images-idx3-ubyte"
training_labels_filename = "train-labels-idx1-ubyte"
testing_image_filename = "t10k-images-idx3-ubyte"
testing_labels_filename = "t10k-labels-idx1-ubyte"

# Big Endian byte storage conversion from bytecode to decimal integer
### Obsolete since int.from_bytes(bytes, byteorder = "big", signed = False)
  # does the same thing
def bytes_to_int(byte_data):
    '''byte_count = len(byte_data)
    total = 0
    for i in range(byte_count):
        total += 16 ** (2 * (byte_count - i - 1)) * byte_data[i]'''
    return int.from_bytes(byte_data, byteorder = "big", signed = False)


# This takes the bytes of the ubyte image file and first finds the total number
# of images to extract.
def get_images_array(image_filename, inverted = True):
    image_bytes = read_bytes(image_filename)
    images = []
    num_images = bytes_to_int(image_bytes[4 : 8])
    rows = bytes_to_int(image_bytes[8 : 12])
    cols = bytes_to_int(image_bytes[12 : 16])
    for i in range(num_images):
        array = np.array(bytearray(image_bytes[16 + i * rows * cols : 16 + rows * cols * (i + 1)])) / 255
        if inverted:
            images.append(1 - array)
        else:
            images.append(array)
    return images

"""
Determine how to split the training and validation images
"""
def get_training_and_validation(training_count = 50000, inverted = True):
    if 0 >= training_count or training_count >= 60000:
        raise ValueError("Training image count must be between 0 and 60000.")
    images = get_images_array(directory + training_image_filename, inverted)
    labels = get_labels(directory + training_labels_filename)
    training_images = images[0 : training_count]
    training_labels = labels[0 : training_count]
    validation_images = images[training_count : 60000]
    validation_labels = labels[training_count : 60000]
    return training_images, training_labels, validation_images, validation_labels

def get_testing_images(inverted = True):
    return get_images_array(directory + testing_image_filename, inverted), get_labels(directory + testing_labels_filename)

# Get the labels corresponding to the images
def get_labels(label_filename):
    label_bytes = read_bytes(label_filename)
    num_labels = bytes_to_int(label_bytes[4 : 8])
    labels = [[0 for i in range(10)] for j in range(num_labels)]
    for i in range(num_labels):
        labels[i][label_bytes[8 + i]] = 1
    return labels

def read_bytes(filename):
    with open(filename, "rb") as f:
        file_bytes = f.read()
        return file_bytes

# Save the image pixel arrays to a json-formatted file and the labels
def save_to_json(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f)

# Read a json file
def load_from_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
        return data

# Convert byte array to png image array
'''def to_png_array(pixels):
    header = b'\x89\x50\x4e\x47\x0d\x0a\x1a\x0a'
    ihdr = b'\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\x1c\x00\x00\x00\x1c'
    idat = bytes(len(pixels))
    print(int.from_bytes(idat, byteorder = "big", signed = False))
    iend = b'\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82'
    return header '''

def export_png(filename, data):
    with open(filename, "wb") as f:
        f.write(data)

""" Take an arbitrary png or jpg file and scale it to 28x28 using any means necessary
"""
def scale_image(imagefile):
    with Image.open(imagefile) as img:
        rescaled = img.resize((28, 28))
        return rescaled

def invert(image_array):
    for i in range(len(image_array)):
        image_array[i] = 1 - image_array[i]
    return image_array

## Invert the images (1 - image)
def get_noisy_mnist():
    ti, tl, vi, vl = get_training_and_validation()
    tei, tel = get_testing_images()
    return ti, tl, vi, vl, tei, tel
