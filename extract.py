import numpy as np
import os
import PIL.Image as Image

## This way the input, validation, and testing data are shuffled the same way 
# every time.
np.random.seed(1)

## Get the one-hot label vectors
def get_label_vector(directory):
    digit = int(directory)
    label = np.array([0 for i in range(10)])
    label[digit] = 1
    return label

## Load all images
def load_from_images():
    dirpath = "data/greyscale/"
    image_vectors = []
    label_vectors = []
    for directory in os.listdir(dirpath):
        imgfiles = os.listdir(dirpath + directory)
        imgdir = dirpath + directory + "/"
        for imgfile in imgfiles:
            with Image.open(imgdir + imgfile) as img:
                image_vectors.append(np.array(img.convert("L")).flatten() / 255)
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
    tr_end_seg = int(tr_frac * len(inputs))
    val_end_seg = int((tr_end_seg + len(inputs)) / 2)
    tr_inputs = inputs[0 : tr_end_seg]
    tr_outputs = outputs[0 : tr_end_seg]
    val_inputs = inputs[tr_end_seg : val_end_seg]
    val_outputs = outputs[tr_end_seg : val_end_seg]
    te_inputs = inputs[val_end_seg : len(inputs)]
    te_outputs = outputs[val_end_seg : len(outputs)]
    return tr_inputs, tr_outputs, val_inputs, val_outputs, te_inputs, te_outputs
