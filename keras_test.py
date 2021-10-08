import numpy as np
import pandas as pd
from keras.datasets import mnist

## Constants
BATCH_SIZE = 64
EPOCHS = 5

def add_noise(img):
    VARIABILITY = .8
    deviation = VARIABILITY * np.random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 1.)
    return img

def process(img):
    return 1 - add_noise(img)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt

plt.imshow(X_train[0])
plt.title(" Digit " + str(y_train[0]))
plt.xticks([])
plt.yticks([])

X_train[0][18]

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_train = X_train.astype("float32")
X_train /= 255

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_test = X_test.astype("float32")
X_test /= 255
X_train[0][18]


from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(y_train[0])

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPool2D
from keras.layers import Conv2D, BatchNormalization

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape = (28, 28, 1)))
BatchNormalization(axis = -1)
classifier.add(Activation("relu"))
classifier.add(MaxPool2D(pool_size = (2, 2)))

BatchNormalization(axis = -1)

classifier.add(Conv2D(64, (3, 3)))
BatchNormalization(axis = -1)
classifier.add(Activation("relu"))

classifier.add(Conv2D(64, (3, 3)))
classifier.add(Activation("relu"))
classifier.add(MaxPool2D(pool_size = (2, 2)))
classifier.add(Flatten())
BatchNormalization()

classifier.add(Dense(512))
BatchNormalization()

classifier.add(Dense(512))
BatchNormalization()
classifier.add(Activation('relu'))

classifier.add(Dropout(0.2))
classifier.add(Dense(10))
classifier.add(Activation('softmax'))

classifier.summary()

classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

# applying transformation to image

train_gen = ImageDataGenerator(rotation_range = 8, 
                               width_shift_range = 0.08,
                               shear_range = 0.3,
                               height_shift_range = 0.08,
                               zoom_range = 0.2,
                               preprocessing_function = process)
test_gen = ImageDataGenerator()

training_set = train_gen.flow(X_train, y_train, batch_size = BATCH_SIZE)
test_set = train_gen.flow(X_test, y_test, batch_size = BATCH_SIZE)

classifier.fit_generator(training_set,
                         steps_per_epoch = 60000 // BATCH_SIZE,
                         validation_data = test_set,
                         validation_steps = 10000 // BATCH_SIZE,
                         epochs = EPOCHS)
