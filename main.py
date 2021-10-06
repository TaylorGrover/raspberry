from functions import*
import numpy as np
import sys
np.set_printoptions(linewidth = 200)
from extract import *
from neural_network import NeuralNetwork

### Invert Images to have white background and black foreground
inverted = True

### Hyperparameters
epochs = 20
batch_size = 20
clip_threshold = .01
eta = .3
decay = .01
architecture = [784, 50, 10]
activations = ["sigmoid", "softargmax"]

np.random.seed(1)
nn = NeuralNetwork(architecture, activations)
nn.cost = cross_entropy
nn.cost_deriv = cross_entropy_deriv
nn.test_validation = True
nn.use_clipping = False
nn.use_dropout = True 
nn.use_L2 = True
nn.show_gradient = False
nn.save_wb = True
nn.use_diff_eq = True

#training_images, training_labels, validation_images, validation_labels, testing_images, testing_labels = get_noisy_mnist()
training_images, training_labels, validation_images, validation_labels = get_training_and_validation(inverted = inverted)
testing_images, testing_labels = get_testing_images(inverted = inverted)
if len(sys.argv) == 3:
   wb_filename = sys.argv[1]
   nn.set_wb(wb_filename)
   if sys.argv[2] == "test":
      print(nn.test([testing_images, testing_labels], iterations = "all"))

# Training the network with starting weights
if len(sys.argv) == 1 or sys.argv[2] == "train":
   nn.train([training_images, training_labels, validation_images, validation_labels, testing_images, testing_labels], \
       batch_size = batch_size, clip_threshold = clip_threshold, decay = decay, epochs = epochs, eta = eta)

