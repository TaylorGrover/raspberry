from extract import *
from neural_network import *
from scipy.ndimage import rotate
import tkinter as tk

def rot(image_array, deg):
    if not image_array.shape == (28, 28):
        rotated = rotate(image_array.reshape((28, 28)), deg)
    else:
        rotated = rotate(image_array, deg)
    return rotated

def get_image_array(filename):
    with Image.open(filename) as img:
        arr = np.array(img.convert("L")).flatten() / 255
    return arr

def main():
    wb_filename = sys.argv[1]
    nn = NeuralNetwork(wb_filename = wb_filename)
    image = get_image_array(sys.argv[2])
    Image.fromarray(image.reshape((28, 28)) * 255).show()
    print(nn.feed(image))
    time.sleep(5)
    print(nn.feed(rot(image, -90)))

if __name__ == "__main__":
    main()
