import numpy as np
import os
import PIL.Image as Image
import sys

def shrink_image(img):
    return img.resize(28, 28)

def greyscale(img):
    grey = np.array([0.2126, 0.7152, 0.0722])
    return np.array(img).dot(grey)

def main():
    if len(sys.argv) != 2:
        print("Usage: fix_images DIGIT")
        sys.exit(1)

    ## The first command-line argument (a digit between 0 - 9) is the label used for the images
    label = sys.argv[1]

    ## The digits directory is the landing place for raw images. 
    # Images will resized and labeled and greyscaled from this directory, then the smaller
    # copies will be moved to the appropriate directories. 
    landing_dir = "digits/"
    temp_files = os.listdir(landing_dir)
    print(temp_files)
    
    ## If the label isn't a valid directory, exit.
    """if not os.path.isdir(label):
        print("Invalid argument.")
        sys.exit(1)"""

    ## Check the selected "label" directory for existing images files. Obtain last image name
    imgfiles = os.listdir("data/greyscale/" + label)
    if len(imgfiles) > 0:
        numeric_suffix = int(imgfiles[-1].split("_")[1].split(".")[0])
    else:
        numeric_suffix = 0

    ## The numeric suffix + 1 will be the starting position of the new images
    for tmp in temp_files:
        numeric_suffix += 1
        with Image.open(landing_dir + tmp) as raw_img:
            grey_array = greyscale(raw_img)
            resized_img = Image.fromarray(grey_array).resize((28, 28), Image.BICUBIC)
            name = "data/greyscale/" + label + "/" + label + "_" + str(numeric_suffix) + ".png"
            print(name)
            resized_img.convert("RGB").save(name)
        os.remove(landing_dir + tmp)

if __name__ == "__main__":
    main()
