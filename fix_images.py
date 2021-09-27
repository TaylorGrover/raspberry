import numpy as np
import os
import PIL.Image as Image
import sys
import time

def shrink_image(img):
    return img.resize(28, 28)

def get_largest_suffix(imgfiles):
    suffixes = [int(imgfiles[i].split("_")[1].split(".")[0]) for i in range(len(imgfiles))]
    return max(suffixes)

def greyscale(img):
    return np.array(img.convert("L"))

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
    temp_files = os.listdir(landing_dir + label)
    print("%d images. Use how many?" % (len(temp_files)), end = " ")
    counter = int(input())
    
    ## If the label isn't a valid directory, exit.
    if not os.path.isdir(landing_dir + label):
        print("Invalid digit.")
        sys.exit(1)

    ## Check the selected "label" directory for existing images files. Obtain largest filename suffix
    imgfiles = os.listdir("data/greyscale/" + label)
    if len(imgfiles) > 0:
        numeric_suffix = get_largest_suffix(imgfiles)
    else:
        numeric_suffix = 0

    ## The numeric suffix + 1 will be the starting position of the new images
    for tmp in temp_files:
        if counter == 0:
            sys.exit(0)
        numeric_suffix += 1
        with Image.open(landing_dir + label + "/" + tmp) as raw_img:
            grey_array = greyscale(raw_img)
            resized_img = Image.fromarray(grey_array).resize((28, 28), Image.BICUBIC)
            name = "data/greyscale/" + label + "/" + label + "_" + str(numeric_suffix) + ".png"
            print(name)
            resized_img.convert("RGB").save(name)
        os.remove(landing_dir + label + "/" + tmp)
        # Prevent program crashing on mobile device
        time.sleep(.01)
        counter -= 1

if __name__ == "__main__":
    main()
