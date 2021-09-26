### If data are deleted from data/greyscale/, then renumber the images

import os
import sys

def get_largest_suffix(imgfiles):
    suffixes = [int(imgfiles[i].split("_")[1].split(".")[0]) for i in range(len(imgfiles))]
    return max(suffixes)

def main():
    datapath = "data/greyscale/"
    for directory in os.listdir(datapath):
        imgfiles = os.listdir(datapath + directory)
        n = len(imgfiles)
        if n == 0:
            continue
        max_suffix = get_largest_suffix(imgfiles)
        if n != max_suffix:
            print("n = %d; max_suffix = %d; dir = %s" % (n, max_suffix, directory))
            suffix = 1
            current_path = datapath + directory + "/"
            for imgfile in imgfiles:
                tmppath = current_path + "tmp_" + str(suffix) + ".png"
                os.rename(current_path + imgfile, tmppath)
                suffix += 1
            tmpfiles = os.listdir(current_path)
            for tmpfile in tmpfiles:
                os.rename(current_path + tmpfile, current_path + directory + "_" + tmpfile.split("_")[1])


if __name__ == "__main__":
    main()
