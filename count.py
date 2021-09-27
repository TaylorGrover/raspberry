import os
topleveldir = "data/greyscale/"

for directory in os.listdir(topleveldir):
    print("%s: %d" % (directory, len(os.listdir(topleveldir + directory))))
