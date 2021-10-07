from neural_network import *
import cv2
print("Imported cv2")
import picamera
print("Imported picamera")
import PIL.Image as Image
print("Imported PIL.Image")
import RPi.GPIO as GPIO
print("Imported RPi.GPIO")
import sys
print("Imported sys")
import time
print("Imported time")

import math

# DIGIT map as array of array ,
#so that arrSeg[0] shows 0, arrSeg[1] shows 1, etc
## Constants
delay = .1 # delay between digits refresh
display_list = [24,25,8,7,1,12,16] # define GPIO ports to use
arrSeg = [[0,0,0,0,0,0,1],\
          [1,0,0,1,1,1,1],\
          [0,0,1,0,0,1,0],\
          [0,0,0,0,1,1,0],\
          [1,0,0,1,1,0,0],\
          [0,1,0,0,1,0,0],\
          [0,1,0,0,0,0,0],\
          [0,0,0,1,1,1,1],\
          [0,0,0,0,0,0,0],\
          [0,0,0,0,1,0,0]]

selDigit = [14,15,18,23]
digitDP = 20

def switch_func(value, x):
    return {
        0: lambda val: "0.0.0.0",
        1: lambda val: "1.1.1.1",
        2: lambda val: "2.2.2.2",
        3: lambda val: "3.3.3.3",
        4: lambda val: "4.4.4.4",
        5: lambda val: "5.5.5.5",
        6: lambda val: "6.6.6.6",
        7: lambda val: "7.7.7.7",
        8: lambda val: "8.8.8.8",
        9: lambda val: "9.9.9.9"
    }.get(value)(x)
 
# --------------------------------------------------------------------
# PINS MAPPING AND SETUP
# selDigit activates the 4 digits to be showed (0 is active, 1 is unactive)
# display_list maps segments to be activated to display a specific number inside the digit
# digitDP activates Dot led
# --------------------------------------------------------------------

# --------------------------------------------------------------------
# MAIN FUNCTIONS
# splitToDisplay(string) split a string containing numbers and dots in
#   an array to be showed
# showDisplay(array) activates DIGITS according to array. An array
#   element to space means digit deactivation
# --------------------------------------------------------------------

def greyscale(img):
    return img.convert("L")

def shrink_image(img):
    return img.resize((28, 28), Image.BICUBIC)

def showDisplay(digit):
 for i in range(0, 4): #loop on 4 digits selectors (from 0 to 3 included)
  sel = [0,0,0,1]
  sel[i] = 0
  GPIO.output(selDigit, sel) # activates selected digit
  if digit[i].replace(".", "") == " ": # space disables digit
   GPIO.output(display_list,0)
   continue
  numDisplay = int(digit[i].replace(".", ""))
  GPIO.output(display_list, arrSeg[numDisplay]) # segments are activated according to digit mapping
  if digit[i].count(".") == 1:
   GPIO.output(digitDP,1)
  else:
   GPIO.output(digitDP,0)
  time.sleep(delay)

def splitToDisplay (toDisplay): # splits string to digits to display
 arrToDisplay=list(toDisplay)
 for i in range(len(arrToDisplay)):
  if arrToDisplay[i] == ".": arrToDisplay[(i-1)] = arrToDisplay[(i-1)] + arrToDisplay[i] # dots are concatenated to previous array element
 while "." in arrToDisplay: arrToDisplay.remove(".") # array items containing dot char alone are removed
 return arrToDisplay

def setup_GPIO():
    # Use BCM GPIO references instead of physical pin numbers
    GPIO.setmode(GPIO.BCM)
    #disp.List ref: A ,B ,C,D,E,F ,G
    GPIO.setwarnings(False)
    for pin in display_list:
      GPIO.setup(pin,GPIO.OUT) # setting pins for segments
    for pin in selDigit:
      GPIO.setup(pin,GPIO.OUT) # setting pins for digit selector
    GPIO.setup(digitDP,GPIO.OUT) # setting dot pin
    GPIO.setwarnings(True)

## Takes PIL.Image object
def increase_contrast(img):
    return img


### In main loop take pictures every few seconds. Prepare image then send to neural network.
### Retrieve index of highest element then set equal to index
def main():
    if len(sys.argv) != 2:
        print("Need to pass wb_filename")
        sys.exit(1)

    wb_filename = sys.argv[1]
    camera = picamera.PiCamera()

    # Set all pins as output
    setup_GPIO()

    GPIO.output(digitDP,0) # DOT pin


    nn = NeuralNetwork(wb_filename = wb_filename)
    imgdir = "rasp_images/"
    imgpath = imgdir + "image.jpg"
    while True:
        camera.capture(imgpath)
        with Image.open(imgpath) as img:
            high_contrast_img = increase_contrast(img)
            grey_img = greyscale(high_contrast_img)
            little_img = shrink_image(grey_img)
            img_array = (np.array(little_img) / 255).flatten()
            prediction = nn.feed(img_array)
            index = prediction.argmax()
            print(prediction)
            x = index
            toDisplay = switch_func(index,x)
            showDisplay(splitToDisplay(toDisplay))
            time.sleep(0.01)
    
    
    ## Just for debugging
    if len(sys.argv) == 2:
        index = int(sys.argv[1])
    else:
        index = 0


if __name__ == "__main__":
    main()
