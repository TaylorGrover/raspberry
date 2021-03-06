import sys
import RPi.GPIO as GPIO
import time

import math

#index will be passed to here, change the 2
#could be simplified more but I tried fiddling 
#with it and this is the one that doesn't get wonky

#index = 9
if len(sys.argv) == 1:
    index = int(sys.argv[1])
else:
    index = 0
x = index
 
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
 
#print(switch_func(index, x))

toDisplay = switch_func(index,x)
'''arrSeg = [[1,1,1,1,1,1,0],\
          [0,1,1,0,0,0,0],\
          [1,1,0,1,1,0,1],\
          [1,1,1,1,0,0,1],\
          [0,1,1,0,0,1,1],\
          [1,0,1,1,0,1,1],\
          [1,0,1,1,1,1,1],\
          [1,1,1,0,0,0,0],\
          [1,1,1,1,1,1,1],\
          [1,1,1,1,0,1,1]]'''

delay = 0.005 # delay between digits refresh

# --------------------------------------------------------------------
# PINS MAPPING AND SETUP
# selDigit activates the 4 digits to be showed (0 is active, 1 is unactive)
# display_list maps segments to be activated to display a specific number inside the digit
# digitDP activates Dot led
# --------------------------------------------------------------------

selDigit = [14,15,18,23]
# Digits:   1, 2, 3, 4

display_list = [24,25,8,7,1,12,16] # define GPIO ports to use
#display_list = [18,22,24,26,28,32,36]
#disp.List ref: A ,B ,C,D,E,F ,G

digitDP = 20
#DOT = GPIO 20

# Use BCM GPIO references instead of physical pin numbers
GPIO.setmode(GPIO.BCM)

# Set all pins as output
GPIO.setwarnings(False)
for pin in display_list:
  GPIO.setup(pin,GPIO.OUT) # setting pins for segments
for pin in selDigit:
  GPIO.setup(pin,GPIO.OUT) # setting pins for digit selector
GPIO.setup(digitDP,GPIO.OUT) # setting dot pin
GPIO.setwarnings(True)

# DIGIT map as array of array ,
#so that arrSeg[0] shows 0, arrSeg[1] shows 1, etc
'''arrSeg = [[1,1,1,1,1,1,0],\
          [0,1,1,0,0,0,0],\
          [1,1,0,1,1,0,1],\
          [1,1,1,1,0,0,1],\
          [0,1,1,0,0,1,1],\
          [1,0,1,1,0,1,1],\
          [1,0,1,1,1,1,1],\
          [1,1,1,0,0,0,0],\
          [1,1,1,1,1,1,1],\
          [1,1,1,1,0,1,1]]'''

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


GPIO.output(digitDP,0) # DOT pin

# --------------------------------------------------------------------
# MAIN FUNCTIONS
# splitToDisplay(string) split a string containing numbers and dots in
#   an array to be showed
# showDisplay(array) activates DIGITS according to array. An array
#   element to space means digit deactivation
# --------------------------------------------------------------------

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

# --------------------------------------------------------------------
# MAIN LOOP
# persistence of vision principle requires that digits are powered
#   on and off at a specific speed. So main loop continuously calls
#   showDisplay function in an infinite loop to let it appear as
#   stable numbers display
# --------------------------------------------------------------------

try:
 while True:
  showDisplay(splitToDisplay(toDisplay))
except KeyboardInterrupt:
 print('interrupted!')
 GPIO.cleanup()
sys.exit()

