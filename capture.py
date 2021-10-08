import picamera
import PIL.Image as Image
import sys

camera = picamera.PiCamera()
imgpath = "rasp_images/image.jpg"
while True:
    x = input("Hit enter: ")
    camera.capture(imgpath)
