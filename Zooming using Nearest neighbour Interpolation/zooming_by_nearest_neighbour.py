import math
import numpy as np
from PIL import Image
import cv2
from google.colab.patches import cv2_imshow

def nearestNeighborScaling(source, newWid, newHt):
    # Get the dimensions of the source image
    height, width, channels = source.shape

    # Create an empty NumPy array for the target image
    target = np.zeros((newHt, newWid, channels), dtype=np.uint8) # Create empty image with same channels and dtype

    #width = getWidth(source) # Replaced with NumPy shape attribute
    #height = getHeight(source) # Replaced with NumPy shape attribute

    for x in range(0, newWid):
        for y in range(0, newHt):
            srcX = int(round(float(x) / float(newWid) * float(width)))
            srcY = int(round(float(y) / float(newHt) * float(height)))
            srcX = min(srcX, width - 1)
            srcY = min(srcY, height - 1)
            #tarPix = getPixel(target, x, y) # Replaced with direct array access
            #srcColor = getColor(getPixel(source, srcX, srcY)) # Replaced with direct array access
            #setColor(tarPix, srcColor) # Replaced with direct array assignment
            target[y, x] = source[srcY, srcX] # Assign color values directly

    return target

img = np.array(Image.open('/content/small.jpg'))
print("original image")
cv2_imshow(img)
resized_img = nearestNeighborScaling(img, 800, 800)
print("After nearest neighbor interpolation")
cv2_imshow(resized_img) # Display the resized image
