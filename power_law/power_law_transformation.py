import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
g=float(input("Enter gamma value: "))
img=cv2.imread('panda.jpg',0)
img = cv2.resize(img, (350, 350))
print("original image")
cv2_imshow(img)
height,width=img.shape[:2]
gamma=1/g
c=255
def evaluatePixel(pixel):
  global gamma,c
  return c*((pixel/255)**gamma)
for i in range(height):
  for j in range(width):
    img[i][j]=evaluatePixel(img[i][j])

img = cv2.resize(img, (350, 350))
print("After power law transformation")
cv2_imshow(img)
histogram=cv2.calcHist([img],[0],None,[256],[0,255])
plt.plot(histogram)
plt.show()
cv2.waitKey(0)
