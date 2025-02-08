import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
img=cv2.imread("panda.jpg")
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (500, 500))
print("original image")
cv2_imshow(img)
histogram=cv2.calcHist([img],[0],None,[256],[0,255])
plt.plot(histogram)
plt.show()
height,width=img.shape[:2]
def evaluatePixelStatic(pixel):
  return np.log10(pixel+1)
def evaluatePixelDynamic(pixel):
  return (255/np.log10(1+255))*np.log10(pixel+1)
for i in range(height):
  for j in range(width):
    img[i][j]=evaluatePixelDynamic(img[i][j])

img = cv2.resize(img, (500, 500))
print("After log transformation")
cv2_imshow(img)
histogram=cv2.calcHist([img],[0],None,[256],[0,255])
plt.plot(histogram)
plt.show()
cv2.waitKey(0)
