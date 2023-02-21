import cv2
import numpy as np

img = np.zeros((512,512,3),np.uint8) ##---> Size of Matrix, add 3 to add color for channels functionality (Matrix hold data of W, H and Color)
print(img.shape)
print(img)

img[:] = 255,0,0 ##---> Full blue color with ":"

img[200:300, 100:300]  = 255,0,0 ##---> Cropped the blue color x1,x2 y1,y2

cv2.line(img,(0,0), (300,300), (0,255,0),2)
cv2.line(img,(0,0), (img.shape[1], img.shape[0]), (0,255,0),3) ##---> Width and Height, green color and thickness 3 for full line
cv2.rectangle(img,(0,0),(250,350),(0,0,255),2) ##----< Width and Height, red color and thickness is 2
cv2.rectangle(img,(0,0),(250,350),(0,0,255),cv2.FILLED) ##----< Width and Height, red color and thickness. Filled the color
cv2.circle(img,(400,50), 30, (255,255,0),5) ## Width and Height, radius, blue color, thickness 5

cv2.putText(img," SHAPES AND TEXT ", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,150,0),2) ##---> Text, width and height, font, scale, color, thickness
cv2.imshow("Image", img)


cv2.waitKey(0)
cv2.destroyAllWindows()