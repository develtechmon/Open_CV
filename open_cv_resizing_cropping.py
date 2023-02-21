import cv2
import numpy as np

print("Package Imported")
img = cv2.imread("Resources\car.jpg")
print(img.shape)

img_resize = cv2.resize(img,(640,480)) ##---> Width Height
print(img_resize.shape)

img_cropped = img_resize[0:200, 200:500] ##--->x1,x2 y1,y2

cv2.imshow("Resize", img_resize)
cv2.imshow("Cropped", img_cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()