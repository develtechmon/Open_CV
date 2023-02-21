import cv2
import numpy as np

img = cv2.imread("Resources/card.jpg")
width,height = 250,350
pts1 = np.float32([[235,307],[394,307],[237,578],[404,568]]) ##----> Start from top left x1, y1 and top right x2,y2
pts2 = np.float32([[0,0], [width,0], [0, height],[width,height]])
matrix = cv2.getPerspectiveTransform(pts1,pts2)

img_output = cv2.warpPerspective(img, matrix, (width,height))

cv2.imshow("Image",img_output)
cv2.waitKey(0)