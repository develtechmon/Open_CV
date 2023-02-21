import cv2
import numpy as np

print("Package imported")

cap = cv2.VideoCapture(0) ##---------> Webcam object with Id 0
cap.set(3,640) ##---> Video width with Id 3
cap.set(4,480) ##---> Video height with Id 4
cap.set(10,100) ##---> Video brightness with Id 10

while True:
    success, img = cap.read()
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
