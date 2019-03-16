import numpy as np
import cv2 as cv
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv.imread('a.jpg')
img2 = cv.imread('b.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGRA2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
cv.imshow('img',img)

faces2 = face_cascade.detectMultiScale(gray2, 1.3, 5)
for (x,y,w,h) in faces2:
    cv.rectangle(img2,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray2 = gray2[y:y+h, x:x+w]
    roi_color2 = img[y:y+h, x:x+w]
cv.imshow('img2',img2)

cv.waitKey(0)
cv.destroyAllWindows()