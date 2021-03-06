import glob
import numpy as np
import cv2 as cv
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

txtfiles = [] 
for file in glob.glob("faces/*.jpg"):
    txtfiles.append(file)
for file in glob.glob("faces/*.jpeg"):
    txtfiles.append(file)

print('Total files', len(txtfiles))
faces_found = 0
for file in txtfiles:
    img = cv.imread(file)
    gray = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces_found += len(faces)
    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        cv.imshow('img', img)
        cv.waitKey(0)
print('Total faces', faces_found)
cv.destroyAllWindows()