import numpy as np
import cv2

model = cv2.data.haarcascades

def face_detection(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_haar_cascade = cv2.CascadeClassifier(model + 'haarcascade_frontalface_default.xml')
    faces = face_haar_cascade.detectMultiScale(gray_img, 1.5, 5)
    return faces, gray_img