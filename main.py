#!/usr/bin/env python3
#-*- coding: utf-8- -*-

import numpy as np
import cv2
from recognition import face_recognition as fr

model = cv2.data.haarcascades
face_cascade = cv2.CascadeClassifier(model + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

cap = cv2.VideoCapture(0)

img = cv2.imread('images/p1.png')

while True:
    ret, frame = cap.read()
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    faces, gray = fr.face_detection(frame)
    print("faces : {}".format(faces)) 
    for (x, y, w, h) in faces:
        print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        #Utilisez ici un model deep learning avec tensorflow par exemple
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 5)
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()