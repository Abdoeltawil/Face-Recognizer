import cv2
import numpy as np
import pickle
import os
labels ={"name":1}

with open("labels.pkl",'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
video = cv2.VideoCapture(0)
while True:
    ret, img = video.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
    for (x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = img[y:y+h,x:x+w]
       #Deep Learning
        _id , conf = recognizer.predict(roi_gray)
        if conf >=45 and conf <= 85:
            print(_id)
            print(labels[_id])
            font = cv2.FONT_HERSHEY_COMPLEX
            name = labels[_id]
            color = (255,255,255)
            stroke = 3
            cv2.putText(img, name, (x-10,y-10), font, 1, color, stroke, cv2.LINE_AA)
        img_item="blue.png"
        cv2.imwrite(img_item,roi_gray)
        color = (100,0,250) #BGR
        stroke=2
        end_cord_x = x+w
        end_cord_y = y+h
        cv2.rectangle(img,(x,y),(end_cord_x,end_cord_y),color,stroke)
    # Display
    cv2.imshow("Frame", img)
    k=cv2.waitKey(1)
    if k == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
