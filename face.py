import cv2
import os
import numpy as np
import pickle
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"pic")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
current_id = 0
labels_id = {}
y_label = []
x_train = []
for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file)
            label = os.path.basename(root).replace(" ","-").lower()
            #print(label,path)
            if not label in labels_id:
                labels_id[label] = current_id
                current_id +=1
            id_ = labels_id[label]
           # print(labels_id)
            #y_label.append(label)
            #x_train.append(path)
            pil_image = Image.open(path).convert("L") #to be grayscale
            size = (550,550)
            final_image = pil_image.resize(size,Image.ANTIALIAS)
            image_arry = np.array(final_image,"uint8")
           # print(image_arry)
            faces = face_cascade.detectMultiScale(image_arry, scaleFactor=1.5, minNeighbors=5)
            for (x,y,w,h) in faces:
                roi = image_arry[y:y+h,x:x+w]
                x_train.append(roi)
                y_label.append(id_)
#print(y_label)
#print(x_train)

with open("labels.pkl",'wb') as f:
    pickle.dump(labels_id,f)
recognizer.train(x_train,np.array(y_label))
recognizer.save("trainner.yml")