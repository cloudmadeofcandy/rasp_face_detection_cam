import os
import cv2 as cv
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
imgdir = os.path.join(BASE_DIR, "dataset")

face_cascade = cv.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_trains = []

count = 0

for root, dirs, files in os.walk(imgdir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            print(path)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "_").lower()
            
            if label in label_ids:
                pass
            else:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            pil_image = Image.open(path).convert("L")
            image_array = np.array(pil_image, np.uint8)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            # if len(faces) == 0:
            #     os.remove(path)
            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_trains.append(roi)
                y_labels.append(id_)
                

with open("label.pkl", "wb") as f:
    pickle.dump(label_ids, f)


recognizer.train(x_trains, np.array(y_labels))
recognizer.save("trainner.yml")