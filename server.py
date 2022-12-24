from flask import Flask, request
import cv2 as cv
import os
import pickle

app = Flask(__name__)





face_cascade = cv.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
labels = {}

with open("label.pkl", "rb") as f:
    labels = pickle.load(f)
    labels = {v:k for k,v in labels.items()}


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if not os.path.isdir("file/"):
        os.makedirs("file")
    file.save(f'file/{file.filename}')
    file_path = f'file/{file.filename}'
    img = cv.imread(file_path)
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for x,y,w,h in face:
        print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray)
        print(conf)
        print(labels[id_])
        if conf >= 0.2:
            name  = labels[id_]
            return {
                "name": name,
                "conf": conf
            }
    os.remove(file_path)
    return {
        "name": "khong nhan ra",
        "conf": 0
    }
