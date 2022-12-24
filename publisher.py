import base64
import threading
import cv2
from sklearn.neighbors import KernelDensity
import zmq
import time
import numpy as np
from datetime import datetime
from Crypto.Util.Padding import pad, unpad
from Crypto.Cipher import AES
BLOCK_SIZE = 32
port = "5555"

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:%s" % port) # binds to anything that wants to connect

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
labels = {2:"Gia Luong", 1:"Thief", 0:"Nothing"}

camera = cv2.VideoCapture(0)  # init the camera

def video_streaming():
    global socket
    global cipher
    global obj
    while True:
        try:
            grabbed, frame = camera.read()  # grab the current frame
            frame = cv2.resize(frame, (640, 480))  # resize the frame
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            face = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
            for x,y,w,h in face:
                print(x,y,w,h)
                roi_gray = gray[y:y+h, x:x+w]

                id_, conf = recognizer.predict(roi_gray)
                print(conf)
                print(labels[id_])
                if conf >= 0.2:
                    name  = labels[id_]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 4)
                    img = cv2.putText(frame, labels[id_], (x, y - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            encoded, buffer = cv2.imencode('.jpg', frame)
            jpg_as_text = np.array(buffer).tobytes()
            socket.send(jpg_as_text)

        except KeyboardInterrupt:
            camera.release()
            cv2.destroyAllWindows()
            break

video_streaming()