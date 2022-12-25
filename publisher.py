import cv2
import zmq
import numpy as np
import pickle
BLOCK_SIZE = 32
port = "5555"

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:%s" % port) # binds to anything that wants to connect

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

# labels = {0: "Owner", 1: "Owner", 2:"Owner"}
with open("label.pkl", "rb") as f:
    labels = pickle.load(f)
    labels = {v:k for k,v in labels.items()}

for i, j in labels.items():
    print(str(i) + " _ " + j)

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
            face = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6)
            for x,y,w,h in face:
                # print(x,y,w,h)
                roi_gray = gray[y:y+h, x:x+w]

                id_, conf = recognizer.predict(roi_gray)
                # print(conf)
                # print(labels[id_])
                if conf >= 70:
                    name  = labels[id_]
                else:
                    name = "thief"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 4)
                img = cv2.putText(frame, str(conf) + " - " + name, (x, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            encoded, buffer = cv2.imencode('.jpg', frame)
            jpg_as_text = np.array(buffer).tobytes()
            socket.send(jpg_as_text)

        except KeyboardInterrupt:
            camera.release()
            cv2.destroyAllWindows()
            break

video_streaming()