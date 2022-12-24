import numpy as np
import cv2 as cv
import pickle



face_cascade = cv.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")
cap = cv.VideoCapture(0)

labels = {}

with open("label.pkl", "rb") as f:
    labels = pickle.load(f)
    labels = {v:k for k,v in labels.items()}

def main():

    while True :
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        face = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for x,y,w,h in face:
            # print(x,y,w,h)
            roi_gray = gray[y:y+h, x:x+h]

            id_, conf = recognizer.predict(roi_gray)
            if conf >= 45 and conf <= 85:
                name  = labels[id_]
                # print(name)
                font = cv.FONT_HERSHEY_SIMPLEX
                color = (0, 255, 0)
                stroke = 2
                cv.putText(frame, name, (x, y), font, 1, color, stroke, cv.LINE_AA)
            
            color = (255, 0, 0)
            stroke = 2
            width = x + w
            heigh = y + h

            cv.rectangle(frame, (x,y), (width, heigh), color, stroke)


        cv.imshow('frame', frame)
        if cv.waitKey(20) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
    return


if __name__ == "__main__":
    main()
