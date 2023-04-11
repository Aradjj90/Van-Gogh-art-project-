import cv2
import mediapipe as mp
import time
import numpy as np
import random

mp_face_detection = mp.solutions.face_detection

models = ["starry_night.t7", "the_scream.t7", "the_wave.t7", "udnie.t7", "candy.t7",
          "feathers.t7","la_muse.t7", "mosaic.t7"]
flag_foto = False
show = False
foto = []

def molbert (scrin, model):
    net = cv2.dnn.readNetFromTorch(model)
    inWidth = scrin.shape[1]
    inHeight = scrin.shape[0]
    inp = cv2.dnn.blobFromImage(scrin, 1.0, (inWidth, inHeight),
                               (103.939, 116.779, 123.68), swapRB=False, crop=False)

    net.setInput(inp)
    out = net.forward()

    out = out.reshape(3, out.shape[2], out.shape[3])
    out[0] += 103.939
    out[1] += 116.779
    out[2] += 123.68
    out /= 255
    out = out.transpose(1, 2, 0)
    out = cv2.medianBlur(out, 1)  
    return out

def detection(frame):
    results = face_detection.process(frame)
    return results.detections


# loop

cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.8) as face_detection:
    while cap.isOpened():
        ret, image = cap.read()

        if flag_foto and show:
            flag_foto = False
            show = False
            print("show 30 sec")
            time.sleep(10)
            ret, image = cap.read()
            cv2.destroyAllWindows()

        if detection(image) and not flag_foto:
            print("I see you in 5 seconds I'll take a picture")
            time.sleep(5)
            ret, image = cap.read()
            if detection(image):
              flag_foto = True
              continue
            else:
              continue

        if  flag_foto and not show:
            foto = molbert(image,random.choice(models))
            cv2.imshow('Styled image', foto)
            show = True

        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
