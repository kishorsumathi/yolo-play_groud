import cv2
import numpy as np
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
confidence = 0.3
Nms= 0.3
class_names = []
with open("coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# Load the pre-trained YOLOv4-Tiny model
arc = cv2.dnn.readNetFromDarknet('yolov4-tiny.cfg', 'yolov4-tiny.weights')

arc.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
arc.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Open the video stream (0 for the default camera, or provide the path to a video file)
cap = cv2.VideoCapture(0)

while True:
    # Read the next frame from the video stream
    ret, frame = cap.read()
    
    if not ret:
        break

    model = cv2.dnn_DetectionModel(arc)
    model.setInputParams(size=(640,640), scale=1/255, swapRB=True)
    x =time.time()
    classes, scores, boxes = model.detect(frame, confidence, Nms)
    y= time.time()
    fps=1/(y-x)
    for (classid, score, box) in zip(classes, scores, boxes):
        label = "%s : %.2f" % (class_names[classid],score)
        cv2.rectangle(frame,box,color=(255, 200, 10),thickness=2)
        cv2.putText(frame, label, (box[0],box[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(25,55,255),2)
        cv2.putText(frame, "FPS:{0:.2f}".format(fps),(20, 25), cv2.FONT_HERSHEY_PLAIN,fontScale=2,color=(255, 0, 0),thickness=2)
    # Display the output frame
    cv2.imshow('Object Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()
