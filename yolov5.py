import torch
import cv2
import time

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
cap=cv2.VideoCapture(0)
while cap.isOpened():
    ret,frame=cap.read()
    if ret:
        start=time.time()
        results=model(frame)
        end=time.time()
        fps=1/(end-start)
        print("[INFO] Fps :",1/(end-start))
        results=results.pandas().xyxy[0]
        if len(results)>0:
            for index, row in results.iterrows():
                xmin = int(row['xmin'])
                ymin = int(row['ymin'])
                xmax = int(row['xmax'])
                ymax = int(row['ymax'])
                confidence = row['confidence']
                class_id = row['class']
                name = row['name']
                if confidence>0.6:
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame,f"{name}:{confidence:.2f}",(xmin, ymin-10),cv2.FONT_HERSHEY_SIMPLEX,1,(233, 255, 10), 2)
                    cv2.putText(frame, "FPS:{0:.2f}".format(fps),(20, 25), cv2.FONT_HERSHEY_PLAIN,fontScale=2,color=(255, 0, 0),thickness=2)

    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  

cap.release()
cv2.destroyAllWindows()