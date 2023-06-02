	
import torch
from super_gradients.training import models
import numpy as np
import time
import cv2
from torchinfo import summary


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(device)
CONFIDENCE_TRESHOLD = 0.35
model = models.get("yolo_nas_m", pretrained_weights="coco").to(device)

cap=cv2.VideoCapture(0)
while cap.isOpened():
    ret,frame=cap.read()
    if ret:
        start=time.time()
        result = list(model.predict(frame, conf=CONFIDENCE_TRESHOLD))[0]
        end=time.time()
        print("[INFO] Fps :",1/(end-start))
        classess=result.class_names
        collect=zip(result.prediction.bboxes_xyxy,result.prediction.labels,result.prediction.confidence)
        if len(result.prediction.labels)>0:
            for bbox,labels,confidence in collect:
                    if confidence>0.6:
                        x1, y1, x2, y2 = np.array(bbox).astype("int")
                        print(x1,y1,x2,y2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame,f"{classess[int(labels)]}:{confidence:.2f}",(x1, y1-10),cv2.FONT_HERSHEY_SIMPLEX,1,(233, 255, 10), 2)

    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  

cap.release()
cv2.destroyAllWindows()
    