	
import torch
from super_gradients.training import models
import cv2
import numpy as np
from torchinfo import summary


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
CONFIDENCE_TRESHOLD = 0.35
model = models.get("yolo_nas_s", pretrained_weights="coco").to(device)



img=cv2.imread("20230424095405213567.jpg")
image_height, image_width, _ = img.shape
result = list(model.predict(img, conf=CONFIDENCE_TRESHOLD))[0]
classess=result.class_names
collect=zip(result.prediction.bboxes_xyxy,result.prediction.labels,result.prediction.confidence)
if len(result.prediction.labels)>0:
    for bbox,labels,confidence in collect:
            x1, y1, x2, y2 = np.array(bbox).astype("int")
            print(x1,y1,x2,y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("frame",img)
cv2.waitKey(0)