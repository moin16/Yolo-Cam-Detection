from ultralytics import YOLO
import cv2

model = YOLO('..\Yolo-Weights\yolov8n.pt')
results= model("D:\IIT B\Others\Season Of Codes 23\YOLO Cam Detection\\boy.jpg", show=True)
cv2.waitKey(0)