from ultralytics import YOLO
import cv2
import cvzone
import math
import cv2
import cvzone
import math
from collections import OrderedDict

cap = cv2.VideoCapture("D:\IIT B\Others\Season Of Codes 23\YOLO Cam Detection\pexels_videos_2034115 (1080p).mp4")

model = YOLO('..\Yolo-Weights\yolov8s.pt')

classnames = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck',
              'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
              'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
              'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
              'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
              'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
              'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
              'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant',
              'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote',
              'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
              'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Dictionary to store object IDs and their class names
tracked_objects = {}

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # Dictionary to store class counts
    class_counts = OrderedDict()
    for class_name in classnames:
        class_counts[class_name] = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1, y1, x2, y2)

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            print(conf)

            # Class Name
            cls = int(box.cls[0])
            current_class = classnames[cls]

            if current_class in ['car', 'person', 'motorbike'] and conf > 0.3:
                cvzone.putTextRect(img, f"{current_class} {conf}", (max(0, x1), max(35, y1)), scale=1, thickness=1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Update tracked_objects dictionary with the current object's ID and class name
                object_id = id(box)
                tracked_objects[object_id] = current_class

            # Count the objects of each class
            if object_id in tracked_objects:
                class_name = tracked_objects[object_id]
                class_counts[class_name] += 1

    # Display the object count on the image
    for class_name, count in class_counts.items():
        text = f"{class_name}: {count}"
        cv2.putText(img, text, (10, 30 + 20 * list(class_counts.keys()).index(class_name)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
