from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

vehicle_classes = [
    'car',
    'motorcycle',
    'bus',
    'truck'
]

def detect_vehicles(frame):

    results = model(frame)

    count = 0

    for r in results:

        boxes = r.boxes

        for box in boxes:

            cls = int(box.cls[0])

            name = model.names[cls]

            if name in vehicle_classes:

                count += 1

                x1,y1,x2,y2 = map(int,
                                  box.xyxy[0])

                cv2.rectangle(
                    frame,
                    (x1,y1),
                    (x2,y2),
                    (0,255,0),
                    2
                )

                cv2.putText(
                    frame,
                    name,
                    (x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,0),
                    2
                )

    return frame,count