import cv2

from vehicle_detector import detect_vehicles
from signal_logic import signal_time

cap = cv2.VideoCapture(
    "videos/video.mp4"
)

while True:

    ret,frame = cap.read()

    if not ret:

        break

    frame,count = detect_vehicles(frame)

    green = signal_time(count)

    cv2.putText(
        frame,
        f"Vehicles:{count}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,0,0),
        2
    )

    cv2.putText(
        frame,
        f"Green:{green}s",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow(
        "Traffic",
        frame
    )

    if cv2.waitKey(1)==27:

        break

cap.release()
cv2.destroyAllWindows()