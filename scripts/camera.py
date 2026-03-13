import time
import cv2
from pathlib import Path

OUTPUT_DIR = Path("webcam_captures")
OUTPUT_DIR.mkdir(exist_ok=True)

INTERVAL = 1

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

count = len(list(OUTPUT_DIR.glob("*.jpg")))
last_capture = time.time()

print("Appuie sur 'q' pour quitter")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    if now - last_capture >= INTERVAL:
        path = OUTPUT_DIR / f"capture_{count:04d}.jpg"
        cv2.imwrite(str(path), frame)
        count += 1
        last_capture = now

    cv2.putText(frame, f"Captures: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow("Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()