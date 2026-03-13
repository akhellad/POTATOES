from collections import defaultdict, deque
import cv2
from ultralytics import YOLO

WINDOW_SIZE = 10
COLORS = {0: (0, 200, 0), 1: (0, 0, 220)}
NAMES = {0: "good", 1: "defect"}
CONF_THRESHOLD = {0: 0.6, 1: 0.35}

model = YOLO("best.pt")
track_history = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))

cap = cv2.VideoCapture("3838071939-preview.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter("output_tracked2.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, conf=0.3, iou=0.5, tracker="bytetrack.yaml", verbose=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        classes = results[0].boxes.cls.int().cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        for box, track_id, cls, conf in zip(boxes, track_ids, classes, confs):
            if conf < CONF_THRESHOLD[cls]:
                cls = 1
            
            track_history[track_id].append(cls)
            smoothed_cls = max(set(track_history[track_id]), key=list(track_history[track_id]).count)

            x1, y1, x2, y2 = map(int, box)
            color = COLORS[smoothed_cls]
            label = f"{NAMES[smoothed_cls]} {conf:.2f} id:{track_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out.write(frame)

cap.release()
out.release()