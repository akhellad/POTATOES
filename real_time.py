from collections import defaultdict, deque
import time
import cv2
from ultralytics import YOLO

WINDOW_SIZE = 15
GHOST_FRAMES = 8
COLORS = {0: (0, 200, 0), 1: (0, 0, 220)}
NAMES = {0: "good", 1: "defect"}
CONF_THRESHOLD = {0: 0.4, 1: 0.8}

model = YOLO("best (6).pt")
track_history = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))
last_box = {}
last_cls = {}
ghost_counter = defaultdict(int)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps_history = deque(maxlen=30)
prev_time = time.time()
writer = None
recording = False


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, conf=0.25, iou=0.5, tracker="bytetrack.yaml", verbose=False)

    active_ids = set()

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        classes = results[0].boxes.cls.int().cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        for box, track_id, cls, conf in zip(boxes, track_ids, classes, confs):
            if conf < CONF_THRESHOLD[cls]:
                cls = 0

            track_history[track_id].append(cls)
            smoothed_cls = max(set(track_history[track_id]), key=list(track_history[track_id]).count)

            last_box[track_id] = box
            last_cls[track_id] = smoothed_cls
            ghost_counter[track_id] = 0
            active_ids.add(track_id)

    for track_id in list(last_box.keys()):
        if track_id not in active_ids:
            ghost_counter[track_id] += 1
            if ghost_counter[track_id] > GHOST_FRAMES:
                del last_box[track_id]
                del last_cls[track_id]
                del ghost_counter[track_id]

    for track_id, box in last_box.items():
        smoothed_cls = last_cls[track_id]
        is_ghost = track_id not in active_ids
        x1, y1, x2, y2 = map(int, box)
        color = COLORS[smoothed_cls]
        alpha = 0.5 if is_ghost else 1.0
        label = f"{NAMES[smoothed_cls]} id:{track_id}"

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overlay, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    curr_time = time.time()
    fps_history.append(1.0 / (curr_time - prev_time))
    prev_time = curr_time
    avg_fps = sum(fps_history) / len(fps_history)

    cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    if recording:
        writer.write(frame)
        cv2.putText(frame, "REC", (frame.shape[1] - 70, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 220), 2)

    cv2.imshow("Potato Defect Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        if not recording:
            filename = f"demo_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
            recording = True
            print(f"Enregistrement démarré : {filename}")
        else:
            recording = False
            writer.release()
            writer = None
            print("Enregistrement arrêté.")

if recording and writer:
    writer.release()

cap.release()
cv2.destroyAllWindows()