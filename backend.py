import requests
import cv2
import numpy as np
from label_studio_ml.model import LabelStudioMLBase
from ultralytics import YOLO

LABEL_STUDIO_URL = "http://localhost:8080"
API_TOKEN = "aabdde767bcde01799ef9a645706e1be558c188f"

class PotatoDetector(LabelStudioMLBase):
    def __init__(self, **kwargs):
        kwargs["model_dir"] = "/tmp/label_studio_ml"
        super().__init__(**kwargs)
        self.model = YOLO("best (4).pt")
        self.labels = ["good", "defect"]

    def predict(self, tasks, **kwargs):
        predictions = []
        for task in tasks:
            image_url = task["data"]["image"]
            if image_url.startswith("/data/"):
                image_url = LABEL_STUDIO_URL + image_url

            response = requests.get(image_url, headers={"Authorization": f"Token {API_TOKEN}"})
            print(f"URL: {image_url}")
            print(f"Status: {response.status_code}")
            print(f"Content length: {len(response.content)}")
            img_array = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            results = self.model(img, conf=0.3)[0]
            boxes = results.boxes

            regions = []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = results.orig_shape[1]
                h = results.orig_shape[0]
                regions.append({
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": {
                        "x": x1 / w * 100,
                        "y": y1 / h * 100,
                        "width": (x2 - x1) / w * 100,
                        "height": (y2 - y1) / h * 100,
                        "rectanglelabels": [self.labels[int(box.cls[0])]]
                    },
                    "score": float(box.conf[0])
                })

            predictions.append({"result": regions})
        return predictions