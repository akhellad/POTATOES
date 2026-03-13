import random
import urllib.request
from pathlib import Path
from pycocotools.coco import COCO

CATEGORIES = ["person", "bowl", "apple", "orange"]
N_PER_CATEGORY = 100
OUTPUT_DIR = Path("coco_negatives")
ANNOTATIONS_FILE = "instances_train2017.json"

OUTPUT_DIR.mkdir(exist_ok=True)

coco = COCO(ANNOTATIONS_FILE)

image_ids = set()
for cat_name in CATEGORIES:
    cat_ids = coco.getCatIds(catNms=[cat_name])
    ids = coco.getImgIds(catIds=cat_ids)
    sampled = random.sample(ids, min(N_PER_CATEGORY, len(ids)))
    image_ids.update(sampled)

images = coco.loadImgs(list(image_ids))

for i, img in enumerate(images):
    dest = OUTPUT_DIR / img["file_name"].split("/")[-1]
    if not dest.exists():
        urllib.request.urlretrieve(img["coco_url"], dest)
    if (i + 1) % 50 == 0:
        print(f"{i + 1}/{len(images)}")

print(f"Done: {len(images)} images saved to {OUTPUT_DIR}")