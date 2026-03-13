import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
import yaml

def visualize_random_samples(dataset_path, n=5):
    dataset_path = Path(dataset_path)
    
    with open(dataset_path.parent.parent / "data.yaml") as f:
        classes = yaml.safe_load(f)["names"]
    
    colors = plt.cm.get_cmap("tab10", len(classes))
    images_dir = dataset_path
    files = list(images_dir.iterdir())
    samples = random.sample(files, n)
    
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    
    for ax, img_path in zip(axes, samples):
        img = Image.open(img_path)
        w, h = img.size
        ax.imshow(img)
        
        label_path = img_path.parent.parent / "labels" / img_path.with_suffix(".txt").name
        
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    cls, xc, yc, bw, bh = map(float, line.strip().split())
                    cls = int(cls)
                    x1 = (xc - bw / 2) * w
                    y1 = (yc - bh / 2) * h
                    rect = patches.Rectangle(
                        (x1, y1), bw * w, bh * h,
                        linewidth=2, edgecolor=colors(cls), facecolor="none"
                    )
                    ax.add_patch(rect)
                    ax.text(x1, y1 - 4, classes[cls], color=colors(cls), fontsize=8)
        
        ax.axis("off")
        ax.set_title(img_path.name, fontsize=7)
    
    plt.tight_layout()
    plt.show()

visualize_random_samples("data_remapped/train/images", n=5)