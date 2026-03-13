from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

IMAGES_DIR = Path("data_remapped/train/images")
LABELS_DIR = Path("data_remapped/train/labels")
NAMES = {0: "good", 1: "defect"}
COLORS = {0: "#00c800", 1: "#dc3c3c"}


def is_webcam_image(path: Path) -> bool:
    return "capture_" in path.stem


def read_labels(label_path: Path) -> list[list[float]]:
    if not label_path.exists():
        return []
    lines = label_path.read_text().strip().splitlines()
    return [[float(v) for v in line.split()] for line in lines if line.strip()]


def write_labels(label_path: Path, labels: list[list[float]]) -> None:
    lines = []
    for row in labels:
        cls = int(row[0])
        coords = " ".join(f"{v:.6f}" for v in row[1:])
        lines.append(f"{cls} {coords}")
    label_path.write_text("\n".join(lines))


def show_image(ax, img_rgb, labels, box_idx, info):
    h, w = img_rgb.shape[:2]
    ax.clear()
    ax.imshow(img_rgb)
    ax.set_title(info, fontsize=9, color="white", pad=4)
    ax.axis("off")

    for i, row in enumerate(labels):
        cls = int(row[0])
        cx, cy, bw, bh = row[1], row[2], row[3], row[4]
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        bw_px = bw * w
        bh_px = bh * h
        color = COLORS[cls]
        lw = 3 if i == box_idx else 1
        rect = patches.Rectangle((x1, y1), bw_px, bh_px, linewidth=lw, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1 - 4, NAMES[cls], color=color, fontsize=8, fontweight="bold" if i == box_idx else "normal")

    ax.figure.canvas.draw()


def main():
    images = sorted([p for p in IMAGES_DIR.glob("*.jpg") if is_webcam_image(p)])

    if not images:
        print("Aucune image webcam trouvée.")
        return

    print(f"{len(images)} images trouvées.")
    print("Touches : [g] good  [d] defect  [n] box suivante  [s] skip  [q] quitter")

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#1a1a1a")
    ax.set_facecolor("#1a1a1a")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)

    state = {"i": 0, "box_idx": 0, "labels": [], "modified": False, "done": False}

    def load_image(idx):
        img_path = images[idx]
        label_path = LABELS_DIR / (img_path.stem + ".txt")
        labels = read_labels(label_path)
        state["labels"] = labels
        state["box_idx"] = 0
        state["modified"] = False
        return img_path, label_path, labels

    def refresh():
        i = state["i"]
        img_path = images[i]
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        labels = state["labels"]
        box_idx = state["box_idx"]
        cls = int(labels[box_idx][0]) if labels else -1
        cls_name = NAMES[cls] if cls >= 0 else "no label"
        info = f"[{i+1}/{len(images)}] {img_path.name} | box {box_idx+1}/{len(labels)} | {cls_name}  —  g=good  d=defect  n=next box  s=skip  q=quit"
        show_image(ax, img_rgb, labels, box_idx, info)

    def next_image():
        i = state["i"]
        if state["modified"]:
            label_path = LABELS_DIR / (images[i].stem + ".txt")
            write_labels(label_path, state["labels"])
        state["i"] += 1
        if state["i"] >= len(images):
            print("Review terminé.")
            plt.close()
            return
        while state["i"] < len(images):
            img_path, label_path, labels = load_image(state["i"])
            if labels:
                break
            state["i"] += 1
        if state["i"] < len(images):
            refresh()

    def advance_box():
        labels = state["labels"]
        state["box_idx"] = (state["box_idx"] + 1) % len(labels)
        if state["box_idx"] == 0:
            next_image()
        else:
            refresh()

    def on_key(event):
        if not state["labels"]:
            return
        key = event.key
        box_idx = state["box_idx"]

        if key == "g":
            state["labels"][box_idx][0] = 0.0
            state["modified"] = True
            advance_box()
        elif key == "d":
            state["labels"][box_idx][0] = 1.0
            state["modified"] = True
            advance_box()
        elif key == "n":
            advance_box()
        elif key == "s":
            next_image()
        elif key == "q":
            i = state["i"]
            if state["modified"]:
                label_path = LABELS_DIR / (images[i].stem + ".txt")
                write_labels(label_path, state["labels"])
            print("Review interrompu.")
            plt.close()

    fig.canvas.mpl_connect("key_press_event", on_key)

    img_path, label_path, labels = load_image(0)
    while not labels and state["i"] < len(images) - 1:
        state["i"] += 1
        img_path, label_path, labels = load_image(state["i"])

    refresh()
    plt.show()


if __name__ == "__main__":
    main()