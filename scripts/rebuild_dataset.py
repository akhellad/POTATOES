import shutil
import yaml
from pathlib import Path

REMAP = {0: 1, 1: 1, 2: 1, 3: 0, 4: 1}

def remap_dataset(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)

    for split in ["train", "valid", "test"]:
        for subdir in ["images", "labels"]:
            (dst / split / subdir).mkdir(parents=True, exist_ok=True)

    for img_path in src.rglob("images/*.jpg"):
        rel = img_path.relative_to(src)
        shutil.copy(img_path, dst / rel)

    for label_path in src.rglob("labels/*.txt"):
        rel = label_path.relative_to(src)
        lines_out = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                parts[0] = str(REMAP[int(parts[0])])
                lines_out.append(" ".join(parts))
        with open(dst / rel, "w") as f:
            f.write("\n".join(lines_out))

    new_yaml = {
        "train": str(dst / "train" / "images"),
        "val": str(dst / "valid" / "images"),
        "test": str(dst / "test" / "images"),
        "nc": 2,
        "names": ["good", "defect"],
    }
    with open(dst / "data.yaml", "w") as f:
        yaml.dump(new_yaml, f, default_flow_style=False, allow_unicode=True)

remap_dataset(Path("data"), Path("data_remapped"))