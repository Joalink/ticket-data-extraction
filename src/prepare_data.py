import os
import shutil
import yaml
from pathlib import Path
from PIL import Image

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
CONFIG_PATH = Path("config/data.yaml")

SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "test": "test"
}

def validate_image(image_path: Path) -> bool:
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError):
        return False

def validate_label(label_path: Path) -> bool:
    if not label_path.exists():
        return False
    if label_path.stat().st_size == 0:
        return False
    return True

def clean_split(split: str) -> tuple[int, int]:
    images_dir = DATA_RAW / split / "images"
    labels_dir = DATA_RAW / split / "labels"

    if not images_dir.exists():
        print(f"Split {split} not found, skipping")
        return 0, 0

    removed = 0
    kept = 0

    for image_path in images_dir.glob("*"):
        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        label_path = labels_dir / image_path.with_suffix(".txt").name

        if not validate_image(image_path):
            print(f"Corrupt image removed: {image_path.name}")
            image_path.unlink()
            removed += 1
            continue

        if not validate_label(label_path):
            print(f"Missing label removed: {image_path.name}")
            image_path.unlink()
            removed += 1
            continue

        kept += 1

    return kept, removed

def copy_to_processed(split: str, processed_split: str) -> None: 
    for folder in ["images", "labels"]:
        src = DATA_RAW / split / folder
        dst = DATA_PROCESSED / processed_split / folder

        if not src.exists():
            continue

        dst.mkdir(parents=True, exist_ok=True)

        for file in src.glob("*"):
            shutil.copy2(file, dst / file.name)

def build_procesed_data_yaml() -> None:
    raw_yaml_path = DATA_RAW / "data.yaml"

    with open(raw_yaml_path) as f:
        config = yaml.safe_load(f)

    processed_config = {
        "path": str(DATA_PROCESSED.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": config["nc"],
        "names": config["names"]
    }

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(processed_config, f, default_flow_style=False)
    
    print(f"Config saved to {CONFIG_PATH}")

def main() -> None:
    print("Starting data preparation")

    for split, processed_split in SPLIT_MAP.items():
        kept, removed = clean_split(split)
        print(f"{split} -> {processed_split}: kept {kept}, removed {removed}")
        copy_to_processed(split, processed_split)

    build_procesed_data_yaml()
    print("Data preparation complete")


if __name__ == "__main__":
    main()
    