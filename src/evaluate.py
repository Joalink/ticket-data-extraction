from pathlib import Path

import yaml
from ultralytics import YOLO

import mlflow

CONFIG_PATH = Path("config/training_config.yaml")
WEIGHTS_PATH = Path("models/weights/best.pt")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def evaluate(config: dict) -> dict:

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Weights not found at {WEIGHTS_PATH}")

    model = YOLO(str(WEIGHTS_PATH))

    results = model.val(
        data=config["paths"]["data_yaml"],
        imgsz=config["model"]["imgsz"],
        split="test",
    )

    return {
        "map50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
        "map50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
        "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
        "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
    }


def main() -> None:
    config = load_config()

    tracking_uri = f"sqlite:///{config['paths']['mlflow_db']}"
    mlflow.set_tracking_uri(tracking_uri)

    print("Running evaluation on test set")
    metrics = evaluate(config)

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
