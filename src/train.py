from pathlib import Path

import mlflow.artifacts
import yaml
from ultralytics import YOLO

import mlflow

CONFIG_PATH = Path("config/training_config.yaml")
# WEIGHTS_PATH = Path(config["paths"]["weights"])


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def setup_mlflow(config: dict) -> None:
    tracking_uri = f"sqlite:///{config['paths']['mlflow_db']}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])


def train(config: dict) -> tuple[YOLO, dict]:
    model = YOLO(config["model"]["base"])

    results = model.train(
        data=config["paths"]["data_yaml"],
        epochs=config["training"]["epochs"],
        imgsz=config["model"]["imgsz"],
        batch=config["training"]["batch"],
        lr0=config["training"]["lr0"],
        patience=config["training"]["patience"],
        save_period=config["training"]["save_period"],
        workers=config["training"]["workers"],
        device=config["training"]["device"],
        project=config["paths"]["output"],
        name="run",
        exist_ok=True,
        flipud=config["augmentation"]["flipud"],
        fliplr=config["augmentation"]["fliplr"],
        degrees=config["augmentation"]["degrees"],
        translate=config["augmentation"]["translate"],
        scale=config["augmentation"]["scale"],
        mosaic=config["augmentation"]["mosaic"],
        hsv_h=config["augmentation"]["hsv_h"],
        hsv_s=config["augmentation"]["hsv_s"],
        hsv_v=config["augmentation"]["hsv_v"],
    )

    metrics = {
        "map50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
        "map50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
        "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
        "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
    }

    return model, metrics


def register_model(
    config: dict,
    metrics: dict,
    weights_path: Path,
) -> bool:
    min_map50 = config["mlflow"]["min_map50"]

    if metrics["map50"] < min_map50:
        print(
            f"mAP50 {metrics['map50']:.3f} below threshold {min_map50}. Not registering."
        )
        return False

    with mlflow.start_run():
        mlflow.log_params(
            {
                "base_model": config["model"]["base"],
                "epochs": config["training"]["epochs"],
                "imgsz": config["model"]["imgsz"],
                "batch": config["training"]["batch"],
                "lr0": config["training"]["lr0"],
            }
        )

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(weights_path))

        mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/artifacts/best.pt",
            config["mlflow"]["model_name"],
        )

    print(f"Model registered with mAP50: {metrics['map50']:.3f}")
    return True


def main() -> None:
    config = load_config(CONFIG_PATH)
    setup_mlflow(config)

    print("Starting training")
    model, metrics = train(config)
    print(f"Training complete. Metrics: {metrics}")

    weights_path = Path(config["paths"]["weights"])

    # weights_path = Path(config["paths"]["output"]) / "run" / "weights" / "best.pt"

    if weights_path.exists():
        best_dest = Path("models/weights/best.pt")
        best_dest.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy2(weights_path, best_dest)
        register_model(config, metrics, best_dest)
    else:
        print("Best weights not found")


if __name__ == "__main__":
    main()
