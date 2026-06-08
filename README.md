# Ticket Data Extraction API

Computer vision API that automatically extracts structured data from receipt images using a fine-tuned YOLOv8n model deployed on Render.


## 🚀 Projects

### Model

- Architecture: YOLOv8n fine-tuned on 1500+ annotated receipt images
- Dataset: Roboflow — 8 field classes, train/val/test split
- Export: ONNX format for optimized CPU inference
- Metrics: mAP50: 0.8071, Precision: 0.7653, Recall: 0.7484

### Detected fields (classes)

| Class | Description |
|-------|-------------|
| Address | Store or delivery address |
| Date | Transaction date |
| Item | Individual line items |
| OrderId | Order or receipt number |
| Subtotal | Pre-tax total |
| Tax | Tax amount |
| Title | Store or receipt title |
| TotalPrice | Final total amount |

### Image structured JSON with every detected field and its location.

```json
{
  "success": true,
  "total_detections": 5,
  "detections": [
    {
      "class_id": 3,
      "class_name": "OrderId",
      "confidence": 0.91,
      "bbox": { "x1": 120.5, "y1": 45.2, "x2": 380.1, "y2": 89.7 }
    }
  ]
}
```



## 🛠 Tech stack

| Layer | Technology |
|-------|------------|
| Model training | Ultralytics YOLOv8 |
| Experiment tracking | MLflow |
| API | FastAPI + ONNX Runtime |
| Containerization | Docker |
| Dependency management | uv |
| Deployment | Render |
| Dev environment | Dev Containers |



## 📁 Project structure

```
ticket-data-extraction/
│
├── .devcontainer/
│   ├── devcontainer.json # Dev container configuration
│   └── Dockerfile
│
├── api/
│   ├── main.py          # FastAPI app
│   ├── schemas.py       # Request/response models
│   ├── dependencies.py  # Model loading
├── config/
│   ├── training_config.yaml
│   └── data.yaml
│
├── models/
│   └── best.onnx        # Exported model weights
│
├── src/
│   ├── prepare_data.py  # Data validation and processing
│   ├── train.py         # Fine-tuning pipeline
│   ├── evaluate.py      # Test set evaluation
│   └── inference.py     # ONNX inference logic
│
│
├── data/
│   ├── raw/             # Original Roboflow dataset
│   └── processed/       # Cleaned and validated data
│
├── Dockerfile
├── pyproject.toml
└── README.md
```

## Getting Started

### Prerequisites
- [Docker](https://www.docker.com/) (for Dev Containers)
- [VS Code](https://code.visualstudio.com/) + [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### Local setup

Requirements: Docker, VS Code, Dev Containers extension.


1. Clone the repo:

```bash
git clone https://github.com/your-username/ticket-data-extraction
cd ticket-data-extraction
```

2. Open in VS Code and Reopen in dev Container

3. Prepare data (place Roboflow dataset in data/raw first)

```bash
uv run python src/prepare_data.py
```

3. Train the model

```bash
uv run python src/train.py
```
4. Evaluate
```bash
uv run python src/evaluate.py
```

5. Start the API

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

### Training pipeline flow

```
raw data (Roboflow)
  → prepare_data.py    validates images, renames splits, generates config/data.yaml
    → train.py         fine-tunes YOLOv8n, logs to MLflow, saves best.pt
      → evaluate.py    runs on test set, reports mAP50/precision/recall
        → export ONNX  optimizes model for production inference
          → deploy      Docker container served via FastAPI on Render
```
---
### MLflow tracking

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --host 0.0.0.0 --port 5000
```

Open `http://localhost:5000` to view experiments, metrics, and model registry.

## Environment variables

| Variable | Description |
|----------|-------------|
| PYTHONPATH | Set to /app on Render |

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Health check |
| `POST` | `/api/v1/predict` | Upload and analize image |

## Deployment
The service is deployed on [Render](https://render.com/)

Base URL: `https://ticket-extraction-api.onrender.com`

> Note: free tier sleeps after 15 minutes of inactivity. First request may take 30 seconds to wake up.


## License

MIT © [JoaLink](https://github.com/JoaLink)