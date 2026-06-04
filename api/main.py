import io
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from PIL import Image

from api.schemas import PredictResponse, HealthResponse
from api.dependencies import get_model
from src.inference import YOLOInference


app = FastAPI(
    title="YOLO Detection API",
    description="Object detection backend service",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    try:
        get_model()
        model_loaded = True
    except Exception:
        model_loaded = False

    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    confidence: float = 0.5,
    iou: float = 0.45,
    model: YOLOInference = Depends(get_model),
) -> PredictResponse:
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are supported",
        )

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    detections = model.predict(image, confidence=confidence, iou=iou)

    return PredictResponse(
        success=True,
        total_detections=len(detections),
        detections=detections,
    )