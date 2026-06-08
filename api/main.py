import io

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from PIL import Image

from api.dependencies import get_model
from api.schemas import HealthResponse, PredictResponse
from src.inference import YOLOInference

app = FastAPI(
    title="Ticket Data Extraction API",
    description="API for extracting data from tickets using a YOLOv8 model",
    version="1.0.0",
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
