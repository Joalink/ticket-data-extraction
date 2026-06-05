from pydantic import BaseModel


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox


class PredictResponse(BaseModel):
    success: bool
    total_detections: int
    detections: list[Detection]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool