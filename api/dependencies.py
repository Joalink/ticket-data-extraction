from functools import lru_cache
from src.inference import YOLOInference


@lru_cache(maxsize=1)
def get_model() -> YOLOInference:
    return YOLOInference()