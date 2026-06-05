from ultralytics import YOLO

model = YOLO("models/weights/best.pt")
results = model.predict(
    source="data/test_api/",
    save=True,
    conf=0.5,
    project="results",
    name="demo",
)

assert results is not None, "Error: Results should not be None."
assert len(results) > 0, "Error: The model did not return any predictions."
