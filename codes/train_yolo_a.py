from ultralytics import YOLO

def train_yolo_frequent():
    model = YOLO("yolov8m.pt")  # modèle de départ (pretrained)

    results = model.train(
        task="detect",
        data="/home/faustino/Documents/detection_project/yolo_ensemble/configs/data_frequent.yaml",
        epochs=20,
        imgsz=512,
        batch=8,
        workers=4,
        cache=False,
        project="runs_yolo",
        name="yolo_frequent",
        exist_ok=True
    )

    return results


if __name__ == "__main__":
    train_yolo_frequent()
