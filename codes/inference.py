import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from torchvision.ops import nms
from PIL import Image
from collections import defaultdict

# =====================================================
# CONFIGURATION
# =====================================================
YOLO_A_PATH = "/home/faustino/Documents/detection_project/yolo_ensemble/train/runs_yolo/yolo_frequent/weights/best.pt"
YOLO_B_PATH = "/home/faustino/Documents/detection_project/yolo_ensemble/train/runs_yolo/yolo_rare/weights/best.pt"

TEST_IMAGES = "/home/faustino/Documents/detection_project/yolo_ensemble/datasets/all/images/test"
TEST_LABELS = "/home/faustino/Documents/detection_project/yolo_ensemble/datasets/all/labels/test"

OUTPUT_DIR = "evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_CLASSES = 14
BG = NUM_CLASSES            # background index
IOU_THRESHOLD = 0.5

MAP_A = {0: 1, 1: 2, 2: 6, 3: 10}
MAP_B = {0: 0, 1: 3, 2: 4, 3: 5, 4: 7, 5: 8, 6: 9, 7: 11, 8: 12, 9: 13}

CLASS_NAMES = [str(i) for i in range(NUM_CLASSES)] + ["background"]

# =====================================================
# LOAD MODELS
# =====================================================
yolo_A = YOLO(YOLO_A_PATH)
yolo_B = YOLO(YOLO_B_PATH)

# =====================================================
# GT LOADING (YOLO → PIXELS)
# =====================================================
def yolo_to_xyxy(line, w, h):
    cls, xc, yc, bw, bh = map(float, line.split())
    x1 = (xc - bw / 2) * w
    y1 = (yc - bh / 2) * h
    x2 = (xc + bw / 2) * w
    y2 = (yc + bh / 2) * h
    return {"class": int(cls), "bbox": [x1, y1, x2, y2]}

def load_gt(label_path, w, h):
    gt = []
    if not os.path.exists(label_path):
        return gt
    with open(label_path) as f:
        for line in f:
            gt.append(yolo_to_xyxy(line, w, h))
    return gt

# =====================================================
# YOLO PRED EXTRACTION (SAFE)
# =====================================================
def extract_preds(result, class_map):
    preds = []
    if result.boxes is None:
        return preds
    for i in range(len(result.boxes.cls)):
        local_cls = int(result.boxes.cls[i].item())
        if local_cls not in class_map:
            continue
        preds.append({
            "class": class_map[local_cls],
            "bbox": result.boxes.xyxy[i].tolist(),
            "score": result.boxes.conf[i].item()
        })
    return preds

# =====================================================
# IOU
# =====================================================
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0

# =====================================================
# ENSEMBLE INFERENCE
# =====================================================
def run_ensemble(img_path, conf_A=0.25, conf_B=0.12):
    rA = yolo_A(img_path, conf=conf_A)[0]
    rB = yolo_B(img_path, conf=conf_B)[0]

    preds = extract_preds(rA, MAP_A) + extract_preds(rB, MAP_B)
    if not preds:
        return []

    boxes = torch.tensor([p["bbox"] for p in preds])
    scores = torch.tensor([p["score"] for p in preds])

    keep = nms(boxes, scores, IOU_THRESHOLD)
    return [preds[i] for i in keep]

# =====================================================
# DATASET INFERENCE
# =====================================================
def infer_dataset():
    preds, gts = {}, {}
    for img_name in os.listdir(TEST_IMAGES):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(TEST_IMAGES, img_name)
        label_path = os.path.join(TEST_LABELS, img_name.replace(".jpg", ".txt"))

        with Image.open(img_path) as img:
            w, h = img.size

        preds[img_name] = run_ensemble(img_path)
        gts[img_name] = load_gt(label_path, w, h)

    return preds, gts

# =====================================================
# EVALUATION WITH BACKGROUND
# =====================================================
def evaluate(preds, gts):
    stats = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})
    cm = np.zeros((NUM_CLASSES + 1, NUM_CLASSES + 1), dtype=int)

    for img in preds:
        gt_used = [False] * len(gts[img])

        for p in preds[img]:
            matched = False
            for i, g in enumerate(gts[img]):
                if gt_used[i]:
                    continue
                if iou(p["bbox"], g["bbox"]) >= IOU_THRESHOLD:
                    matched = True
                    gt_used[i] = True
                    cm[g["class"], p["class"]] += 1
                    stats[p["class"]]["TP"] += 1
                    break

            if not matched:
                cm[BG, p["class"]] += 1
                stats[p["class"]]["FP"] += 1

        for i, g in enumerate(gts[img]):
            if not gt_used[i]:
                cm[g["class"], BG] += 1
                stats[g["class"]]["FN"] += 1

    return stats, cm

# =====================================================
# METRICS
# =====================================================
def compute_metrics(stats):
    precision, recall = [], []
    for c in range(NUM_CLASSES):
        tp = stats[c]["TP"]
        fp = stats[c]["FP"]
        fn = stats[c]["FN"]
        precision.append(tp / (tp + fp) if tp + fp else 0)
        recall.append(tp / (tp + fn) if tp + fn else 0)
    return precision, recall

def compute_global_recall(stats):
    TP = sum(stats[c]["TP"] for c in stats)
    FN = sum(stats[c]["FN"] for c in stats)
    return TP / (TP + FN) if TP + FN > 0 else 0

# =====================================================
# PLOTS
# =====================================================
def normalize_cm(cm):
    cm_norm = np.zeros_like(cm, dtype=float)
    for i in range(cm.shape[0]):
        s = cm[i].sum()
        if s > 0:
            cm_norm[i] = cm[i] / s
    return cm_norm

def plot_confusion_matrix_percent(cm):
    cmn = normalize_cm(cm)
    plt.figure(figsize=(11, 9))
    plt.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(label="Percentage")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=90)
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.title("Confusion Matrix with Background (%)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_percent.png")
    plt.close()

def plot_recall_curve(preds, gts):
    confs = np.linspace(0.05, 0.95, 10)
    recalls = []

    for c in confs:
        filtered = {k: [p for p in preds[k] if p["score"] >= c] for k in preds}
        stats, _ = evaluate(filtered, gts)
        recalls.append(compute_global_recall(stats))

    plt.figure()
    plt.plot(confs, recalls, marker="o")
    plt.xlabel("Confidence threshold")
    plt.ylabel("Recall")
    plt.title("Recall vs Confidence (YOLO-style)")
    plt.grid()
    plt.savefig(f"{OUTPUT_DIR}/recall_curve.png")
    plt.close()

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    print("[INFO] Inference...")
    preds, gts = infer_dataset()

    print("[INFO] Evaluation...")
    stats, cm = evaluate(preds, gts)
    precision, recall = compute_metrics(stats)

    print("[INFO] Plots...")
    plot_confusion_matrix_percent(cm)
    plot_recall_curve(preds, gts)

    print("[DONE] Everything completed.")
