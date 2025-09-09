import os
import cv2
import json
import time
import numpy as np
from tqdm import tqdm

from detectors.dnn_detector import load_dnn_model, detect_faces_dnn
from detectors.haar_detector import load_haar_cascade, detect_faces_haar
from detectors.retina_detector import load_retina, detect_faces_retina
from utils.evaluation import calculate_precision_recall

def run_pipeline(detector_name, annotations, data_path, n_images=50):
    """
    Запускает пайплайн для указанного метода.

    Args:
        detector_name (str): 'dnn', 'haar' или 'retina'.
        annotations (list[dict]): список аннотаций из JSON.
        data_path (str): путь к изображениям.
        n_images (int): количество изображений для теста.

    Returns:
        dict: {'precision': float, 'recall': float, 'time': float}.
    """
    all_precisions, all_recalls, times = [], [], []

    if detector_name == "dnn":
        net = load_dnn_model("opencv_face_detector/opencv_face_detector_uint8.pb",
                             "opencv_face_detector/opencv_face_detector.pbtxt")
    elif detector_name == "haar":
        cascade = load_haar_cascade()
    elif detector_name == "retina":
        app = load_retina()
    else:
        raise ValueError("Unknown detector")

    for entry in tqdm(annotations[:n_images]):
        img_path = os.path.join(data_path, os.path.basename(entry["img_path"])).replace("\\", "/")
        img = cv2.imread(img_path)
        if img is None:
            continue

        gt_boxes = [[x, y, x+w, y+h] for (x, y, w, h) in entry["annotations"]["bbox"]]

        start = time.time()
        if detector_name == "dnn":
            pred_boxes = detect_faces_dnn(img, net)
        elif detector_name == "haar":
            pred_boxes = detect_faces_haar(img, cascade)
        elif detector_name == "retina":
            pred_boxes = detect_faces_retina(img, app)
        else:
            raise ValueError("Unsupported value")
        end = time.time()

        times.append(end - start)
        precision, recall = calculate_precision_recall(pred_boxes, gt_boxes)
        all_precisions.append(precision)
        all_recalls.append(recall)

    return {
        "precision": np.mean(all_precisions),
        "recall": np.mean(all_recalls),
        "time": np.mean(times)
    }

if __name__ == "__main__":
    DATA_PATH = "/app/test_data"
    ANNOT_JSON = "/app/test_data/annos.json"

    with open(ANNOT_JSON, "r") as f:
        annotations = json.load(f)

    results = {}
    for method in ["dnn", "haar", "retina"]:
        print(f"\nRunning {method.upper()} detector...")
        results[method] = run_pipeline(method, annotations, DATA_PATH, n_images=50)

    print("\n=== Итоговый сравнительный отчет ===")
    print(f"{'Method':<10} | {'Precision':<10} | {'Recall':<10} | {'Avg Time (s)':<12}")
    print("-" * 50)
    for method, metrics in results.items():
        print(f"{method:<10} | {metrics['precision']:.4f}    | "
              f"{metrics['recall']:.4f}    | {metrics['time']:.4f}")
