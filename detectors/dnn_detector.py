import cv2
import numpy as np

def load_dnn_model(model_path, config_path):
    """
    Загружает предобученный OpenCV DNN face detector.

    Args:
        model_path (str): путь к .pb файлу (веса модели).
        config_path (str): путь к .pbtxt файлу (конфигурация).

    Returns:
        cv2.dnn_Net: загруженная модель.
    """
    return cv2.dnn.readNetFromTensorflow(model_path, config_path)

def detect_faces_dnn(img: np.ndarray, net: cv2.dnn.Net, conf_threshold: float=0.5) -> list[list[int]]:
    """
    Детектирует лица с помощью OpenCV DNN face detector.

    Args:
        img (np.ndarray): исходное изображение (BGR формат).
        conf_threshold (float): порог вероятности (по умолчанию 0.5).
        net (cv2.dnn.Net): модель OpenCV DNN

    Returns:
        list[list[int]]: список предсказанных боксов [x1, y1, x2, y2].
    """
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            boxes.append([x1, y1, x2, y2])
    return boxes
