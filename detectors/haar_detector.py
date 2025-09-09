import numpy as np
import cv2

def load_haar_cascade():
    """
    Загружает каскад Хаара для лиц.

    Args:
        xml_path (str): путь к XML файлу (по умолчанию берется встроенный).

    Returns:
        cv2.CascadeClassifier: каскад Хаара.
    """
    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    haar_cascade = cv2.CascadeClassifier(haar_path)
    return haar_cascade

def detect_faces_haar(img: np.ndarray, haar_cascade: cv2.CascadeClassifier) -> list[list[int]]:
    """
    Детекция лиц через Haar Cascade.

    Args:
      img (np.ndarray): исходное изображение
      haar_cascade (cv2.CascadeClassifier): модель Haar Cascade

    Returns:
       list[list[int]]: список предсказанных боксов [x1, y1, x2, y2].
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray, 1.3, 5)
    boxes = [[x, y, x+w, y+h] for (x,y,w,h) in faces]
    return boxes
