import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_image_with_boxes(img: np.ndarray, boxes: list[list[int]],
                          title: str="") -> None:
    """
    Визуализирует изображение с нарисованными bounding boxes.

    Args:
        img (np.ndarray): исходное изображение (BGR формат).
        boxes (list[list[int]]): список боксов в формате [x1, y1, x2, y2].
        title (str): заголовок для изображения (по умолчанию пустая строка).

    Returns:
        None. Показывает изображение с прямоугольниками.
    """
    img_draw = img.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0,255,0), 2)
    plt.figure(figsize=(8,8))
    plt.imshow(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()
