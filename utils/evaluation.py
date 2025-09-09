import numpy as np

def iou(boxA: list[int], boxB: list[int]) -> float:
    """
    Вычисляет метрику Intersection over Union (IoU) между двумя bounding boxes.

    Args:
        boxA (list[int]): бокс в формате [x1, y1, x2, y2].
        boxB (list[int]): бокс в формате [x1, y1, x2, y2].

    Returns:
        float: значение IoU (от 0 до 1).
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def calculate_precision_recall(pred_boxes: list[list[int]], gt_boxes: list[list[int]],
                               iou_threshold: float=0.5) -> tuple[float]:
    """
    Считает precision и recall для одного изображения.

    Args:
        pred_boxes (list[list[int]]): предсказанные боксы в формате [x1, y1, x2, y2].
        gt_boxes (list[list[int]]): ground truth боксы в формате [x1, y1, x2, y2].
        iou_threshold (float): порог для IoU, выше которого предсказание считается true positive.

    Returns:
        tuple:
            precision (float): доля корректно найденных лиц среди всех найденных.
            recall (float): доля найденных лиц среди всех лиц на изображении.
    """
    matched_gt = set()
    tp, fp = 0, 0
    for pb in pred_boxes:
        match_found = False
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            if iou(pb, gt) >= iou_threshold:
                tp += 1
                matched_gt.add(i)
                match_found = True
                break
        if not match_found:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return precision, recall
