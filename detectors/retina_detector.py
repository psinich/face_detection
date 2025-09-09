import numpy as np
import insightface

def load_retina(model_name="scrfd_500m_bnkps", det_size=(640, 640)):
    """
    Загружает SCRFD через InsightFace FaceAnalysis.

    Args:
        model_name (str): имя модели (например, scrfd_500m_bnkps).
        det_size (tuple): размер входа (по умолчанию 640x640).

    Returns:
        FaceAnalysis: объект детектора.
    """
    app = insightface.app.face_analysis.FaceAnalysis(allowed_modules=['detection'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

def detect_faces_retina(img: np.ndarray, app: insightface.app.face_analysis.FaceAnalysis, conf_threshold=0.5) -> list[list[int]]:
    """
    Детектирует лица с использованием RetinaFace через FaceAnalysis.

    Args:
        img (np.ndarray): BGR-изображение.
        conf_threshold (float): порог уверенности. По умолчанию 0.5
        app (insightface.app.face_analysis.FaceAnalysis): модель RetinaFace

    Returns:
        list[list[int]]: список боксов [x1, y1, x2, y2].
    """
    faces = app.get(img)  # детектируем
    boxes = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        boxes.append([x1, y1, x2, y2])
    return boxes
