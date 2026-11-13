"""On-device services for the native Kulu assistant."""

import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import urllib.request


MODEL_SHA256 = "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
MODEL_URL = (
    "https://huggingface.co/opencv/object_detection_yolox/resolve/"
    "78c368f74ce73ee28fc7a1be418a598c71b58b52/object_detection_yolox_2022nov.onnx"
)
MODEL_MAX_BYTES = 100 * 1024 * 1024
COCO_LABELS = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


def run_local_command(command, cwd, on_output=None):
    """Run a confirmed command using the current user's normal OS privileges."""
    if os.name == "nt":
        import ctypes

        if ctypes.windll.shell32.IsUserAnAdmin():
            raise PermissionError(
                "Kulu will not execute commands when the app is running elevated. "
                "Restart it as a standard user."
            )
    process = subprocess.Popen(
        command,
        cwd=cwd,
        shell=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    if process.stdout is not None:
        for line in iter(process.stdout.readline, ""):
            if on_output is not None:
                on_output(line)
        process.stdout.close()
    return process.wait()


def _model_path():
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "Kulu" / "models" / "object_detection_yolox_2022nov.onnx"


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_object_model():
    model_path = _model_path()
    if model_path.is_file():
        if _sha256_file(model_path) == MODEL_SHA256:
            return model_path
        model_path.unlink()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=30) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MODEL_MAX_BYTES:
                raise RuntimeError("The object detection model exceeds the 100 MB safety limit.")
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=model_path.parent, prefix="model-", suffix=".download",
                delete=False,
            ) as downloaded:
                temporary_path = Path(downloaded.name)
                total_bytes = 0
                digest = hashlib.sha256()
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > MODEL_MAX_BYTES:
                        raise RuntimeError(
                            "The object detection model exceeds the 100 MB safety limit."
                        )
                    downloaded.write(chunk)
                    digest.update(chunk)
        if total_bytes < 1024 * 1024:
            raise RuntimeError("The downloaded object detection model is incomplete.")
        if digest.hexdigest() != MODEL_SHA256:
            raise RuntimeError("The downloaded model failed its SHA-256 integrity check.")
        os.replace(temporary_path, model_path)
        return model_path
    except Exception as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Object detection needs its official local ONNX model (about 35 MB). "
            "Connect to the internet once to download it, then recognition works offline. "
            f"Download failed: {exc}"
        ) from exc


_object_net = None
_ocr_engine = None


def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise RuntimeError(
                "OCR dependencies are missing. Install them with "
                "`python -m pip install -r requirements.txt`."
            ) from exc
        _ocr_engine = RapidOCR()
    return _ocr_engine


def _get_object_net():
    global _object_net
    if _object_net is None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is missing. Install dependencies with "
                "`python -m pip install -r requirements.txt`."
            ) from exc
        model_path = _download_object_model()
        try:
            _object_net = cv2.dnn.readNetFromONNX(str(model_path))
        except cv2.error as exc:
            model_path.unlink(missing_ok=True)
            raise RuntimeError(
                "The downloaded object detection model could not be loaded. "
                "It was removed; try again while online."
            ) from exc
    return _object_net


def _read_image(path):
    import cv2
    import numpy as np

    image_bytes = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not open this image: {path}")
    return image


def _detect_objects(image):
    import cv2
    import numpy as np

    net = _get_object_net()
    height, width = image.shape[:2]
    ratio = min(640 / height, 640 / width)
    resized = cv2.resize(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        (int(width * ratio), int(height * ratio)),
        interpolation=cv2.INTER_LINEAR,
    )
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    canvas[:resized.shape[0], :resized.shape[1]] = resized
    net.setInput(np.transpose(canvas.astype(np.float32), (2, 0, 1))[None, ...])
    output = net.forward()
    predictions = np.asarray(output).reshape(-1, 85)

    strides = (8, 16, 32)
    grids = []
    expanded_strides = []
    for stride in strides:
        grid_size = 640 // stride
        grid_x, grid_y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
        grids.append(np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2))
        expanded_strides.append(np.full((grid_size * grid_size, 1), stride))
    grid = np.concatenate(grids).astype(np.float32)
    expanded_stride = np.concatenate(expanded_strides).astype(np.float32)
    if predictions.shape[0] != grid.shape[0]:
        raise RuntimeError("The object detection model returned an unsupported output shape.")

    boxes = np.empty((len(predictions), 4), dtype=np.float32)
    boxes[:, :2] = (predictions[:, :2] + grid) * expanded_stride
    boxes[:, 2:] = np.exp(predictions[:, 2:4]) * expanded_stride
    boxes[:, :2] -= boxes[:, 2:] / 2
    class_scores = predictions[:, 4:5] * predictions[:, 5:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_scores)), class_ids]
    candidates = np.flatnonzero(scores >= 0.35)
    detections = []
    for class_id in np.unique(class_ids[candidates]):
        indices = candidates[class_ids[candidates] == class_id]
        class_boxes = boxes[indices].tolist()
        class_scores_for_nms = scores[indices].tolist()
        keep = cv2.dnn.NMSBoxes(class_boxes, class_scores_for_nms, 0.35, 0.5)
        for index in np.asarray(keep).reshape(-1):
            source_index = indices[int(index)]
            x, y, box_width, box_height = boxes[source_index]
            left = max(0, min(width, int(x / ratio)))
            top = max(0, min(height, int(y / ratio)))
            right = max(left, min(width, int((x + box_width) / ratio)))
            bottom = max(top, min(height, int((y + box_height) / ratio)))
            detections.append(
                {
                    "label": COCO_LABELS[int(class_ids[source_index])],
                    "score": float(scores[source_index]),
                    "box": (left, top, right, bottom),
                }
            )
    return detections


def analyze_image(path, use_ocr=True, use_objects=True):
    """Analyze an image locally and return text, detections, and a preview image."""
    import cv2
    import numpy as np

    image = _read_image(path)
    ocr_results = []
    object_results = []
    errors = []

    if use_ocr:
        try:
            ocr_result, _ = _get_ocr_engine()(image)
            for item in ocr_result or ():
                if len(item) >= 3:
                    ocr_results.append(
                        {"text": str(item[1]), "score": float(item[2]), "box": item[0]}
                    )
        except Exception as exc:
            errors.append(f"OCR: {exc}")

    if use_objects:
        try:
            object_results = _detect_objects(image)
        except Exception as exc:
            errors.append(f"Object detection: {exc}")

    preview = image.copy()
    for detection in object_results:
        left, top, right, bottom = detection["box"]
        cv2.rectangle(preview, (left, top), (right, bottom), (50, 180, 90), 2)
        label = f"{detection['label']} {detection['score']:.0%}"
        cv2.putText(
            preview, label, (left, max(18, top - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 120, 55), 2,
        )
    for result in ocr_results:
        points = np.asarray(result["box"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(preview, [points], True, (220, 100, 40), 2)

    return {
        "ocr": ocr_results,
        "objects": object_results,
        "errors": errors,
        "ocr_requested": use_ocr,
        "objects_requested": use_objects,
        "preview": cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
    }
