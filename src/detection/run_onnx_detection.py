import os
import csv
import json
import time
import cv2
import numpy as np
import onnxruntime as ort


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

VIDEO_PATH = os.path.join(
    BASE_DIR, "input", "videos", "PNNL_Parking_LOT(1).avi"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "onnx", "yolo11m.onnx"
)

OUTPUT_VIDEO = os.path.join(
    BASE_DIR, "outputs", "onnx", "videos", "yolo11m_onnx_output.mp4"
)

OUTPUT_CSV = os.path.join(
    BASE_DIR, "outputs", "onnx", "detections",
    "yolo11m_onnx_detections.csv"
)

OUTPUT_JSON = os.path.join(
    BASE_DIR, "outputs", "onnx", "detections",
    "yolo11m_onnx_detections.json"
)

OUTPUT_METRICS = os.path.join(
    BASE_DIR, "outputs", "onnx", "metrics",
    "yolo11m_onnx_metrics.json"
)

OUTPUT_SUMMARY = os.path.join(
    BASE_DIR, "outputs", "onnx", "metrics",
    "yolo11m_onnx_summary.txt"
)

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMAGE_SIZE = 640


# COCO class names
CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_METRICS), exist_ok=True)


# ============================================================
# LETTERBOX PREPROCESSING
# ============================================================

def letterbox(image, new_shape=(640, 640)):
    """
    Resize image while preserving aspect ratio.
    Add padding to obtain 640x640.
    """

    shape = image.shape[:2]  # height, width

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(
        new_shape[0] / shape[0],
        new_shape[1] / shape[1]
    )

    new_unpad = (
        int(round(shape[1] * r)),
        int(round(shape[0] * r))
    )

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(
            image,
            new_unpad,
            interpolation=cv2.INTER_LINEAR
        )

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )

    return image, r, dw, dh


# ============================================================
# IOU FUNCTION
# ============================================================

def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes.
    Boxes: [x1, y1, x2, y2]
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = intersection_width * intersection_height

    area1 = max(0, box1[2] - box1[0]) * max(
        0, box1[3] - box1[1]
    )

    area2 = max(0, box2[2] - box2[0]) * max(
        0, box2[3] - box2[1]
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# CLASS-WISE NMS
# ============================================================

def nms_classwise(boxes, scores, class_ids, iou_threshold):
    """
    Perform class-wise Non-Maximum Suppression.
    """

    keep = []

    unique_classes = np.unique(class_ids)

    for class_id in unique_classes:

        indices = np.where(class_ids == class_id)[0]

        class_boxes = boxes[indices]
        class_scores = scores[indices]

        order = np.argsort(class_scores)[::-1]

        while len(order) > 0:

            current = order[0]

            keep.append(indices[current])

            if len(order) == 1:
                break

            remaining = order[1:]

            current_box = class_boxes[current]

            ious = []

            for idx in remaining:
                ious.append(
                    calculate_iou(
                        current_box,
                        class_boxes[idx]
                    )
                )

            ious = np.array(ious)

            order = remaining[
                ious <= iou_threshold
            ]

    return keep


# ============================================================
# LOAD ONNX MODEL
# ============================================================

print("=" * 70)
print("YOLO11m ONNX DETECTION")
print("=" * 70)

print(f"Model: {MODEL_PATH}")
print(f"Video: {VIDEO_PATH}")

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print(f"ONNX input: {input_name}")
print(f"Providers: {session.get_providers()}")


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )

video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = cap.get(cv2.CAP_PROP_FPS)
expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Resolution: {video_width}x{video_height}")
print(f"FPS: {video_fps}")
print(f"Expected frames: {expected_frames}")


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    video_fps,
    (video_width, video_height)
)

if not writer.isOpened():
    raise RuntimeError(
        f"Could not create output video: {OUTPUT_VIDEO}"
    )


# ============================================================
# STORAGE
# ============================================================

all_detections = []

frames_processed = 0
frames_with_detections = 0
total_detections = 0

total_inference_time = 0.0
total_pipeline_time = 0.0

confidence_values = []

detections_by_class = {}


# ============================================================
# PROCESS VIDEO
# ============================================================

while True:

    pipeline_start = time.perf_counter()

    ret, frame = cap.read()

    if not ret:
        break

    frames_processed += 1

    original_frame = frame.copy()

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    preprocessed, ratio, dw, dh = letterbox(
        frame,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    image = cv2.cvtColor(
        preprocessed,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(np.float32) / 255.0

    image = np.transpose(
        image,
        (2, 0, 1)
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    image = np.ascontiguousarray(image)

    # --------------------------------------------------------
    # ONNX INFERENCE
    # --------------------------------------------------------

    inference_start = time.perf_counter()

    outputs = session.run(
        None,
        {input_name: image}
    )

    inference_end = time.perf_counter()

    inference_time = inference_end - inference_start

    total_inference_time += inference_time

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    predictions = outputs[0]

    # Shape: [1, 84, 8400]
    predictions = predictions[0]

    # Convert to [8400, 84]
    predictions = predictions.T

    boxes = predictions[:, :4]

    class_scores = predictions[:, 4:]

    class_ids = np.argmax(
        class_scores,
        axis=1
    )

    scores = class_scores[
        np.arange(len(class_scores)),
        class_ids
    ]

    # --------------------------------------------------------
    # CONFIDENCE FILTER
    # --------------------------------------------------------

    mask = scores >= CONF_THRESHOLD

    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    # --------------------------------------------------------
    # CONVERT XYWH → XYXY
    # --------------------------------------------------------

    if len(boxes) > 0:

        x_center = boxes[:, 0]
        y_center = boxes[:, 1]
        width = boxes[:, 2]
        height = boxes[:, 3]

        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        boxes = np.column_stack(
            [x1, y1, x2, y2]
        )

        # ----------------------------------------------------
        # NMS
        # ----------------------------------------------------

        keep = nms_classwise(
            boxes,
            scores,
            class_ids,
            IOU_THRESHOLD
        )

        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

    # --------------------------------------------------------
    # SCALE BOXES BACK TO ORIGINAL VIDEO
    # --------------------------------------------------------

    frame_detections = []

    for box, score, class_id in zip(
        boxes,
        scores,
        class_ids
    ):

        x1, y1, x2, y2 = box

        x1 = (x1 - dw) / ratio
        y1 = (y1 - dh) / ratio
        x2 = (x2 - dw) / ratio
        y2 = (y2 - dh) / ratio

        x1 = max(0, min(video_width - 1, x1))
        y1 = max(0, min(video_height - 1, y1))
        x2 = max(0, min(video_width - 1, x2))
        y2 = max(0, min(video_height - 1, y2))

        class_id = int(class_id)
        score = float(score)

        class_name = CLASS_NAMES[class_id]

        width = x2 - x1
        height = y2 - y1

        detection = {
            "frame_number": frames_processed,
            "timestamp_seconds": (
                (frames_processed - 1) / video_fps
                if video_fps > 0 else 0
            ),
            "class_id": class_id,
            "class_name": class_name,
            "confidence": score,
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "width": float(width),
            "height": float(height)
        }

        frame_detections.append(detection)
        all_detections.append(detection)

        confidence_values.append(score)

        detections_by_class[class_name] = (
            detections_by_class.get(class_name, 0) + 1
        )

        # ----------------------------------------------------
        # DRAW DETECTION
        # ----------------------------------------------------

        p1 = (
            int(x1),
            int(y1)
        )

        p2 = (
            int(x2),
            int(y2)
        )

        cv2.rectangle(
            original_frame,
            p1,
            p2,
            (0, 255, 0),
            2
        )

        label = (
            f"{class_name} {score:.2f}"
        )

        cv2.putText(
            original_frame,
            label,
            (int(x1), max(20, int(y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    if len(frame_detections) > 0:
        frames_with_detections += 1

    total_detections += len(frame_detections)

    # --------------------------------------------------------
    # WRITE VIDEO
    # --------------------------------------------------------

    writer.write(original_frame)

    pipeline_end = time.perf_counter()

    total_pipeline_time += (
        pipeline_end - pipeline_start
    )

    if frames_processed % 50 == 0:
        print(
            f"Processed {frames_processed}/{expected_frames} "
            f"| detections={total_detections}"
        )


# ============================================================
# RELEASE
# ============================================================

cap.release()
writer.release()


# ============================================================
# CALCULATE METRICS
# ============================================================

average_inference_time = (
    total_inference_time / frames_processed
    if frames_processed > 0 else 0
)

average_pipeline_time = (
    total_pipeline_time / frames_processed
    if frames_processed > 0 else 0
)

inference_fps = (
    frames_processed / total_inference_time
    if total_inference_time > 0 else 0
)

pipeline_fps = (
    frames_processed / total_pipeline_time
    if total_pipeline_time > 0 else 0
)

average_confidence = (
    float(np.mean(confidence_values))
    if confidence_values else 0
)

minimum_confidence = (
    float(np.min(confidence_values))
    if confidence_values else 0
)

maximum_confidence = (
    float(np.max(confidence_values))
    if confidence_values else 0
)


# ============================================================
# SAVE CSV
# ============================================================

fieldnames = [
    "frame_number",
    "timestamp_seconds",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "width",
    "height"
]

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer_csv = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer_csv.writeheader()

    writer_csv.writerows(all_detections)


# ============================================================
# SAVE JSON
# ============================================================

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_detections,
        f,
        indent=2
    )


# ============================================================
# METRICS JSON
# ============================================================

metrics = {
    "experiment": {
        "model": "YOLO11m",
        "model_format": "ONNX",
        "onnx_runtime": "ONNX Runtime",
        "execution_provider": "CPUExecutionProvider",
        "confidence_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "image_size": IMAGE_SIZE
    },

    "input_video": {
        "filename": os.path.basename(VIDEO_PATH),
        "width": video_width,
        "height": video_height,
        "fps": video_fps,
        "total_frames_expected": expected_frames
    },

    "processing": {
        "frames_processed": frames_processed,
        "frames_with_detections": frames_with_detections,
        "total_detections": total_detections,
        "total_inference_time_seconds": total_inference_time,
        "total_pipeline_time_seconds": total_pipeline_time,
        "average_inference_time_per_frame_seconds":
            average_inference_time,
        "average_inference_time_per_frame_ms":
            average_inference_time * 1000,
        "inference_fps": inference_fps,
        "complete_pipeline_fps": pipeline_fps
    },

    "confidence": {
        "average": average_confidence,
        "minimum": minimum_confidence,
        "maximum": maximum_confidence
    },

    "detections_by_class": detections_by_class,

    "evaluation_metrics": {
        "precision": None,
        "recall": None,
        "f1_score": None,
        "map50": None,
        "map50_95": None,
        "iou": None,
        "reason":
            "Ground-truth annotations are required for these "
            "evaluation metrics."
    }
}


with open(
    OUTPUT_METRICS,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=2
    )


# ============================================================
# SUMMARY TXT
# ============================================================

summary = f"""
YOLO11m ONNX DETECTION SUMMARY
==============================

Model:
YOLO11m ONNX

Input video:
{os.path.basename(VIDEO_PATH)}

Resolution:
{video_width} x {video_height}

Video FPS:
{video_fps}

Expected frames:
{expected_frames}

Frames processed:
{frames_processed}

Frames with detections:
{frames_with_detections}

Total detections:
{total_detections}

Average confidence:
{average_confidence:.6f}

Minimum confidence:
{minimum_confidence:.6f}

Maximum confidence:
{maximum_confidence:.6f}

Total inference time:
{total_inference_time:.4f} seconds

Average inference time/frame:
{average_inference_time * 1000:.4f} ms

Inference FPS:
{inference_fps:.4f}

Complete pipeline FPS:
{pipeline_fps:.4f}

Detection thresholds:
Confidence = {CONF_THRESHOLD}
IoU = {IOU_THRESHOLD}
Image size = {IMAGE_SIZE}

Detections by class:
{json.dumps(detections_by_class, indent=2)}

Evaluation metrics:
Precision = N/A
Recall = N/A
F1-score = N/A
mAP@50 = N/A
mAP@50-95 = N/A
IoU = N/A

Reason:
Ground-truth annotations are required for these
evaluation metrics.
"""


with open(
    OUTPUT_SUMMARY,
    "w",
    encoding="utf-8"
) as f:

    f.write(summary)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("ONNX DETECTION COMPLETED")
print("=" * 70)

print(f"Frames processed       : {frames_processed}")
print(f"Total detections      : {total_detections}")
print(f"Average confidence    : {average_confidence:.4f}")
print(
    f"Average inference     : "
    f"{average_inference_time * 1000:.2f} ms/frame"
)
print(f"Inference FPS         : {inference_fps:.2f}")
print(f"Pipeline FPS          : {pipeline_fps:.2f}")

print()
print("Output files:")
print(OUTPUT_VIDEO)
print(OUTPUT_CSV)
print(OUTPUT_JSON)
print(OUTPUT_METRICS)
print(OUTPUT_SUMMARY)

print("=" * 70)