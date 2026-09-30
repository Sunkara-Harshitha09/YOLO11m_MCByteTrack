import csv
import json
import os
import time
from collections import Counter

import cv2
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "pt", "yolo11m.pt"
)

INPUT_VIDEO = os.path.join(
    BASE_DIR, "input", "videos", "PNNL_Parking_LOT(1).avi"
)

OUTPUT_VIDEO = os.path.join(
    BASE_DIR, "outputs", "pt", "videos", "yolo11m_pt_output.mp4"
)

OUTPUT_CSV = os.path.join(
    BASE_DIR, "outputs", "pt", "detections", "yolo11m_pt_detections.csv"
)

OUTPUT_JSON = os.path.join(
    BASE_DIR, "outputs", "pt", "detections", "yolo11m_pt_detections.json"
)

OUTPUT_METRICS_JSON = os.path.join(
    BASE_DIR, "outputs", "pt", "metrics", "yolo11m_pt_metrics.json"
)

OUTPUT_METRICS_TXT = os.path.join(
    BASE_DIR, "outputs", "pt", "metrics", "yolo11m_pt_summary.txt"
)


# Detection settings
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMAGE_SIZE = 640
DEVICE = "cpu"


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_METRICS_JSON), exist_ok=True)


# ============================================================
# VERIFY INPUTS
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"YOLO11m model not found:\n{MODEL_PATH}"
    )

if not os.path.exists(INPUT_VIDEO):
    raise FileNotFoundError(
        f"Input video not found:\n{INPUT_VIDEO}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("YOLO11m PT DETECTION")
print("=" * 70)

print(f"Model : {MODEL_PATH}")
print(f"Video : {INPUT_VIDEO}")
print(f"Device: {DEVICE}")
print(f"Confidence threshold: {CONF_THRESHOLD}")
print(f"IoU threshold       : {IOU_THRESHOLD}")
print(f"Image size          : {IMAGE_SIZE}")
print()


model = YOLO(MODEL_PATH)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise RuntimeError("Could not open input video.")


fps_input = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames_expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc_input = int(cap.get(cv2.CAP_PROP_FOURCC))

codec_input = "".join(
    chr((fourcc_input >> (8 * i)) & 0xFF)
    for i in range(4)
)


# ============================================================
# OUTPUT VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps_input,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Could not create output video.")


# ============================================================
# METRIC VARIABLES
# ============================================================

frame_count = 0
frames_with_detections = 0
total_detections = 0

confidence_values = []

class_counter = Counter()

detection_rows = []

inference_times = []


# ============================================================
# PROCESS VIDEO
# ============================================================

experiment_start = time.perf_counter()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # --------------------------------------------------------
    # YOLO INFERENCE
    # --------------------------------------------------------

    inference_start = time.perf_counter()

    results = model.predict(
        source=frame,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        verbose=False
    )

    inference_end = time.perf_counter()

    inference_time = inference_end - inference_start

    inference_times.append(inference_time)

    result = results[0]

    frame_detection_count = 0

    # --------------------------------------------------------
    # PROCESS DETECTIONS
    # --------------------------------------------------------

    if result.boxes is not None:

        boxes = result.boxes

        for i in range(len(boxes)):

            xyxy = boxes.xyxy[i].cpu().numpy()

            confidence = float(
                boxes.conf[i].cpu().item()
            )

            class_id = int(
                boxes.cls[i].cpu().item()
            )

            class_name = model.names[class_id]

            x1, y1, x2, y2 = [
                float(value) for value in xyxy
            ]

            frame_detection_count += 1
            total_detections += 1

            confidence_values.append(confidence)

            class_counter[class_name] += 1

            detection_rows.append(
                {
                    "frame_number": frame_count,
                    "timestamp_seconds": (
                        (frame_count - 1) / fps_input
                        if fps_input > 0
                        else None
                    ),
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "width": x2 - x1,
                    "height": y2 - y1,
                }
            )

    if frame_detection_count > 0:
        frames_with_detections += 1

    # --------------------------------------------------------
    # DRAW DETECTIONS
    # --------------------------------------------------------

    annotated_frame = result.plot()

    writer.write(annotated_frame)

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if frame_count % 50 == 0:

        print(
            f"Processed frames: "
            f"{frame_count}/{total_frames_expected}"
        )


# ============================================================
# FINISH
# ============================================================

experiment_end = time.perf_counter()

total_processing_time = (
    experiment_end - experiment_start
)

cap.release()
writer.release()


# ============================================================
# CALCULATE METRICS
# ============================================================

average_inference_time = (
    sum(inference_times) / len(inference_times)
    if inference_times
    else 0
)

inference_fps = (
    1 / average_inference_time
    if average_inference_time > 0
    else 0
)

pipeline_fps = (
    frame_count / total_processing_time
    if total_processing_time > 0
    else 0
)

average_confidence = (
    sum(confidence_values) / len(confidence_values)
    if confidence_values
    else 0
)

minimum_confidence = (
    min(confidence_values)
    if confidence_values
    else 0
)

maximum_confidence = (
    max(confidence_values)
    if confidence_values
    else 0
)


# ============================================================
# METRICS OBJECT
# ============================================================

metrics = {

    "experiment": {
        "model": "YOLO11m",
        "model_format": "PyTorch (.pt)",
        "device": DEVICE,
        "confidence_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "image_size": IMAGE_SIZE
    },

    "input_video": {
        "filename": os.path.basename(INPUT_VIDEO),
        "width": width,
        "height": height,
        "fps": fps_input,
        "total_frames_expected": total_frames_expected,
        "codec": codec_input
    },

    "processing": {
        "frames_processed": frame_count,
        "frames_with_detections": frames_with_detections,
        "total_detections": total_detections,
        "total_processing_time_seconds": total_processing_time,
        "average_inference_time_per_frame_seconds": average_inference_time,
        "average_inference_time_per_frame_ms": (
            average_inference_time * 1000
        ),
        "inference_fps": inference_fps,
        "complete_pipeline_fps": pipeline_fps
    },

    "confidence": {
        "average": average_confidence,
        "minimum": minimum_confidence,
        "maximum": maximum_confidence
    },

    "detections_by_class": dict(class_counter),

    "evaluation_metrics": {
        "precision": None,
        "recall": None,
        "f1_score": None,
        "map50": None,
        "map50_95": None,
        "iou": None,
        "reason": (
            "Ground-truth annotations are required "
            "for these evaluation metrics."
        )
    }
}


# ============================================================
# SAVE CSV
# ============================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as file:

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

    writer_csv = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer_csv.writeheader()
    writer_csv.writerows(detection_rows)


# ============================================================
# SAVE DETECTIONS JSON
# ============================================================

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        detection_rows,
        file,
        indent=2
    )


# ============================================================
# SAVE METRICS JSON
# ============================================================

with open(
    OUTPUT_METRICS_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=2
    )


# ============================================================
# SAVE HUMAN-READABLE SUMMARY
# ============================================================

with open(
    OUTPUT_METRICS_TXT,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "YOLO11m PT DETECTION SUMMARY\n"
    )

    file.write("=" * 60 + "\n\n")

    file.write(
        f"Model: YOLO11m\n"
        f"Format: PyTorch (.pt)\n"
        f"Device: {DEVICE}\n\n"
    )

    file.write(
        "INPUT VIDEO\n"
    )

    file.write("-" * 60 + "\n")

    file.write(
        f"Filename: {os.path.basename(INPUT_VIDEO)}\n"
        f"Resolution: {width}x{height}\n"
        f"FPS: {fps_input}\n"
        f"Expected frames: {total_frames_expected}\n"
        f"Codec: {codec_input}\n\n"
    )

    file.write(
        "DETECTION RESULTS\n"
    )

    file.write("-" * 60 + "\n")

    file.write(
        f"Frames processed: {frame_count}\n"
        f"Frames with detections: {frames_with_detections}\n"
        f"Total detections: {total_detections}\n"
        f"Average confidence: {average_confidence:.6f}\n"
        f"Minimum confidence: {minimum_confidence:.6f}\n"
        f"Maximum confidence: {maximum_confidence:.6f}\n\n"
    )

    file.write(
        "PERFORMANCE\n"
    )

    file.write("-" * 60 + "\n")

    file.write(
        f"Total processing time: "
        f"{total_processing_time:.6f} seconds\n"
        f"Average inference time/frame: "
        f"{average_inference_time * 1000:.3f} ms\n"
        f"Inference FPS: {inference_fps:.3f}\n"
        f"Complete pipeline FPS: {pipeline_fps:.3f}\n\n"
    )

    file.write(
        "DETECTIONS BY CLASS\n"
    )

    file.write("-" * 60 + "\n")

    for class_name, count in class_counter.items():

        file.write(
            f"{class_name}: {count}\n"
        )

    file.write("\n")

    file.write(
        "EVALUATION METRICS\n"
    )

    file.write("-" * 60 + "\n")

    file.write(
        "Precision: Not calculated\n"
        "Recall: Not calculated\n"
        "F1-score: Not calculated\n"
        "mAP@50: Not calculated\n"
        "mAP@50-95: Not calculated\n"
        "IoU: Not calculated\n\n"
    )

    file.write(
        "Reason: Ground-truth annotations are required "
        "for these metrics.\n"
    )


# ============================================================
# FINAL CONSOLE OUTPUT
# ============================================================

print()
print("=" * 70)
print("PT INFERENCE COMPLETE")
print("=" * 70)

print(f"Frames processed       : {frame_count}")
print(f"Total detections       : {total_detections}")
print(f"Average confidence     : {average_confidence:.4f}")
print(
    f"Avg inference/frame    : "
    f"{average_inference_time * 1000:.2f} ms"
)
print(f"Inference FPS          : {inference_fps:.2f}")
print(f"Complete pipeline FPS  : {pipeline_fps:.2f}")
print(
    f"Total processing time  : "
    f"{total_processing_time:.2f} sec"
)

print()
print("OUTPUT FILES")
print("-" * 70)
print(f"Video   : {OUTPUT_VIDEO}")
print(f"CSV     : {OUTPUT_CSV}")
print(f"JSON    : {OUTPUT_JSON}")
print(f"Metrics : {OUTPUT_METRICS_JSON}")
print(f"Summary : {OUTPUT_METRICS_TXT}")
print("=" * 70)