import csv
import json
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_DIR / "models" / "tensorrt" / "yolo11m.engine"
VIDEO_PATH = PROJECT_DIR / "input" / "videos" / "PNNL_Parking_LOT(1).avi"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "tensorrt"

VIDEO_OUTPUT_DIR = OUTPUT_DIR / "videos"
DETECTION_OUTPUT_DIR = OUTPUT_DIR / "detections"
METRICS_OUTPUT_DIR = OUTPUT_DIR / "metrics"

ANNOTATED_VIDEO_PATH = VIDEO_OUTPUT_DIR / "yolo11m_tensorrt_annotated.mp4"
DETECTIONS_CSV_PATH = DETECTION_OUTPUT_DIR / "yolo11m_tensorrt_detections.csv"
METRICS_JSON_PATH = METRICS_OUTPUT_DIR / "yolo11m_tensorrt_metrics.json"

IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
DEVICE = 0


# ============================================================
# 2. CREATE OUTPUT DIRECTORIES
# ============================================================

for directory in [
    VIDEO_OUTPUT_DIR,
    DETECTION_OUTPUT_DIR,
    METRICS_OUTPUT_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. CHECK FILES AND GPU
# ============================================================

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"TensorRT engine not found: {MODEL_PATH}")

if not VIDEO_PATH.is_file():
    raise FileNotFoundError(f"Input video not found: {VIDEO_PATH}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. This script requires the GPU.")

print("=" * 65)
print("YOLO11m TENSORRT GPU INFERENCE")
print("=" * 65)
print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")
print(f"TensorRT engine: {MODEL_PATH}")
print(f"Input video: {VIDEO_PATH}")
print(f"Output directory: {OUTPUT_DIR}")


# ============================================================
# 4. LOAD TENSORRT ENGINE
# ============================================================

print("\nLoading TensorRT engine...")

model = YOLO(str(MODEL_PATH), task="detect")

print("TensorRT engine loaded.")


# ============================================================
# 5. OPEN INPUT VIDEO
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
input_fps = cap.get(cv2.CAP_PROP_FPS)
expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if input_fps <= 0:
    input_fps = 30.0

print("\nInput video properties:")
print(f"Resolution: {width} x {height}")
print(f"Input FPS: {input_fps:.2f}")
print(f"Expected frames: {expected_frames}")


# ============================================================
# 6. INITIALIZE VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video_writer = cv2.VideoWriter(
    str(ANNOTATED_VIDEO_PATH),
    fourcc,
    input_fps,
    (width, height),
)

if not video_writer.isOpened():
    cap.release()
    raise RuntimeError(
        f"Could not create output video: {ANNOTATED_VIDEO_PATH}"
    )


# ============================================================
# 7. RUN TENSORRT INFERENCE
# ============================================================

frame_count = 0
total_detections = 0
total_inference_seconds = 0.0
total_processing_seconds = 0.0

class_counts = {}

print("\nStarting inference...")
print("Progress will be displayed every 100 frames.\n")

with open(DETECTIONS_CSV_PATH, "w", newline="", encoding="utf-8") as csv_file:

    csv_writer = csv.writer(csv_file)

    csv_writer.writerow([
        "frame_number",
        "timestamp_seconds",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    ])

    try:
        while True:

            frame_start = time.perf_counter()

            success, frame = cap.read()

            if not success:
                break

            frame_count += 1

            # Synchronize GPU before timing inference.
            torch.cuda.synchronize()
            inference_start = time.perf_counter()

            results = model.predict(
                source=frame,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                device=DEVICE,
                verbose=False,
            )

            torch.cuda.synchronize()
            inference_end = time.perf_counter()

            inference_seconds = inference_end - inference_start

            total_inference_seconds += inference_seconds

            result = results[0]
            boxes = result.boxes

            frame_detections = 0

            if boxes is not None and len(boxes) > 0:

                coordinates = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.int().cpu().numpy()

                for box, confidence, class_id in zip(
                    coordinates,
                    confidences,
                    class_ids,
                ):

                    x1, y1, x2, y2 = box.tolist()
                    class_id = int(class_id)
                    confidence = float(confidence)

                    class_name = model.names.get(
                        class_id,
                        str(class_id),
                    )

                    timestamp_seconds = (
                        (frame_count - 1) / input_fps
                    )

                    csv_writer.writerow([
                        frame_count,
                        round(timestamp_seconds, 4),
                        class_id,
                        class_name,
                        round(confidence, 6),
                        round(x1, 2),
                        round(y1, 2),
                        round(x2, 2),
                        round(y2, 2),
                    ])

                    frame_detections += 1

                    class_counts[class_name] = (
                        class_counts.get(class_name, 0) + 1
                    )

            total_detections += frame_detections

            # Draw detection boxes and labels.
            annotated_frame = result.plot()

            video_writer.write(annotated_frame)

            frame_processing_seconds = (
                time.perf_counter() - frame_start
            )

            total_processing_seconds += frame_processing_seconds

            if frame_count % 100 == 0:
                print(
                    f"Processed {frame_count}/{expected_frames} frames | "
                    f"Detections: {total_detections} | "
                    f"Last inference: {inference_seconds * 1000:.2f} ms"
                )

    finally:
        cap.release()
        video_writer.release()


# ============================================================
# 8. CALCULATE PERFORMANCE METRICS
# ============================================================

average_inference_ms = (
    total_inference_seconds / frame_count * 1000
    if frame_count else 0.0
)

average_processing_ms = (
    total_processing_seconds / frame_count * 1000
    if frame_count else 0.0
)

inference_fps = (
    frame_count / total_inference_seconds
    if total_inference_seconds > 0 else 0.0
)

end_to_end_fps = (
    frame_count / total_processing_seconds
    if total_processing_seconds > 0 else 0.0
)

metrics = {
    "model": "YOLO11m TensorRT FP16",
    "device": torch.cuda.get_device_name(DEVICE),
    "input_video": str(VIDEO_PATH),
    "engine_path": str(MODEL_PATH),
    "image_size": IMAGE_SIZE,
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "iou_threshold": IOU_THRESHOLD,
    "video_width": width,
    "video_height": height,
    "input_fps": input_fps,
    "expected_frames": expected_frames,
    "processed_frames": frame_count,
    "total_detections": total_detections,
    "detections_by_class": class_counts,
    "total_inference_seconds": round(
        total_inference_seconds, 4
    ),
    "average_inference_ms_per_frame": round(
        average_inference_ms, 4
    ),
    "inference_fps": round(inference_fps, 4),
    "total_processing_seconds": round(
        total_processing_seconds, 4
    ),
    "average_processing_ms_per_frame": round(
        average_processing_ms, 4
    ),
    "end_to_end_fps": round(end_to_end_fps, 4),
}


# ============================================================
# 9. SAVE METRICS
# ============================================================

with open(
    METRICS_JSON_PATH,
    "w",
    encoding="utf-8",
) as metrics_file:

    json.dump(metrics, metrics_file, indent=4)


# ============================================================
# 10. PRINT FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("TENSORRT INFERENCE COMPLETED")
print("=" * 65)

print(f"GPU: {metrics['device']}")
print(f"Frames processed: {frame_count}/{expected_frames}")
print(f"Total detections: {total_detections}")
print(f"Average inference time: {average_inference_ms:.2f} ms/frame")
print(f"Inference FPS: {inference_fps:.2f}")
print(f"End-to-end FPS: {end_to_end_fps:.2f}")

print("\nDetections by class:")

for class_name, count in sorted(class_counts.items()):
    print(f"  {class_name}: {count}")

print("\nGenerated files:")
print(f"Annotated video: {ANNOTATED_VIDEO_PATH}")
print(f"Detection CSV: {DETECTIONS_CSV_PATH}")
print(f"Metrics JSON: {METRICS_JSON_PATH}")

print("\nAll TensorRT outputs are inside outputs/tensorrt/")