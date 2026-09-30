from pathlib import Path
import csv
import json
import time

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_PATH = PROJECT_ROOT / "input" / "videos" / "PNNL_Parking_LOT(1).avi"
MODEL_PATH = PROJECT_ROOT / "models" / "pt" / "yolo11m.pt"

OUTPUT_VIDEO = (
    PROJECT_ROOT
    / "outputs"
    / "mcbytetrack"
    / "videos"
    / "yolo11m_mcbytetrack_output.mp4"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "mcbytetrack"
    / "tracking"
    / "mcbytetrack_tracks.csv"
)

OUTPUT_JSON = (
    PROJECT_ROOT
    / "outputs"
    / "mcbytetrack"
    / "tracking"
    / "mcbytetrack_tracks.json"
)

OUTPUT_METRICS = (
    PROJECT_ROOT
    / "outputs"
    / "mcbytetrack"
    / "metrics"
    / "mcbytetrack_metrics.json"
)

OUTPUT_SUMMARY = (
    PROJECT_ROOT
    / "outputs"
    / "mcbytetrack"
    / "metrics"
    / "mcbytetrack_summary.txt"
)


# ============================================================
# SETTINGS
# ============================================================

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMAGE_SIZE = 640
DEVICE = "cpu"


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_METRICS.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# VALIDATE INPUTS
# ============================================================

if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("MCByteTrack - YOLO11m Tracking")
print("=" * 70)

print(f"Video : {VIDEO_PATH}")
print(f"Model : {MODEL_PATH}")
print()

print("Loading YOLO11m...")
model = YOLO(str(MODEL_PATH))

print("YOLO11m loaded successfully.")
print(f"Classes: {len(model.names)}")
print()


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Could not open input video.")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
input_fps = float(cap.get(cv2.CAP_PROP_FPS))
expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    input_fps,
    (frame_width, frame_height),
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Could not create output video.")


# ============================================================
# INITIALIZE BYTETRACK
# ============================================================

tracker = sv.ByteTrack()


# ============================================================
# STORAGE
# ============================================================

tracking_records = []

unique_track_ids = set()

frames_with_tracks = 0
total_detections = 0

inference_times = []
frame_processing_times = []

track_lengths = {}


# ============================================================
# TRACKING LOOP
# ============================================================

frame_number = 0

total_start_time = time.perf_counter()

print("Starting YOLO11m + ByteTrack processing...")
print(f"Expected frames: {expected_frames}")
print()

while True:

    frame_start_time = time.perf_counter()

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # --------------------------------------------------------
    # YOLO INFERENCE
    # --------------------------------------------------------

    inference_start = time.perf_counter()

    results = model.predict(
        source=frame,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        verbose=False,
    )

    inference_time = time.perf_counter() - inference_start
    inference_times.append(inference_time)

    result = results[0]

    # --------------------------------------------------------
    # CONVERT YOLO DETECTIONS TO SUPERVISION
    # --------------------------------------------------------

    detections = sv.Detections.from_ultralytics(result)

    total_detections += len(detections)

    # --------------------------------------------------------
    # BYTE TRACKING
    # --------------------------------------------------------

    tracked_detections = tracker.update_with_detections(detections)

    # --------------------------------------------------------
    # DRAW TRACKING RESULTS
    # --------------------------------------------------------

    annotated_frame = frame.copy()

    if len(tracked_detections) > 0:

        frames_with_tracks += 1

        tracker_ids = tracked_detections.tracker_id

        for i in range(len(tracked_detections)):

            tracker_id = int(tracker_ids[i])

            unique_track_ids.add(tracker_id)

            class_id = int(tracked_detections.class_id[i])

            confidence = float(tracked_detections.confidence[i])

            x1, y1, x2, y2 = (
                tracked_detections.xyxy[i].astype(int)
            )

            class_name = model.names[class_id]

            # Track length
            track_lengths[tracker_id] = (
                track_lengths.get(tracker_id, 0) + 1
            )

            # Save tracking record
            tracking_records.append(
                {
                    "frame_number": frame_number,
                    "timestamp_seconds": round(
                        (frame_number - 1) / input_fps,
                        6,
                    ),
                    "track_id": tracker_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "width": int(x2 - x1),
                    "height": int(y2 - y1),
                }
            )

            # Draw bounding box
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            # Draw label
            label = f"ID {tracker_id} | {class_name} | {confidence:.2f}"

            cv2.putText(
                annotated_frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    # --------------------------------------------------------
    # FRAME PROCESSING TIME
    # --------------------------------------------------------

    frame_processing_time = (
        time.perf_counter() - frame_start_time
    )

    frame_processing_times.append(frame_processing_time)

    # --------------------------------------------------------
    # WRITE FRAME
    # --------------------------------------------------------

    writer.write(annotated_frame)

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if frame_number % 100 == 0:

        elapsed = time.perf_counter() - total_start_time

        current_fps = frame_number / elapsed

        print(
            f"Processed {frame_number}/{expected_frames} "
            f"frames | "
            f"Tracks: {len(unique_track_ids)} | "
            f"FPS: {current_fps:.2f}"
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()
writer.release()

total_processing_time = time.perf_counter() - total_start_time


# ============================================================
# CALCULATE METRICS
# ============================================================

frames_processed = frame_number

average_inference_time = (
    sum(inference_times) / len(inference_times)
    if inference_times
    else 0
)

average_frame_processing_time = (
    sum(frame_processing_times)
    / len(frame_processing_times)
    if frame_processing_times
    else 0
)

average_inference_time_ms = (
    average_inference_time * 1000
)

average_frame_processing_time_ms = (
    average_frame_processing_time * 1000
)

inference_fps = (
    1 / average_inference_time
    if average_inference_time > 0
    else 0
)

pipeline_fps = (
    1 / average_frame_processing_time
    if average_frame_processing_time > 0
    else 0
)

average_confidence = (
    sum(record["confidence"] for record in tracking_records)
    / len(tracking_records)
    if tracking_records
    else 0
)

track_length_values = list(track_lengths.values())

average_track_length = (
    sum(track_length_values) / len(track_length_values)
    if track_length_values
    else 0
)

minimum_track_length = (
    min(track_length_values)
    if track_length_values
    else 0
)

maximum_track_length = (
    max(track_length_values)
    if track_length_values
    else 0
)

average_tracks_per_frame = (
    len(tracking_records) / frames_processed
    if frames_processed > 0
    else 0
)


# ============================================================
# TRACKS BY CLASS
# ============================================================

tracks_by_class = {}

for record in tracking_records:

    class_name = record["class_name"]

    tracks_by_class.setdefault(class_name, set())

    tracks_by_class[class_name].add(
        record["track_id"]
    )

tracks_by_class = {
    class_name: len(track_ids)
    for class_name, track_ids in tracks_by_class.items()
}


# ============================================================
# METRICS JSON
# ============================================================

metrics = {
    "experiment": {
        "tracker": "ByteTrack",
        "model": "YOLO11m",
        "model_format": "PyTorch (.pt)",
        "device": DEVICE,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "image_size": IMAGE_SIZE,
    },

    "input_video": {
        "filename": VIDEO_PATH.name,
        "width": frame_width,
        "height": frame_height,
        "fps": input_fps,
        "total_frames_expected": expected_frames,
    },

    "processing": {
        "frames_processed": frames_processed,
        "frames_with_tracks": frames_with_tracks,
        "total_detections": total_detections,
        "total_tracking_records": len(tracking_records),
        "total_processing_time_seconds": total_processing_time,
        "average_inference_time_per_frame_seconds": average_inference_time,
        "average_inference_time_per_frame_ms": average_inference_time_ms,
        "average_frame_processing_time_ms": average_frame_processing_time_ms,
        "inference_fps": inference_fps,
        "complete_pipeline_fps": pipeline_fps,
    },

    "tracking": {
        "total_unique_track_ids": len(unique_track_ids),
        "average_tracks_per_frame": average_tracks_per_frame,
        "average_track_length_frames": average_track_length,
        "minimum_track_length_frames": minimum_track_length,
        "maximum_track_length_frames": maximum_track_length,
        "tracks_by_class": tracks_by_class,
    },

    "confidence": {
        "average": average_confidence,
    },

    "evaluation_metrics": {
        "precision": None,
        "recall": None,
        "f1_score": None,
        "map50": None,
        "map50_95": None,
        "iou": None,
        "mota": None,
        "motp": None,
        "idf1": None,
        "hota": None,
        "reason": (
            "Ground-truth detection and tracking annotations "
            "are required for these evaluation metrics."
        ),
    },
}


# ============================================================
# SAVE JSON
# ============================================================

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        tracking_records,
        f,
        indent=2,
    )


with open(
    OUTPUT_METRICS,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metrics,
        f,
        indent=2,
    )


# ============================================================
# SAVE CSV
# ============================================================

if tracking_records:

    fieldnames = list(tracking_records[0].keys())

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer_csv = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer_csv.writeheader()

        writer_csv.writerows(tracking_records)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_lines = [
    "MCByteTrack / YOLO11m Tracking Summary",
    "=" * 60,
    "",
    f"Video: {VIDEO_PATH.name}",
    f"Model: YOLO11m",
    f"Tracker: ByteTrack",
    f"Device: {DEVICE}",
    "",
    f"Frames processed: {frames_processed}",
    f"Frames with tracks: {frames_with_tracks}",
    f"Total detections: {total_detections}",
    f"Total tracking records: {len(tracking_records)}",
    "",
    f"Unique track IDs: {len(unique_track_ids)}",
    f"Average tracks/frame: {average_tracks_per_frame:.4f}",
    f"Average track length: {average_track_length:.2f} frames",
    f"Minimum track length: {minimum_track_length} frames",
    f"Maximum track length: {maximum_track_length} frames",
    "",
    f"Average confidence: {average_confidence:.6f}",
    "",
    f"Total processing time: {total_processing_time:.4f} seconds",
    f"Average inference time/frame: {average_inference_time_ms:.4f} ms",
    f"Inference FPS: {inference_fps:.4f}",
    f"Complete pipeline FPS: {pipeline_fps:.4f}",
    "",
    "Ground-truth dependent metrics:",
    "Precision: Ground truth required",
    "Recall: Ground truth required",
    "F1-score: Ground truth required",
    "mAP@50: Ground truth required",
    "mAP@50-95: Ground truth required",
    "IoU: Ground truth required",
    "MOTA: Ground truth required",
    "MOTP: Ground truth required",
    "IDF1: Ground truth required",
    "HOTA: Ground truth required",
]


with open(
    OUTPUT_SUMMARY,
    "w",
    encoding="utf-8",
) as f:

    f.write("\n".join(summary_lines))


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("MCByteTrack processing completed successfully.")
print("=" * 70)

print(f"Frames processed       : {frames_processed}")
print(f"Total detections       : {total_detections}")
print(f"Tracking records       : {len(tracking_records)}")
print(f"Unique track IDs       : {len(unique_track_ids)}")
print(f"Average track length   : {average_track_length:.2f} frames")
print(f"Average confidence     : {average_confidence:.4f}")
print(f"Total processing time  : {total_processing_time:.2f} sec")
print(f"Average inference time : {average_inference_time_ms:.2f} ms/frame")
print(f"Inference FPS          : {inference_fps:.2f}")
print(f"Pipeline FPS           : {pipeline_fps:.2f}")
print()
print(f"Video output : {OUTPUT_VIDEO}")
print(f"CSV output   : {OUTPUT_CSV}")
print(f"JSON output  : {OUTPUT_JSON}")
print(f"Metrics      : {OUTPUT_METRICS}")
print(f"Summary      : {OUTPUT_SUMMARY}")