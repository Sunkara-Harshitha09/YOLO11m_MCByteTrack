from pathlib import Path
import csv
import json
import time
import logging
import statistics

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VIDEO_PATH = (
    PROJECT_ROOT
    / "input"
    / "videos"
    / "PNNL_Parking_LOT(1).avi"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "pt"
    / "yolo11m.pt"
)

# ------------------------------------------------------------
# Existing outputs
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# NEW JSON - SAME STRUCTURE AS BoT-SORT REFERENCE
# ------------------------------------------------------------

OUTPUT_TRACKING_METRICS = (
    PROJECT_ROOT
    / "outputs"
    / "mcbytetrack"
    / "metrics"
    / "mcbytetrack_tracking_metrics.json"
)

# ------------------------------------------------------------
# Log
# ------------------------------------------------------------

OUTPUT_LOG = (
    PROJECT_ROOT
    / "outputs"
    / "mcbytetrack"
    / "logs"
    / "mcbytetrack.log"
)


# ============================================================
# SETTINGS
# ============================================================

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMAGE_SIZE = 640
DEVICE = "cpu"

TRACKER_NAME = "MCByteTrack"
TRACKER_CONFIG = "supervision.ByteTrack"


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_TRACKING_METRICS.parent.mkdir(
    parents=True,
    exist_ok=True
)
OUTPUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_LOG.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    filename=str(OUTPUT_LOG),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("MCByteTrack")


# ============================================================
# VALIDATE INPUTS
# ============================================================

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


# ============================================================
# RESOURCE MONITORING
# ============================================================

process = None

process_cpu_samples = []
process_ram_samples = []
system_cpu_samples = []

if PSUTIL_AVAILABLE:
    process = psutil.Process()


def collect_resource_usage():
    """
    Collect process and system CPU/RAM information.
    """

    if not PSUTIL_AVAILABLE:
        return

    try:
        process_cpu = process.cpu_percent(interval=None)

        process_ram = (
            process.memory_info().rss
            / (1024 * 1024)
        )

        system_cpu = psutil.cpu_percent(
            interval=None
        )

        process_cpu_samples.append(
            process_cpu
        )

        process_ram_samples.append(
            process_ram
        )

        system_cpu_samples.append(
            system_cpu
        )

    except Exception:
        pass


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("YOLO11m + MCByteTrack")
print("=" * 70)

print(f"Video : {VIDEO_PATH}")
print(f"Model : {MODEL_PATH}")
print()

logger.info(
    "Starting YOLO11m + MCByteTrack"
)

logger.info(
    f"Video: {VIDEO_PATH}"
)

logger.info(
    f"Model: {MODEL_PATH}"
)

print("Loading YOLO11m...")

model = YOLO(str(MODEL_PATH))

print("YOLO11m loaded successfully.")
print(f"Classes: {len(model.names)}")
print()

logger.info(
    "YOLO11m loaded successfully"
)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    str(VIDEO_PATH)
)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open input video."
    )

frame_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

frame_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

input_fps = float(
    cap.get(cv2.CAP_PROP_FPS)
)

expected_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    input_fps,
    (
        frame_width,
        frame_height
    ),
)

if not writer.isOpened():
    cap.release()

    raise RuntimeError(
        "Could not create output video."
    )


# ============================================================
# INITIALIZE TRACKER
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

active_tracks_per_frame = []


# ============================================================
# TRACKING LOOP
# ============================================================

frame_number = 0

total_start_time = time.perf_counter()

print(
    "Starting YOLO11m + MCByteTrack processing..."
)

print(
    f"Expected frames: {expected_frames}"
)

print()

while True:

    frame_start_time = (
        time.perf_counter()
    )

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # --------------------------------------------------------
    # YOLO INFERENCE
    # --------------------------------------------------------

    inference_start = (
        time.perf_counter()
    )

    results = model.predict(
        source=frame,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        verbose=False,
    )

    inference_time = (
        time.perf_counter()
        - inference_start
    )

    inference_times.append(
        inference_time
    )

    result = results[0]

    # --------------------------------------------------------
    # YOLO -> SUPERVISION
    # --------------------------------------------------------

    detections = (
        sv.Detections.from_ultralytics(
            result
        )
    )

    total_detections += len(
        detections
    )

    # --------------------------------------------------------
    # TRACKING
    # --------------------------------------------------------

    tracked_detections = (
        tracker.update_with_detections(
            detections
        )
    )

    # --------------------------------------------------------
    # ANNOTATION
    # --------------------------------------------------------

    annotated_frame = frame.copy()

    current_frame_track_count = 0

    if len(tracked_detections) > 0:

        frames_with_tracks += 1

        tracker_ids = (
            tracked_detections.tracker_id
        )

        current_frame_track_count = (
            len(tracked_detections)
        )

        for i in range(
            len(tracked_detections)
        ):

            tracker_id = int(
                tracker_ids[i]
            )

            unique_track_ids.add(
                tracker_id
            )

            class_id = int(
                tracked_detections.class_id[i]
            )

            confidence = float(
                tracked_detections.confidence[i]
            )

            x1, y1, x2, y2 = (
                tracked_detections
                .xyxy[i]
                .astype(int)
            )

            class_name = model.names[
                class_id
            ]

            # Track lifetime
            track_lengths[
                tracker_id
            ] = (
                track_lengths.get(
                    tracker_id,
                    0
                ) + 1
            )

            # Tracking record
            tracking_records.append(
                {
                    "frame_number":
                        frame_number,

                    "timestamp_seconds":
                        round(
                            (
                                frame_number - 1
                            )
                            / input_fps,
                            6,
                        ),

                    "track_id":
                        tracker_id,

                    "class_id":
                        class_id,

                    "class_name":
                        class_name,

                    "confidence":
                        confidence,

                    "x1":
                        int(x1),

                    "y1":
                        int(y1),

                    "x2":
                        int(x2),

                    "y2":
                        int(y2),

                    "width":
                        int(x2 - x1),

                    "height":
                        int(y2 - y1),
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
            label = (
                f"ID {tracker_id} | "
                f"{class_name} | "
                f"{confidence:.2f}"
            )

            cv2.putText(
                annotated_frame,
                label,
                (
                    x1,
                    max(y1 - 10, 20),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    active_tracks_per_frame.append(
        current_frame_track_count
    )

    # --------------------------------------------------------
    # FRAME PROCESSING TIME
    # --------------------------------------------------------

    frame_processing_time = (
        time.perf_counter()
        - frame_start_time
    )

    frame_processing_times.append(
        frame_processing_time
    )

    # --------------------------------------------------------
    # RESOURCE MONITORING
    # --------------------------------------------------------

    collect_resource_usage()

    # --------------------------------------------------------
    # WRITE FRAME
    # --------------------------------------------------------

    writer.write(
        annotated_frame
    )

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if frame_number % 100 == 0:

        elapsed = (
            time.perf_counter()
            - total_start_time
        )

        current_fps = (
            frame_number / elapsed
            if elapsed > 0
            else 0
        )

        print(
            f"Processed "
            f"{frame_number}/"
            f"{expected_frames} "
            f"frames | "
            f"Tracks: "
            f"{len(unique_track_ids)} | "
            f"FPS: "
            f"{current_fps:.2f}"
        )

        logger.info(
            f"Processed "
            f"{frame_number}/"
            f"{expected_frames} "
            f"frames | "
            f"Tracks: "
            f"{len(unique_track_ids)} | "
            f"FPS: "
            f"{current_fps:.2f}"
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

writer.release()

total_processing_time = (
    time.perf_counter()
    - total_start_time
)


# ============================================================
# BASIC METRICS
# ============================================================

frames_processed = frame_number

average_inference_time = (
    sum(inference_times)
    / len(inference_times)
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

pipeline_fps = (
    1 / average_frame_processing_time
    if average_frame_processing_time > 0
    else 0
)


# ============================================================
# TRACKING STATISTICS
# ============================================================

average_active_tracks = (
    sum(active_tracks_per_frame)
    / len(active_tracks_per_frame)
    if active_tracks_per_frame
    else 0
)

maximum_active_tracks = (
    max(active_tracks_per_frame)
    if active_tracks_per_frame
    else 0
)

track_length_values = list(
    track_lengths.values()
)

average_track_lifetime = (
    sum(track_length_values)
    / len(track_length_values)
    if track_length_values
    else 0
)

longest_track_lifetime = (
    max(track_length_values)
    if track_length_values
    else 0
)


# ============================================================
# P95 LATENCY
# ============================================================

latencies_ms = [
    value * 1000
    for value in frame_processing_times
]

if latencies_ms:

    p95_latency_ms = float(
        np.percentile(
            latencies_ms,
            95
        )
    )

else:

    p95_latency_ms = 0.0


# ============================================================
# RESOURCE METRICS
# ============================================================

if process_cpu_samples:

    average_process_cpu = float(
        statistics.mean(
            process_cpu_samples
        )
    )

    peak_process_cpu = float(
        max(process_cpu_samples)
    )

else:

    average_process_cpu = None

    peak_process_cpu = None


if process_ram_samples:

    average_process_ram = float(
        statistics.mean(
            process_ram_samples
        )
    )

    peak_process_ram = float(
        max(process_ram_samples)
    )

else:

    average_process_ram = None

    peak_process_ram = None


if system_cpu_samples:

    average_system_cpu = float(
        statistics.mean(
            system_cpu_samples
        )
    )

    peak_system_cpu = float(
        max(system_cpu_samples)
    )

else:

    average_system_cpu = None

    peak_system_cpu = None


# ============================================================
# TRACKS BY CLASS
# ============================================================

tracks_by_class = {}

for record in tracking_records:

    class_name = record[
        "class_name"
    ]

    tracks_by_class.setdefault(
        class_name,
        set()
    )

    tracks_by_class[
        class_name
    ].add(
        record["track_id"]
    )

tracks_by_class = {

    class_name:
        len(track_ids)

    for class_name, track_ids
    in tracks_by_class.items()
}


# ============================================================
# EXISTING METRICS JSON
# ============================================================

metrics = {

    "experiment": {

        "tracker": TRACKER_NAME,

        "model": "YOLO11m",

        "model_format":
            "PyTorch (.pt)",

        "device": DEVICE,

        "confidence_threshold":
            CONFIDENCE_THRESHOLD,

        "iou_threshold":
            IOU_THRESHOLD,

        "image_size":
            IMAGE_SIZE,
    },

    "input_video": {

        "filename":
            VIDEO_PATH.name,

        "width":
            frame_width,

        "height":
            frame_height,

        "fps":
            input_fps,

        "total_frames_expected":
            expected_frames,
    },

    "processing": {

        "frames_processed":
            frames_processed,

        "frames_with_tracks":
            frames_with_tracks,

        "total_detections":
            total_detections,

        "total_tracking_records":
            len(tracking_records),

        "total_processing_time_seconds":
            total_processing_time,

        "average_inference_time_per_frame_seconds":
            average_inference_time,

        "average_inference_time_per_frame_ms":
            average_inference_time_ms,

        "average_frame_processing_time_ms":
            average_frame_processing_time_ms,

        "complete_pipeline_fps":
            pipeline_fps,
    },

    "tracking": {

        "total_unique_track_ids":
            len(unique_track_ids),

        "average_tracks_per_frame":
            (
                len(tracking_records)
                / frames_processed
                if frames_processed > 0
                else 0
            ),

        "average_track_length_frames":
            average_track_lifetime,

        "maximum_track_length_frames":
            longest_track_lifetime,

        "tracks_by_class":
            tracks_by_class,
    },
}


with open(
    OUTPUT_METRICS,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metrics,
        f,
        indent=4,
    )


# ============================================================
# TRACKING RECORDS JSON
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


# ============================================================
# NEW JSON
# SAME STRUCTURE AS BOT-SORT REFERENCE
# ============================================================

tracking_metrics = {

    "project": {

        "name":
            "YOLO11m + MCByteTrack Object Tracking",

        "tracker":
            TRACKER_NAME,
    },

    "model": {

        "name":
            "YOLO11m",

        "format":
            "pt",

        "path":
            str(MODEL_PATH),
    },

    "configuration": {

        "confidence_threshold":
            CONFIDENCE_THRESHOLD,

        "iou_threshold":
            IOU_THRESHOLD,

        "tracker_config":
            TRACKER_CONFIG,

        "device":
            DEVICE.upper(),
    },

    "video": {

        "input":
            str(VIDEO_PATH),

        "width":
            frame_width,

        "height":
            frame_height,

        "fps":
            input_fps,

        "total_frames":
            expected_frames,
    },

    "tracking_statistics": {

        "frame_count":
            frames_processed,

        "total_detections":
            total_detections,

        "unique_track_ids":
            len(unique_track_ids),

        "average_active_tracks":
            average_active_tracks,

        "maximum_active_tracks":
            maximum_active_tracks,

        "average_track_lifetime":
            average_track_lifetime,

        "longest_track_lifetime":
            longest_track_lifetime,

        "average_fps":
            pipeline_fps,

        "average_latency_ms":
            average_frame_processing_time_ms,

        "p95_latency_ms":
            p95_latency_ms,

        "total_processing_time_seconds":
            total_processing_time,
    },

    "resources": {

        "process": {

            "average_cpu_percent":
                average_process_cpu,

            "peak_cpu_percent":
                peak_process_cpu,

            "average_ram_mb":
                average_process_ram,

            "peak_ram_mb":
                peak_process_ram,
        },

        "system": {

            "average_cpu_percent":
                average_system_cpu,

            "peak_cpu_percent":
                peak_system_cpu,
        },
    },

    "outputs": {

        "video":
            str(OUTPUT_VIDEO),

        "json":
            str(OUTPUT_TRACKING_METRICS),

        "log":
            str(OUTPUT_LOG),
    },
}


# ============================================================
# SAVE NEW JSON
# ============================================================

with open(
    OUTPUT_TRACKING_METRICS,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        tracking_metrics,
        f,
        indent=4,
    )


# ============================================================
# SAVE CSV
# ============================================================

if tracking_records:

    fieldnames = list(
        tracking_records[0].keys()
    )

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

        writer_csv.writerows(
            tracking_records
        )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_lines = [

    "MCByteTrack / YOLO11m Tracking Summary",

    "=" * 60,

    "",

    f"Video: {VIDEO_PATH.name}",

    "Model: YOLO11m",

    f"Tracker: {TRACKER_NAME}",

    f"Device: {DEVICE}",

    "",

    f"Frames processed: "
    f"{frames_processed}",

    f"Frames with tracks: "
    f"{frames_with_tracks}",

    f"Total detections: "
    f"{total_detections}",

    "",

    f"Unique track IDs: "
    f"{len(unique_track_ids)}",

    f"Average active tracks: "
    f"{average_active_tracks:.4f}",

    f"Maximum active tracks: "
    f"{maximum_active_tracks}",

    f"Average track lifetime: "
    f"{average_track_lifetime:.2f} frames",

    f"Longest track lifetime: "
    f"{longest_track_lifetime} frames",

    "",

    f"Total processing time: "
    f"{total_processing_time:.4f} seconds",

    f"Average FPS: "
    f"{pipeline_fps:.4f}",

    f"Average latency: "
    f"{average_frame_processing_time_ms:.4f} ms",

    f"P95 latency: "
    f"{p95_latency_ms:.4f} ms",
]


with open(
    OUTPUT_SUMMARY,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "\n".join(summary_lines)
    )


# ============================================================
# FINAL LOG
# ============================================================

logger.info(
    "Processing completed successfully."
)

logger.info(
    f"Frames processed: {frames_processed}"
)

logger.info(
    f"Total detections: {total_detections}"
)

logger.info(
    f"Unique tracks: {len(unique_track_ids)}"
)

logger.info(
    f"Average FPS: {pipeline_fps}"
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()

print("=" * 70)

print(
    "MCByteTrack processing completed successfully."
)

print("=" * 70)

print(
    f"Frames processed       : "
    f"{frames_processed}"
)

print(
    f"Total detections       : "
    f"{total_detections}"
)

print(
    f"Unique track IDs       : "
    f"{len(unique_track_ids)}"
)

print(
    f"Average active tracks  : "
    f"{average_active_tracks:.2f}"
)

print(
    f"Maximum active tracks  : "
    f"{maximum_active_tracks}"
)

print(
    f"Average track lifetime : "
    f"{average_track_lifetime:.2f}"
)

print(
    f"Longest track lifetime : "
    f"{longest_track_lifetime}"
)

print(
    f"Average FPS            : "
    f"{pipeline_fps:.2f}"
)

print(
    f"Average latency        : "
    f"{average_frame_processing_time_ms:.2f} ms"
)

print(
    f"P95 latency            : "
    f"{p95_latency_ms:.2f} ms"
)

print(
    f"Processing time        : "
    f"{total_processing_time:.2f} sec"
)

print()

print(
    f"Video output : "
    f"{OUTPUT_VIDEO}"
)

print(
    f"New metrics : "
    f"{OUTPUT_TRACKING_METRICS}"
)

print(
    f"Log output   : "
    f"{OUTPUT_LOG}"
)

print("=" * 70)