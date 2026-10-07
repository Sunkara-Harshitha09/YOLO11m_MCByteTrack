from pathlib import Path
import csv
import json
import time
import logging
import statistics
import subprocess
import shutil

import cv2
from ultralytics import YOLO
import supervision as sv


# ============================================================
# APPLICATION PATHS
# ============================================================

APP_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = APP_ROOT / "model" / "yolo11m.pt"

OUTPUT_ROOT = APP_ROOT / "output"

OUTPUT_VIDEO_DIR = OUTPUT_ROOT / "videos"
OUTPUT_TRACKING_DIR = OUTPUT_ROOT / "tracking"
OUTPUT_METRICS_DIR = OUTPUT_ROOT / "metrics"
OUTPUT_LOG_DIR = OUTPUT_ROOT / "logs"


# ============================================================
# SETTINGS
# ============================================================

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMAGE_SIZE = 640
DEVICE = "cpu"

TRACKER_NAME = "MCByteTrack"
TRACKER_CONFIG = "supervision.ByteTrack"

VIDEO_CODEC = "H.264"
H264_ENCODER = "libx264"


# ============================================================
# OUTPUT FILES
# ============================================================

OUTPUT_VIDEO = (
    OUTPUT_VIDEO_DIR /
    "yolo11m_mcbytetrack_output.mp4"
)

OUTPUT_CSV = (
    OUTPUT_TRACKING_DIR /
    "mcbytetrack_tracks.csv"
)

OUTPUT_JSON = (
    OUTPUT_TRACKING_DIR /
    "mcbytetrack_tracks.json"
)

OUTPUT_METRICS = (
    OUTPUT_METRICS_DIR /
    "mcbytetrack_metrics.json"
)

OUTPUT_TRACKING_METRICS = (
    OUTPUT_METRICS_DIR /
    "mcbytetrack_tracking_metrics.json"
)

OUTPUT_SUMMARY = (
    OUTPUT_METRICS_DIR /
    "mcbytetrack_summary.txt"
)

OUTPUT_LOG = (
    OUTPUT_LOG_DIR /
    "mcbytetrack.log"
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for directory in [
    OUTPUT_VIDEO_DIR,
    OUTPUT_TRACKING_DIR,
    OUTPUT_METRICS_DIR,
    OUTPUT_LOG_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger("MCByteTrack")
logger.setLevel(logging.INFO)

logger.handlers.clear()

file_handler = logging.FileHandler(
    OUTPUT_LOG,
    mode="w",
    encoding="utf-8"
)

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
)

file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# ============================================================
# FIND FFMPEG
# ============================================================

def find_ffmpeg():

    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path is None:

        raise RuntimeError(
            "FFmpeg was not found in PATH.\n"
            "Please make sure FFmpeg is installed "
            "and available in the terminal."
        )

    return ffmpeg_path


# ============================================================
# INITIALIZE MCByteTrack
# ============================================================

def initialize_tracker():

    tracker = sv.ByteTrack()

    logger.info(
        "Tracker initialized: %s",
        TRACKER_NAME
    )

    logger.info(
        "Tracker implementation: %s",
        TRACKER_CONFIG
    )

    return tracker


# ============================================================
# RUN MCByteTrack
# ============================================================

def run_mcbytetrack(input_video):

    input_video = Path(input_video)

    print()
    print("=" * 70)
    print("YOLO11m + MCByteTrack")
    print("=" * 70)

    print(f"Input video : {input_video}")
    print(f"Model       : {MODEL_PATH}")
    print(f"Output      : {OUTPUT_ROOT}")
    print()

    logger.info("Starting MCByteTrack")
    logger.info("Input video: %s", input_video)
    logger.info("Model: %s", MODEL_PATH)

    # ========================================================
    # VALIDATE INPUTS
    # ========================================================

    if not input_video.exists():

        raise FileNotFoundError(
            f"Input video not found: {input_video}"
        )

    if not input_video.is_file():

        raise FileNotFoundError(
            f"Input path is not a file: {input_video}"
        )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"YOLO11m model not found: {MODEL_PATH}"
        )

    # ========================================================
    # FIND FFMPEG
    # ========================================================

    ffmpeg = find_ffmpeg()

    print(f"FFmpeg      : {ffmpeg}")
    print()

    logger.info(
        "FFmpeg path: %s",
        ffmpeg
    )

    # ========================================================
    # LOAD YOLO11m
    # ========================================================

    print("Loading YOLO11m...")

    model = YOLO(
        str(MODEL_PATH)
    )

    print("YOLO11m loaded successfully.")

    logger.info(
        "YOLO11m loaded successfully."
    )

    # ========================================================
    # INITIALIZE TRACKER
    # ========================================================

    print("Initializing MCByteTrack...")

    tracker = initialize_tracker()

    print("MCByteTrack initialized.")
    print()

    # ========================================================
    # OPEN VIDEO
    # ========================================================

    cap = cv2.VideoCapture(
        str(input_video)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video: {input_video}"
        )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps <= 0:

        fps = 30.0

    print("Video information")
    print("-" * 70)
    print(f"Resolution : {width} x {height}")
    print(f"FPS        : {fps:.2f}")
    print(f"Frames     : {total_frames}")
    print()

    logger.info(
        "Video resolution: %s x %s",
        width,
        height
    )

    logger.info(
        "Video FPS: %.2f",
        fps
    )

    logger.info(
        "Video frames: %s",
        total_frames
    )

    # ========================================================
    # FFMPEG H.264 COMMAND
    # ========================================================

    ffmpeg_command = [
        ffmpeg,

        "-y",

        # Raw video input
        "-f",
        "rawvideo",

        "-pix_fmt",
        "bgr24",

        "-video_size",
        f"{width}x{height}",

        "-framerate",
        str(fps),

        "-i",
        "-",

        # No audio
        "-an",

        # H.264
        "-c:v",
        H264_ENCODER,

        # Encoding quality
        "-preset",
        "medium",

        "-crf",
        "23",

        # Compatibility
        "-pix_fmt",
        "yuv420p",

        # Better MP4 playback/startup
        "-movflags",
        "+faststart",

        str(OUTPUT_VIDEO),
    ]

    print("Starting H.264 encoder...")
    print()

    logger.info(
        "Starting FFmpeg H.264 encoder"
    )

    # ========================================================
    # OPEN FFMPEG LOG
    # ========================================================

    ffmpeg_log_file = open(
        OUTPUT_LOG,
        "a",
        encoding="utf-8"
    )

    # ========================================================
    # START FFMPEG
    # ========================================================

    ffmpeg_process = subprocess.Popen(
        ffmpeg_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=ffmpeg_log_file,
        bufsize=0
    )

    print("H.264 encoder started.")
    print()

    logger.info(
        "H.264 encoder started."
    )

    # ========================================================
    # METRICS VARIABLES
    # ========================================================

    detection_count = 0

    frames_processed = 0

    frames_with_detections = 0

    unique_track_ids = set()

    active_tracks_per_frame = []

    track_first_frame = {}

    track_last_frame = {}

    track_lengths = {}

    confidence_values = []

    class_counts = {}

    frame_times = []

    tracking_records = []

    start_time = time.perf_counter()

    ffmpeg_finished = False

    # ========================================================
    # PROCESS VIDEO
    # ========================================================

    print("Processing video...")
    print()

    try:

        while True:

            ret, frame = cap.read()

            if not ret:

                break

            frame_start = time.perf_counter()

            frames_processed += 1

            # =================================================
            # YOLO DETECTION
            # =================================================

            results = model.predict(
                source=frame,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                imgsz=IMAGE_SIZE,
                device=DEVICE,
                verbose=False
            )

            result = results[0]

            # =================================================
            # CONVERT YOLO → SUPERVISION
            # =================================================

            detections = (
                sv.Detections
                .from_ultralytics(result)
            )

            detection_count += len(
                detections
            )

            if len(detections) > 0:

                frames_with_detections += 1

            # =================================================
            # DETECTION STATISTICS
            # =================================================

            for index in range(
                len(detections)
            ):

                # Confidence
                if detections.confidence is not None:

                    confidence = float(
                        detections.confidence[index]
                    )

                    confidence_values.append(
                        confidence
                    )

                # Class
                if detections.class_id is not None:

                    class_id = int(
                        detections.class_id[index]
                    )

                    class_name = (
                        model.names[class_id]
                        if class_id in model.names
                        else str(class_id)
                    )

                    class_counts[class_name] = (
                        class_counts.get(
                            class_name,
                            0
                        ) + 1
                    )

            # =================================================
            # MCByteTrack
            # =================================================

            tracked_detections = (
                tracker.update_with_detections(
                    detections
                )
            )

            current_track_ids = []

            if tracked_detections.tracker_id is not None:

                for index, tracker_id in enumerate(
                    tracked_detections.tracker_id
                ):

                    tracker_id = int(
                        tracker_id
                    )

                    current_track_ids.append(
                        tracker_id
                    )

                    unique_track_ids.add(
                        tracker_id
                    )

                    # -----------------------------------------
                    # Track lifetime
                    # -----------------------------------------

                    if tracker_id not in track_first_frame:

                        track_first_frame[
                            tracker_id
                        ] = frames_processed

                    track_last_frame[
                        tracker_id
                    ] = frames_processed

                    track_lengths[
                        tracker_id
                    ] = (
                        track_last_frame[
                            tracker_id
                        ]
                        -
                        track_first_frame[
                            tracker_id
                        ]
                        +
                        1
                    )

                    # -----------------------------------------
                    # Bounding box
                    # -----------------------------------------

                    xyxy = (
                        tracked_detections
                        .xyxy[index]
                    )

                    x1 = float(xyxy[0])
                    y1 = float(xyxy[1])
                    x2 = float(xyxy[2])
                    y2 = float(xyxy[3])

                    # -----------------------------------------
                    # Confidence
                    # -----------------------------------------

                    confidence = None

                    if (
                        tracked_detections.confidence
                        is not None
                    ):

                        confidence = float(
                            tracked_detections
                            .confidence[index]
                        )

                    # -----------------------------------------
                    # Class
                    # -----------------------------------------

                    class_id = None
                    class_name = None

                    if (
                        tracked_detections.class_id
                        is not None
                    ):

                        class_id = int(
                            tracked_detections
                            .class_id[index]
                        )

                        class_name = (
                            model.names[class_id]
                            if class_id in model.names
                            else str(class_id)
                        )

                    # -----------------------------------------
                    # Tracking record
                    # -----------------------------------------

                    tracking_records.append(
                        {
                            "frame":
                                frames_processed,

                            "track_id":
                                tracker_id,

                            "class_id":
                                class_id,

                            "class_name":
                                class_name,

                            "confidence":
                                confidence,

                            "x1":
                                x1,

                            "y1":
                                y1,

                            "x2":
                                x2,

                            "y2":
                                y2,
                        }
                    )

            active_tracks_per_frame.append(
                len(current_track_ids)
            )

            # =================================================
            # ANNOTATE FRAME
            # =================================================

            annotated_frame = frame.copy()

            labels = []

            if tracked_detections.tracker_id is not None:

                for index, tracker_id in enumerate(
                    tracked_detections.tracker_id
                ):

                    class_name = "object"

                    if (
                        tracked_detections.class_id
                        is not None
                    ):

                        class_id = int(
                            tracked_detections
                            .class_id[index]
                        )

                        class_name = (
                            model.names[class_id]
                            if class_id in model.names
                            else str(class_id)
                        )

                    confidence = 0.0

                    if (
                        tracked_detections.confidence
                        is not None
                    ):

                        confidence = float(
                            tracked_detections
                            .confidence[index]
                        )

                    labels.append(
                        f"{class_name} "
                        f"ID:{int(tracker_id)} "
                        f"{confidence:.2f}"
                    )

            # =================================================
            # DRAW BOXES
            # =================================================

            box_annotator = sv.BoxAnnotator()

            label_annotator = sv.LabelAnnotator()

            annotated_frame = (
                box_annotator.annotate(
                    scene=annotated_frame,
                    detections=tracked_detections
                )
            )

            annotated_frame = (
                label_annotator.annotate(
                    scene=annotated_frame,
                    detections=tracked_detections,
                    labels=labels
                )
            )

            # =================================================
            # SEND FRAME TO FFMPEG
            # =================================================

            try:

                ffmpeg_process.stdin.write(
                    annotated_frame.tobytes()
                )

            except BrokenPipeError:

                raise RuntimeError(
                    "FFmpeg stopped unexpectedly. "
                    f"Check the log file: {OUTPUT_LOG}"
                )

            # =================================================
            # TIMING
            # =================================================

            frame_time = (
                time.perf_counter()
                -
                frame_start
            )

            frame_times.append(
                frame_time
            )

            # =================================================
            # PROGRESS
            # =================================================

            if (
                frames_processed % 50 == 0
                or frames_processed == total_frames
            ):

                progress = (
                    frames_processed
                    /
                    total_frames
                    *
                    100
                    if total_frames > 0
                    else 0
                )

                print(
                    f"\rProcessed: "
                    f"{frames_processed}/"
                    f"{total_frames} "
                    f"({progress:.1f}%)",
                    end="",
                    flush=True
                )

    finally:

        # ====================================================
        # CLOSE INPUT VIDEO
        # ====================================================

        cap.release()

        print()
        print()
        print("Finishing H.264 encoding...")
        print()

        logger.info(
            "Finished processing video frames."
        )

        # ====================================================
        # CLOSE FFMPEG INPUT
        # ====================================================

        try:

            if ffmpeg_process.stdin:

                ffmpeg_process.stdin.close()

        except Exception as close_error:

            logger.warning(
                "FFmpeg stdin close warning: %s",
                close_error
            )

        # ====================================================
        # WAIT FOR FFMPEG
        # ====================================================

        return_code = (
            ffmpeg_process.wait()
        )

        ffmpeg_finished = True

        ffmpeg_log_file.close()

        if return_code != 0:

            raise RuntimeError(
                "FFmpeg H.264 encoding failed. "
                f"Check: {OUTPUT_LOG}"
            )

    print("H.264 encoding completed.")
    print()

    logger.info(
        "H.264 encoding completed."
    )

    # ========================================================
    # TOTAL PROCESSING TIME
    # ========================================================

    total_processing_time = (
        time.perf_counter()
        -
        start_time
    )

    # ========================================================
    # PROCESSING METRICS
    # ========================================================

    average_frame_time = (
        statistics.mean(frame_times)
        if frame_times
        else 0
    )

    average_fps = (
        frames_processed
        /
        total_processing_time
        if total_processing_time > 0
        else 0
    )

    average_latency_ms = (
        average_frame_time
        *
        1000
    )

    if frame_times:

        sorted_times = sorted(
            frame_times
        )

        p95_index = int(
            len(sorted_times) * 0.95
        )

        p95_index = min(
            p95_index,
            len(sorted_times) - 1
        )

        p95_latency_ms = (
            sorted_times[p95_index]
            *
            1000
        )

    else:

        p95_latency_ms = 0

    # ========================================================
    # TRACKING METRICS
    # ========================================================

    unique_tracks = len(
        unique_track_ids
    )

    average_active_tracks = (
        statistics.mean(
            active_tracks_per_frame
        )
        if active_tracks_per_frame
        else 0
    )

    max_active_tracks = (
        max(active_tracks_per_frame)
        if active_tracks_per_frame
        else 0
    )

    average_track_lifetime = (
        statistics.mean(
            track_lengths.values()
        )
        if track_lengths
        else 0
    )

    longest_track = (
        max(track_lengths.values())
        if track_lengths
        else 0
    )

    # ========================================================
    # CONFIDENCE METRICS
    # ========================================================

    if confidence_values:

        average_confidence = statistics.mean(
            confidence_values
        )

        minimum_confidence = min(
            confidence_values
        )

        maximum_confidence = max(
            confidence_values
        )

    else:

        average_confidence = 0
        minimum_confidence = 0
        maximum_confidence = 0

    # ========================================================
    # SAVE TRACKING CSV
    # ========================================================

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "frame",
            "track_id",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
        ]

        csv_writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        csv_writer.writeheader()

        csv_writer.writerows(
            tracking_records
        )

    # ========================================================
    # SAVE TRACKING JSON
    # ========================================================

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            tracking_records,
            file,
            indent=2
        )

    # ========================================================
    # MAIN METRICS JSON
    # ========================================================

    metrics = {

        "project":
            "YOLO11m + MCByteTrack",

        "tracker_name":
            TRACKER_NAME,

        "tracker_implementation":
            TRACKER_CONFIG,

        "video_codec":
            VIDEO_CODEC,

        "h264_encoder":
            H264_ENCODER,

        "input_video":
            str(input_video),

        "model":
            str(MODEL_PATH),

        "video": {

            "width":
                width,

            "height":
                height,

            "fps":
                fps,

            "total_frames":
                total_frames,
        },

        "configuration": {

            "confidence_threshold":
                CONFIDENCE_THRESHOLD,

            "iou_threshold":
                IOU_THRESHOLD,

            "image_size":
                IMAGE_SIZE,

            "device":
                DEVICE,
        },

        "processing": {

            "frames_processed":
                frames_processed,

            "frames_with_detections":
                frames_with_detections,

            "total_detections":
                detection_count,

            "processing_time_seconds":
                total_processing_time,

            "average_fps":
                average_fps,

            "average_latency_ms":
                average_latency_ms,

            "p95_latency_ms":
                p95_latency_ms,
        },

        "tracking": {

            "unique_track_ids":
                unique_tracks,

            "average_active_tracks":
                average_active_tracks,

            "max_active_tracks":
                max_active_tracks,

            "average_track_lifetime_frames":
                average_track_lifetime,

            "longest_track_frames":
                longest_track,
        },

        "confidence": {

            "average":
                average_confidence,

            "minimum":
                minimum_confidence,

            "maximum":
                maximum_confidence,
        },

        "detections_by_class":
            class_counts,

        "output_files": {

            "video":
                str(OUTPUT_VIDEO),

            "csv":
                str(OUTPUT_CSV),

            "json":
                str(OUTPUT_JSON),

            "metrics":
                str(OUTPUT_METRICS),

            "tracking_metrics":
                str(OUTPUT_TRACKING_METRICS),

            "summary":
                str(OUTPUT_SUMMARY),

            "log":
                str(OUTPUT_LOG),
        },
    }

    with open(
        OUTPUT_METRICS,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # ========================================================
    # TRACKING METRICS JSON
    # ========================================================

    tracking_metrics = {

        "tracker":
            TRACKER_NAME,

        "tracker_implementation":
            TRACKER_CONFIG,

        "video_codec":
            VIDEO_CODEC,

        "h264_encoder":
            H264_ENCODER,

        "frames_processed":
            frames_processed,

        "total_detections":
            detection_count,

        "unique_track_ids":
            unique_tracks,

        "average_active_tracks":
            average_active_tracks,

        "max_active_tracks":
            max_active_tracks,

        "average_track_lifetime":
            average_track_lifetime,

        "longest_track":
            longest_track,

        "average_fps":
            average_fps,

        "average_latency_ms":
            average_latency_ms,

        "p95_latency_ms":
            p95_latency_ms,

        "processing_time_seconds":
            total_processing_time,

        "confidence": {

            "average":
                average_confidence,

            "minimum":
                minimum_confidence,

            "maximum":
                maximum_confidence,
        },

        "detections_by_class":
            class_counts,
    }

    with open(
        OUTPUT_TRACKING_METRICS,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            tracking_metrics,
            file,
            indent=4
        )

    # ========================================================
    # SUMMARY TXT
    # ========================================================

    summary = f"""
YOLO11m + MCByteTrack
============================================================

Input Video:
{input_video}

Model:
{MODEL_PATH}

Tracker:
{TRACKER_NAME}

Tracker Implementation:
{TRACKER_CONFIG}

Video Codec:
{VIDEO_CODEC}

H.264 Encoder:
{H264_ENCODER}

------------------------------------------------------------
Configuration
------------------------------------------------------------

Confidence Threshold: {CONFIDENCE_THRESHOLD}
IoU Threshold: {IOU_THRESHOLD}
Image Size: {IMAGE_SIZE}
Device: {DEVICE}

------------------------------------------------------------
Video
------------------------------------------------------------

Resolution: {width} x {height}
FPS: {fps:.2f}
Total Frames: {total_frames}

------------------------------------------------------------
Processing
------------------------------------------------------------

Frames Processed: {frames_processed}
Frames With Detections: {frames_with_detections}
Total Detections: {detection_count}

Processing Time:
{total_processing_time:.2f} seconds

Average FPS:
{average_fps:.2f}

Average Latency:
{average_latency_ms:.2f} ms

P95 Latency:
{p95_latency_ms:.2f} ms

------------------------------------------------------------
Tracking
------------------------------------------------------------

Unique Track IDs:
{unique_tracks}

Average Active Tracks:
{average_active_tracks:.2f}

Maximum Active Tracks:
{max_active_tracks}

Average Track Lifetime:
{average_track_lifetime:.2f} frames

Longest Track:
{longest_track} frames

------------------------------------------------------------
Confidence
------------------------------------------------------------

Average:
{average_confidence:.4f}

Minimum:
{minimum_confidence:.4f}

Maximum:
{maximum_confidence:.4f}

------------------------------------------------------------
Output
------------------------------------------------------------

H.264 Video:
{OUTPUT_VIDEO}

Tracking CSV:
{OUTPUT_CSV}

Tracking JSON:
{OUTPUT_JSON}

Metrics JSON:
{OUTPUT_METRICS}

Tracking Metrics JSON:
{OUTPUT_TRACKING_METRICS}

Summary:
{OUTPUT_SUMMARY}

Log:
{OUTPUT_LOG}
"""

    with open(
        OUTPUT_SUMMARY,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            summary.strip()
        )

    # ========================================================
    # FINAL CONSOLE OUTPUT
    # ========================================================

    print("=" * 70)
    print("MCByteTrack COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print()

    print(
        f"Frames processed       : "
        f"{frames_processed}"
    )

    print(
        f"Total detections       : "
        f"{detection_count}"
    )

    print(
        f"Unique track IDs       : "
        f"{unique_tracks}"
    )

    print(
        f"Average active tracks  : "
        f"{average_active_tracks:.2f}"
    )

    print(
        f"Maximum active tracks  : "
        f"{max_active_tracks}"
    )

    print(
        f"Average track lifetime : "
        f"{average_track_lifetime:.2f}"
    )

    print(
        f"Longest track          : "
        f"{longest_track}"
    )

    print(
        f"Average FPS            : "
        f"{average_fps:.2f}"
    )

    print(
        f"Average latency        : "
        f"{average_latency_ms:.2f} ms"
    )

    print()
    print("Video codec : H.264")
    print("Encoder     : libx264")
    print()
    print("H.264 video:")
    print(OUTPUT_VIDEO)
    print()
    print("All outputs:")
    print(OUTPUT_ROOT)
    print()

    print("=" * 70)

    logger.info(
        "MCByteTrack completed successfully."
    )

    return metrics


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    print()
    print("YOLO11m + MCByteTrack")
    print()

    video_path = input(
        "Enter input video path: "
    ).strip().strip('"')

    if not video_path:

        raise ValueError(
            "Input video path cannot be empty."
        )

    run_mcbytetrack(
        video_path
    )