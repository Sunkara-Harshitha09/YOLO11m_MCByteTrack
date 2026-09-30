from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRACKING_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "mcbytetrack"
    / "tracking"
    / "mcbytetrack_tracks.csv"
)

TRACKING_JSON = (
    PROJECT_ROOT
    / "outputs"
    / "mcbytetrack"
    / "tracking"
    / "mcbytetrack_tracks.json"
)

METRICS_JSON = (
    PROJECT_ROOT
    / "outputs"
    / "mcbytetrack"
    / "metrics"
    / "mcbytetrack_metrics.json"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "mcbytetrack"
    / "metrics"
)

PLOTS_DIR = METRICS_DIR / "plots"


# ============================================================
# CREATE DIRECTORIES
# ============================================================

METRICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# VALIDATE INPUT FILES
# ============================================================

for file_path in [
    TRACKING_CSV,
    TRACKING_JSON,
    METRICS_JSON,
]:

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}"
        )


print("=" * 70)
print("MCByteTrack Result Analysis")
print("=" * 70)

print(f"Tracking CSV : {TRACKING_CSV}")
print(f"Tracking JSON: {TRACKING_JSON}")
print(f"Metrics JSON : {METRICS_JSON}")
print()


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(TRACKING_CSV)

with open(
    METRICS_JSON,
    "r",
    encoding="utf-8",
) as f:

    metrics = json.load(f)


if df.empty:
    raise RuntimeError(
        "Tracking CSV is empty. No tracking records available."
    )


print(f"Tracking records loaded: {len(df)}")
print()


# ============================================================
# BASIC INFORMATION
# ============================================================

frames_processed = metrics["processing"]["frames_processed"]

unique_track_ids = df["track_id"].nunique()

unique_classes = df["class_name"].nunique()

total_records = len(df)

average_confidence = df["confidence"].mean()

minimum_confidence = df["confidence"].min()

maximum_confidence = df["confidence"].max()


# ============================================================
# TRACKS PER FRAME
# ============================================================

tracks_per_frame = (
    df.groupby("frame_number")
    .agg(
        active_tracks=("track_id", "nunique"),
        tracking_records=("track_id", "count"),
    )
    .reset_index()
)

tracks_per_frame["timestamp_seconds"] = (
    df.groupby("frame_number")["timestamp_seconds"]
    .first()
    .values
)

tracks_per_frame.to_csv(
    METRICS_DIR / "tracks_per_frame.csv",
    index=False,
)


# ============================================================
# TRACK LENGTH
# ============================================================

track_length_df = (
    df.groupby(
        ["track_id", "class_name"],
        as_index=False,
    )
    .agg(
        first_frame=("frame_number", "min"),
        last_frame=("frame_number", "max"),
        track_length_frames=("frame_number", "count"),
        average_confidence=("confidence", "mean"),
        minimum_confidence=("confidence", "min"),
        maximum_confidence=("confidence", "max"),
    )
)

track_length_df["track_duration_seconds"] = (
    track_length_df["track_length_frames"]
    / metrics["input_video"]["fps"]
)

track_length_df.to_csv(
    METRICS_DIR / "track_length_statistics.csv",
    index=False,
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

class_distribution = (
    df.groupby("class_name")
    .agg(
        tracking_records=("track_id", "count"),
        unique_tracks=("track_id", "nunique"),
        average_confidence=("confidence", "mean"),
    )
    .reset_index()
)

class_distribution["record_percentage"] = (
    class_distribution["tracking_records"]
    / total_records
    * 100
)

class_distribution.to_csv(
    METRICS_DIR / "class_distribution.csv",
    index=False,
)


# ============================================================
# CONFIDENCE STATISTICS
# ============================================================

confidence_statistics = pd.DataFrame(
    [
        {
            "metric": "average",
            "value": average_confidence,
        },
        {
            "metric": "minimum",
            "value": minimum_confidence,
        },
        {
            "metric": "maximum",
            "value": maximum_confidence,
        },
        {
            "metric": "median",
            "value": df["confidence"].median(),
        },
        {
            "metric": "standard_deviation",
            "value": df["confidence"].std(),
        },
    ]
)

confidence_statistics.to_csv(
    METRICS_DIR / "confidence_statistics.csv",
    index=False,
)


# ============================================================
# OVERALL TRACKING METRICS
# ============================================================

track_lengths = track_length_df["track_length_frames"]

tracking_analysis_metrics = {
    "total_frames": int(frames_processed),
    "total_tracking_records": int(total_records),
    "unique_track_ids": int(unique_track_ids),
    "unique_classes": int(unique_classes),

    "average_tracks_per_frame": float(
        tracks_per_frame["active_tracks"].mean()
    ),

    "minimum_tracks_per_frame": int(
        tracks_per_frame["active_tracks"].min()
    ),

    "maximum_tracks_per_frame": int(
        tracks_per_frame["active_tracks"].max()
    ),

    "average_track_length_frames": float(
        track_lengths.mean()
    ),

    "minimum_track_length_frames": int(
        track_lengths.min()
    ),

    "maximum_track_length_frames": int(
        track_lengths.max()
    ),

    "median_track_length_frames": float(
        track_lengths.median()
    ),

    "average_track_duration_seconds": float(
        track_length_df["track_duration_seconds"].mean()
    ),

    "average_confidence": float(
        df["confidence"].mean()
    ),

    "minimum_confidence": float(
        df["confidence"].min()
    ),

    "maximum_confidence": float(
        df["confidence"].max()
    ),

    "total_processing_time_seconds": float(
        metrics["processing"]
        ["total_processing_time_seconds"]
    ),

    "average_inference_time_per_frame_ms": float(
        metrics["processing"]
        ["average_inference_time_per_frame_ms"]
    ),

    "inference_fps": float(
        metrics["processing"]["inference_fps"]
    ),

    "complete_pipeline_fps": float(
        metrics["processing"]["complete_pipeline_fps"]
    ),

    "ground_truth_metrics": {
        "precision": "Ground truth required",
        "recall": "Ground truth required",
        "f1_score": "Ground truth required",
        "map50": "Ground truth required",
        "map50_95": "Ground truth required",
        "iou": "Ground truth required",
        "mota": "Ground truth required",
        "motp": "Ground truth required",
        "idf1": "Ground truth required",
        "hota": "Ground truth required",
    },
}


# ============================================================
# SAVE ANALYSIS METRICS
# ============================================================

with open(
    METRICS_DIR / "mcbytetrack_analysis_metrics.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        tracking_analysis_metrics,
        f,
        indent=2,
    )


# ============================================================
# EXPERIMENT CONFIG
# ============================================================

experiment_config = {
    "model": "YOLO11m",
    "model_format": "PyTorch (.pt)",
    "tracker": "ByteTrack",
    "video": metrics["input_video"],
    "confidence_threshold": metrics["experiment"]
    ["confidence_threshold"],
    "iou_threshold": metrics["experiment"]
    ["iou_threshold"],
    "image_size": metrics["experiment"]
    ["image_size"],
    "device": metrics["experiment"]["device"],
}


with open(
    METRICS_DIR / "experiment_config.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        experiment_config,
        f,
        indent=2,
    )


# ============================================================
# PLOT 1 — TRACKS PER FRAME
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    tracks_per_frame["frame_number"],
    tracks_per_frame["active_tracks"],
)

plt.xlabel("Frame Number")
plt.ylabel("Active Track IDs")
plt.title("MCByteTrack - Active Tracks Per Frame")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "tracks_per_frame.png",
    dpi=150,
)

plt.close()


# ============================================================
# PLOT 2 — TRACK LENGTH DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    track_lengths,
    bins=20,
)

plt.xlabel("Track Length (Frames)")
plt.ylabel("Number of Tracks")
plt.title("MCByteTrack - Track Length Distribution")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "track_length_distribution.png",
    dpi=150,
)

plt.close()


# ============================================================
# PLOT 3 — CLASS DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    class_distribution["class_name"],
    class_distribution["unique_tracks"],
)

plt.xlabel("Class")
plt.ylabel("Unique Track IDs")
plt.title("MCByteTrack - Unique Tracks by Class")
plt.xticks(rotation=30)
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "class_distribution.png",
    dpi=150,
)

plt.close()


# ============================================================
# PLOT 4 — CONFIDENCE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["confidence"],
    bins=30,
)

plt.xlabel("Confidence")
plt.ylabel("Tracking Records")
plt.title("MCByteTrack - Confidence Distribution")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "confidence_distribution.png",
    dpi=150,
)

plt.close()


# ============================================================
# DETAILED TEXT REPORT
# ============================================================

report_lines = [
    "MCByteTrack Detailed Analysis Report",
    "=" * 70,
    "",
    "EXPERIMENT",
    "-" * 70,
    f"Model: YOLO11m",
    f"Model format: PyTorch (.pt)",
    f"Tracker: ByteTrack",
    f"Video: {metrics['input_video']['filename']}",
    f"Device: {metrics['experiment']['device']}",
    f"Confidence threshold: {metrics['experiment']['confidence_threshold']}",
    f"IoU threshold: {metrics['experiment']['iou_threshold']}",
    f"Image size: {metrics['experiment']['image_size']}",
    "",
    "TRACKING RESULTS",
    "-" * 70,
    f"Frames processed: {frames_processed}",
    f"Tracking records: {total_records}",
    f"Unique track IDs: {unique_track_ids}",
    f"Unique classes: {unique_classes}",
    f"Average tracks/frame: {tracking_analysis_metrics['average_tracks_per_frame']:.4f}",
    f"Minimum tracks/frame: {tracking_analysis_metrics['minimum_tracks_per_frame']}",
    f"Maximum tracks/frame: {tracking_analysis_metrics['maximum_tracks_per_frame']}",
    f"Average track length: {tracking_analysis_metrics['average_track_length_frames']:.2f} frames",
    f"Median track length: {tracking_analysis_metrics['median_track_length_frames']:.2f} frames",
    f"Minimum track length: {tracking_analysis_metrics['minimum_track_length_frames']} frames",
    f"Maximum track length: {tracking_analysis_metrics['maximum_track_length_frames']} frames",
    f"Average track duration: {tracking_analysis_metrics['average_track_duration_seconds']:.2f} seconds",
    "",
    "CONFIDENCE",
    "-" * 70,
    f"Average confidence: {average_confidence:.6f}",
    f"Minimum confidence: {minimum_confidence:.6f}",
    f"Maximum confidence: {maximum_confidence:.6f}",
    f"Median confidence: {df['confidence'].median():.6f}",
    "",
    "PERFORMANCE",
    "-" * 70,
    f"Total processing time: {tracking_analysis_metrics['total_processing_time_seconds']:.4f} seconds",
    f"Average inference time/frame: {tracking_analysis_metrics['average_inference_time_per_frame_ms']:.4f} ms",
    f"Inference FPS: {tracking_analysis_metrics['inference_fps']:.4f}",
    f"Complete pipeline FPS: {tracking_analysis_metrics['complete_pipeline_fps']:.4f}",
    "",
    "TRACKS BY CLASS",
    "-" * 70,
]

for _, row in class_distribution.iterrows():

    report_lines.append(
        f"{row['class_name']}: "
        f"{int(row['unique_tracks'])} unique tracks, "
        f"{int(row['tracking_records'])} records, "
        f"average confidence "
        f"{row['average_confidence']:.4f}"
    )


report_lines.extend(
    [
        "",
        "GROUND-TRUTH DEPENDENT METRICS",
        "-" * 70,
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
        "",
        "Note:",
        "These metrics are not calculated because no ground-truth",
        "detection/tracking annotations are available for this video.",
    ]
)


with open(
    METRICS_DIR / "mcbytetrack_detailed_report.txt",
    "w",
    encoding="utf-8",
) as f:

    f.write("\n".join(report_lines))


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("MCByteTrack analysis completed successfully.")
print("=" * 70)

print()
print("Generated files:")

for file_path in sorted(METRICS_DIR.rglob("*")):

    if file_path.is_file():

        print(
            f"  {file_path.relative_to(PROJECT_ROOT)}"
        )

print()
print(f"Tracking records : {total_records}")
print(f"Unique track IDs : {unique_track_ids}")
print(
    f"Average track length : "
    f"{track_lengths.mean():.2f} frames"
)
print(
    f"Average confidence : "
    f"{average_confidence:.4f}"
)