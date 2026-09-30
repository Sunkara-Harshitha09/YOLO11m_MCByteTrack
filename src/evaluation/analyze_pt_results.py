import json
import os

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DETECTIONS_CSV = os.path.join(
    BASE_DIR,
    "outputs",
    "pt",
    "detections",
    "yolo11m_pt_detections.csv",
)

METRICS_JSON = os.path.join(
    BASE_DIR,
    "outputs",
    "pt",
    "metrics",
    "yolo11m_pt_metrics.json",
)

METRICS_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "pt",
    "metrics",
)

PLOTS_DIR = os.path.join(
    METRICS_DIR,
    "plots",
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ============================================================
# LOAD DETECTIONS
# ============================================================

if not os.path.exists(DETECTIONS_CSV):
    raise FileNotFoundError(
        f"Detection CSV not found:\n{DETECTIONS_CSV}"
    )

df = pd.read_csv(DETECTIONS_CSV)

if df.empty:
    raise RuntimeError("Detection CSV is empty.")


print("=" * 70)
print("YOLO11m PT RESULT ANALYSIS")
print("=" * 70)

print(f"Detection records: {len(df):,}")
print(f"Frames in detections: {df['frame_number'].nunique():,}")


# ============================================================
# LOAD ORIGINAL METRICS
# ============================================================

with open(METRICS_JSON, "r", encoding="utf-8") as file:
    metrics = json.load(file)


# ============================================================
# 1. METRICS CSV
# ============================================================

processing = metrics["processing"]
confidence = metrics["confidence"]

metrics_rows = [
    ["model", "YOLO11m"],
    ["model_format", "PyTorch (.pt)"],
    ["device", metrics["experiment"]["device"]],
    ["confidence_threshold", metrics["experiment"]["confidence_threshold"]],
    ["iou_threshold", metrics["experiment"]["iou_threshold"]],
    ["image_size", metrics["experiment"]["image_size"]],

    ["video_filename", metrics["input_video"]["filename"]],
    ["video_width", metrics["input_video"]["width"]],
    ["video_height", metrics["input_video"]["height"]],
    ["video_fps", metrics["input_video"]["fps"]],
    ["total_frames", processing["frames_processed"]],

    ["frames_with_detections", processing["frames_with_detections"]],
    ["total_detections", processing["total_detections"]],
    [
        "average_detections_per_frame",
        processing["total_detections"]
        / processing["frames_processed"],
    ],

    ["average_confidence", confidence["average"]],
    ["minimum_confidence", confidence["minimum"]],
    ["maximum_confidence", confidence["maximum"]],

    [
        "total_processing_time_seconds",
        processing["total_processing_time_seconds"],
    ],

    [
        "average_inference_time_per_frame_ms",
        processing["average_inference_time_per_frame_ms"],
    ],

    ["inference_fps", processing["inference_fps"]],
    ["complete_pipeline_fps", processing["complete_pipeline_fps"]],

    # Ground truth dependent metrics
    ["precision", None],
    ["recall", None],
    ["f1_score", None],
    ["map50", None],
    ["map50_95", None],
    ["iou", None],
]

metrics_df = pd.DataFrame(
    metrics_rows,
    columns=["metric", "value"],
)

metrics_csv = os.path.join(
    METRICS_DIR,
    "yolo11m_pt_metrics.csv",
)

metrics_df.to_csv(
    metrics_csv,
    index=False,
)


# ============================================================
# 2. DETECTIONS PER FRAME
# ============================================================

detections_per_frame = (
    df.groupby("frame_number")
    .size()
    .reset_index(name="detection_count")
)

all_frames = pd.DataFrame(
    {
        "frame_number": range(
            1,
            int(metrics["input_video"]["total_frames_expected"]) + 1,
        )
    }
)

detections_per_frame = all_frames.merge(
    detections_per_frame,
    on="frame_number",
    how="left",
)

detections_per_frame["detection_count"] = (
    detections_per_frame["detection_count"]
    .fillna(0)
    .astype(int)
)

detections_per_frame["timestamp_seconds"] = (
    detections_per_frame["frame_number"] - 1
) / metrics["input_video"]["fps"]

detections_per_frame_csv = os.path.join(
    METRICS_DIR,
    "detections_per_frame.csv",
)

detections_per_frame.to_csv(
    detections_per_frame_csv,
    index=False,
)


# ============================================================
# 3. CLASS DISTRIBUTION
# ============================================================

class_distribution = (
    df.groupby(
        ["class_id", "class_name"]
    )
    .size()
    .reset_index(name="detection_count")
    .sort_values(
        "detection_count",
        ascending=False,
    )
)

class_distribution["percentage"] = (
    class_distribution["detection_count"]
    / len(df)
    * 100
)

class_distribution_csv = os.path.join(
    METRICS_DIR,
    "class_distribution.csv",
)

class_distribution.to_csv(
    class_distribution_csv,
    index=False,
)


# ============================================================
# 4. CONFIDENCE STATISTICS
# ============================================================

confidence_statistics = pd.DataFrame(
    {
        "statistic": [
            "count",
            "mean",
            "median",
            "std",
            "minimum",
            "25_percentile",
            "50_percentile",
            "75_percentile",
            "maximum",
        ],
        "value": [
            df["confidence"].count(),
            df["confidence"].mean(),
            df["confidence"].median(),
            df["confidence"].std(),
            df["confidence"].min(),
            df["confidence"].quantile(0.25),
            df["confidence"].quantile(0.50),
            df["confidence"].quantile(0.75),
            df["confidence"].max(),
        ],
    }
)

confidence_csv = os.path.join(
    METRICS_DIR,
    "confidence_statistics.csv",
)

confidence_statistics.to_csv(
    confidence_csv,
    index=False,
)


# ============================================================
# 5. EXPERIMENT CONFIGURATION
# ============================================================

experiment_config = {
    "model": "YOLO11m",
    "model_format": "PyTorch (.pt)",
    "input_video": metrics["input_video"]["filename"],
    "device": metrics["experiment"]["device"],
    "confidence_threshold": metrics["experiment"][
        "confidence_threshold"
    ],
    "iou_threshold": metrics["experiment"]["iou_threshold"],
    "image_size": metrics["experiment"]["image_size"],
    "video_width": metrics["input_video"]["width"],
    "video_height": metrics["input_video"]["height"],
    "video_fps": metrics["input_video"]["fps"],
    "total_frames": metrics["input_video"][
        "total_frames_expected"
    ],
    "evaluation_metrics": {
        "precision": None,
        "recall": None,
        "f1_score": None,
        "map50": None,
        "map50_95": None,
        "iou": None,
        "reason": (
            "Ground-truth annotations for the common video "
            "have not been established."
        ),
    },
}

config_json = os.path.join(
    METRICS_DIR,
    "experiment_config.json",
)

with open(
    config_json,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        experiment_config,
        file,
        indent=2,
    )


# ============================================================
# 6. CONFIDENCE DISTRIBUTION PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["confidence"],
    bins=20,
)

plt.xlabel("Confidence")
plt.ylabel("Number of Detections")
plt.title("YOLO11m PT - Confidence Distribution")
plt.grid(True, alpha=0.3)

confidence_plot = os.path.join(
    PLOTS_DIR,
    "confidence_distribution.png",
)

plt.savefig(
    confidence_plot,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 7. DETECTIONS PER FRAME PLOT
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    detections_per_frame["frame_number"],
    detections_per_frame["detection_count"],
)

plt.xlabel("Frame Number")
plt.ylabel("Number of Detections")
plt.title("YOLO11m PT - Detections Per Frame")
plt.grid(True, alpha=0.3)

detections_plot = os.path.join(
    PLOTS_DIR,
    "detections_per_frame.png",
)

plt.savefig(
    detections_plot,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 8. CLASS DISTRIBUTION PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    class_distribution["class_name"],
    class_distribution["detection_count"],
)

plt.xlabel("Class")
plt.ylabel("Number of Detections")
plt.title("YOLO11m PT - Detection Class Distribution")
plt.xticks(
    rotation=45,
    ha="right",
)

plt.grid(
    True,
    axis="y",
    alpha=0.3,
)

class_plot = os.path.join(
    PLOTS_DIR,
    "class_distribution.png",
)

plt.savefig(
    class_plot,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 9. DETAILED ANALYSIS REPORT
# ============================================================

report_path = os.path.join(
    METRICS_DIR,
    "yolo11m_pt_detailed_report.txt",
)

with open(
    report_path,
    "w",
    encoding="utf-8",
) as file:

    file.write(
        "YOLO11m PT DETAILED ANALYSIS REPORT\n"
    )

    file.write("=" * 70 + "\n\n")

    file.write("MODEL\n")
    file.write("-" * 70 + "\n")
    file.write("Model: YOLO11m\n")
    file.write("Format: PyTorch (.pt)\n")
    file.write("Device: CPU\n")
    file.write("Confidence threshold: 0.25\n")
    file.write("IoU threshold: 0.45\n")
    file.write("Image size: 640\n\n")

    file.write("VIDEO\n")
    file.write("-" * 70 + "\n")
    file.write(
        f"Filename: {metrics['input_video']['filename']}\n"
    )
    file.write(
        f"Resolution: "
        f"{metrics['input_video']['width']}x"
        f"{metrics['input_video']['height']}\n"
    )
    file.write(
        f"FPS: {metrics['input_video']['fps']}\n"
    )
    file.write(
        f"Frames: "
        f"{metrics['input_video']['total_frames_expected']}\n\n"
    )

    file.write("DETECTION RESULTS\n")
    file.write("-" * 70 + "\n")
    file.write(
        f"Total detections: "
        f"{processing['total_detections']}\n"
    )
    file.write(
        f"Average detections/frame: "
        f"{processing['total_detections'] / processing['frames_processed']:.4f}\n"
    )
    file.write(
        f"Average confidence: "
        f"{confidence['average']:.6f}\n"
    )
    file.write(
        f"Minimum confidence: "
        f"{confidence['minimum']:.6f}\n"
    )
    file.write(
        f"Maximum confidence: "
        f"{confidence['maximum']:.6f}\n\n"
    )

    file.write("PERFORMANCE\n")
    file.write("-" * 70 + "\n")
    file.write(
        f"Total processing time: "
        f"{processing['total_processing_time_seconds']:.4f} seconds\n"
    )
    file.write(
        f"Average inference/frame: "
        f"{processing['average_inference_time_per_frame_ms']:.4f} ms\n"
    )
    file.write(
        f"Inference FPS: "
        f"{processing['inference_fps']:.4f}\n"
    )
    file.write(
        f"Complete pipeline FPS: "
        f"{processing['complete_pipeline_fps']:.4f}\n\n"
    )

    file.write("CLASS DISTRIBUTION\n")
    file.write("-" * 70 + "\n")

    for _, row in class_distribution.iterrows():

        file.write(
            f"{row['class_name']}: "
            f"{int(row['detection_count'])} "
            f"({row['percentage']:.2f}%)\n"
        )

    file.write("\n")

    file.write("GROUND-TRUTH METRICS\n")
    file.write("-" * 70 + "\n")
    file.write(
        "Precision: Not calculated\n"
        "Recall: Not calculated\n"
        "F1-score: Not calculated\n"
        "mAP@50: Not calculated\n"
        "mAP@50-95: Not calculated\n"
        "IoU: Not calculated\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("PT ANALYSIS COMPLETE")
print("=" * 70)

print(f"Metrics CSV        : {metrics_csv}")
print(f"Detections/frame   : {detections_per_frame_csv}")
print(f"Class distribution : {class_distribution_csv}")
print(f"Confidence stats   : {confidence_csv}")
print(f"Experiment config  : {config_json}")
print(f"Detailed report    : {report_path}")

print()
print("Plots:")
print(f"  {confidence_plot}")
print(f"  {detections_plot}")
print(f"  {class_plot}")

print("=" * 70)