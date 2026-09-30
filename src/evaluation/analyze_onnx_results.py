import os
import json
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DETECTIONS_CSV = os.path.join(
    BASE_DIR,
    "outputs",
    "onnx",
    "detections",
    "yolo11m_onnx_detections.csv"
)

METRICS_JSON = os.path.join(
    BASE_DIR,
    "outputs",
    "onnx",
    "metrics",
    "yolo11m_onnx_metrics.json"
)

METRICS_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "onnx",
    "metrics"
)

PLOTS_DIR = os.path.join(
    METRICS_DIR,
    "plots"
)

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("YOLO11m ONNX RESULT ANALYSIS")
print("=" * 70)

print(f"Reading detections:")
print(DETECTIONS_CSV)

df = pd.read_csv(DETECTIONS_CSV)

with open(METRICS_JSON, "r", encoding="utf-8") as f:
    metrics = json.load(f)


# ============================================================
# BASIC INFORMATION
# ============================================================

total_frames = int(
    metrics["processing"]["frames_processed"]
)

total_detections = len(df)

frames_with_detections = int(
    df["frame_number"].nunique()
)

average_detections_per_frame = (
    total_detections / total_frames
    if total_frames > 0
    else 0
)

average_confidence = float(
    df["confidence"].mean()
)

minimum_confidence = float(
    df["confidence"].min()
)

maximum_confidence = float(
    df["confidence"].max()
)


# ============================================================
# CONFIDENCE STATISTICS
# ============================================================

confidence_statistics = pd.DataFrame({
    "metric": [
        "mean",
        "median",
        "std",
        "minimum",
        "maximum",
        "25_percentile",
        "50_percentile",
        "75_percentile",
        "90_percentile",
        "95_percentile"
    ],
    "value": [
        df["confidence"].mean(),
        df["confidence"].median(),
        df["confidence"].std(),
        df["confidence"].min(),
        df["confidence"].max(),
        df["confidence"].quantile(0.25),
        df["confidence"].quantile(0.50),
        df["confidence"].quantile(0.75),
        df["confidence"].quantile(0.90),
        df["confidence"].quantile(0.95)
    ]
})

confidence_statistics.to_csv(
    os.path.join(
        METRICS_DIR,
        "confidence_statistics.csv"
    ),
    index=False
)


# ============================================================
# DETECTIONS PER FRAME
# ============================================================

detections_per_frame = (
    df.groupby("frame_number")
    .size()
    .reset_index(name="detection_count")
)

all_frames = pd.DataFrame({
    "frame_number": range(1, total_frames + 1)
})

detections_per_frame = all_frames.merge(
    detections_per_frame,
    on="frame_number",
    how="left"
)

detections_per_frame["detection_count"] = (
    detections_per_frame["detection_count"]
    .fillna(0)
    .astype(int)
)

detections_per_frame.to_csv(
    os.path.join(
        METRICS_DIR,
        "detections_per_frame.csv"
    ),
    index=False
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

class_distribution = (
    df.groupby(
        ["class_id", "class_name"]
    )
    .size()
    .reset_index(name="detection_count")
    .sort_values(
        "detection_count",
        ascending=False
    )
)

class_distribution["percentage"] = (
    class_distribution["detection_count"]
    / total_detections
    * 100
)

class_distribution.to_csv(
    os.path.join(
        METRICS_DIR,
        "class_distribution.csv"
    ),
    index=False
)


# ============================================================
# MAIN METRICS CSV
# ============================================================

processing = metrics["processing"]
confidence = metrics["confidence"]
evaluation = metrics["evaluation_metrics"]

metrics_table = pd.DataFrame([
    {
        "model": "YOLO11m",
        "format": "ONNX",
        "device": metrics["experiment"]["execution_provider"],
        "confidence_threshold":
            metrics["experiment"]["confidence_threshold"],
        "iou_threshold":
            metrics["experiment"]["iou_threshold"],
        "image_size":
            metrics["experiment"]["image_size"],

        "total_frames":
            total_frames,

        "frames_with_detections":
            frames_with_detections,

        "total_detections":
            total_detections,

        "average_detections_per_frame":
            average_detections_per_frame,

        "average_confidence":
            confidence["average"],

        "minimum_confidence":
            confidence["minimum"],

        "maximum_confidence":
            confidence["maximum"],

        "total_inference_time_seconds":
            processing["total_inference_time_seconds"],

        "total_pipeline_time_seconds":
            processing["total_pipeline_time_seconds"],

        "average_inference_time_per_frame_ms":
            processing[
                "average_inference_time_per_frame_ms"
            ],

        "inference_fps":
            processing["inference_fps"],

        "complete_pipeline_fps":
            processing["complete_pipeline_fps"],

        "precision":
            evaluation["precision"],

        "recall":
            evaluation["recall"],

        "f1_score":
            evaluation["f1_score"],

        "map50":
            evaluation["map50"],

        "map50_95":
            evaluation["map50_95"],

        "iou":
            evaluation["iou"]
    }
])

metrics_table.to_csv(
    os.path.join(
        METRICS_DIR,
        "yolo11m_onnx_metrics.csv"
    ),
    index=False
)


# ============================================================
# EXPERIMENT CONFIG
# ============================================================

experiment_config = {
    "model": "YOLO11m",
    "model_format": "ONNX",
    "runtime": "ONNX Runtime",
    "execution_provider": "CPUExecutionProvider",
    "video": metrics["input_video"],
    "confidence_threshold":
        metrics["experiment"]["confidence_threshold"],
    "iou_threshold":
        metrics["experiment"]["iou_threshold"],
    "image_size":
        metrics["experiment"]["image_size"],
    "evaluation_metrics_status": {
        "precision": "Ground truth required",
        "recall": "Ground truth required",
        "f1_score": "Ground truth required",
        "map50": "Ground truth required",
        "map50_95": "Ground truth required",
        "iou": "Ground truth required"
    }
}

with open(
    os.path.join(
        METRICS_DIR,
        "experiment_config.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        experiment_config,
        f,
        indent=2
    )


# ============================================================
# DETAILED REPORT
# ============================================================

report_path = os.path.join(
    METRICS_DIR,
    "yolo11m_onnx_detailed_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "YOLO11m ONNX DETAILED REPORT\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write("MODEL\n")
    f.write("-" * 70 + "\n")
    f.write("Model: YOLO11m\n")
    f.write("Format: ONNX\n")
    f.write("Runtime: ONNX Runtime\n")
    f.write("Device: CPU\n\n")

    f.write("INPUT VIDEO\n")
    f.write("-" * 70 + "\n")
    f.write(
        f"Filename: "
        f"{metrics['input_video']['filename']}\n"
    )
    f.write(
        f"Resolution: "
        f"{metrics['input_video']['width']}x"
        f"{metrics['input_video']['height']}\n"
    )
    f.write(
        f"Video FPS: "
        f"{metrics['input_video']['fps']}\n"
    )
    f.write(
        f"Total frames: {total_frames}\n\n"
    )

    f.write("DETECTION RESULTS\n")
    f.write("-" * 70 + "\n")
    f.write(
        f"Total detections: "
        f"{total_detections}\n"
    )
    f.write(
        f"Frames with detections: "
        f"{frames_with_detections}\n"
    )
    f.write(
        f"Average detections/frame: "
        f"{average_detections_per_frame:.4f}\n"
    )
    f.write(
        f"Average confidence: "
        f"{average_confidence:.6f}\n"
    )
    f.write(
        f"Minimum confidence: "
        f"{minimum_confidence:.6f}\n"
    )
    f.write(
        f"Maximum confidence: "
        f"{maximum_confidence:.6f}\n\n"
    )

    f.write("PERFORMANCE\n")
    f.write("-" * 70 + "\n")
    f.write(
        f"Total inference time: "
        f"{processing['total_inference_time_seconds']:.4f} sec\n"
    )
    f.write(
        f"Total pipeline time: "
        f"{processing['total_pipeline_time_seconds']:.4f} sec\n"
    )
    f.write(
        f"Average inference/frame: "
        f"{processing['average_inference_time_per_frame_ms']:.4f} ms\n"
    )
    f.write(
        f"Inference FPS: "
        f"{processing['inference_fps']:.4f}\n"
    )
    f.write(
        f"Pipeline FPS: "
        f"{processing['complete_pipeline_fps']:.4f}\n\n"
    )

    f.write("DETECTIONS BY CLASS\n")
    f.write("-" * 70 + "\n")

    for _, row in class_distribution.iterrows():

        f.write(
            f"{row['class_name']}: "
            f"{int(row['detection_count'])} "
            f"({row['percentage']:.2f}%)\n"
        )

    f.write("\n")

    f.write("EVALUATION METRICS\n")
    f.write("-" * 70 + "\n")
    f.write(
        "Precision: N/A - ground truth required\n"
    )
    f.write(
        "Recall: N/A - ground truth required\n"
    )
    f.write(
        "F1-score: N/A - ground truth required\n"
    )
    f.write(
        "mAP@50: N/A - ground truth required\n"
    )
    f.write(
        "mAP@50-95: N/A - ground truth required\n"
    )
    f.write(
        "IoU: N/A - ground truth required\n"
    )


# ============================================================
# PLOT 1 — CONFIDENCE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["confidence"],
    bins=30
)

plt.xlabel("Confidence")
plt.ylabel("Number of Detections")
plt.title(
    "YOLO11m ONNX Confidence Distribution"
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "confidence_distribution.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# PLOT 2 — DETECTIONS PER FRAME
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    detections_per_frame["frame_number"],
    detections_per_frame["detection_count"]
)

plt.xlabel("Frame Number")
plt.ylabel("Number of Detections")
plt.title(
    "YOLO11m ONNX Detections per Frame"
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "detections_per_frame.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# PLOT 3 — CLASS DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    class_distribution["class_name"],
    class_distribution["detection_count"]
)

plt.xlabel("Class")
plt.ylabel("Detection Count")
plt.title(
    "YOLO11m ONNX Class Distribution"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "class_distribution.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("ONNX RESULT ANALYSIS COMPLETED")
print("=" * 70)

print(f"Total frames             : {total_frames}")
print(f"Total detections         : {total_detections}")
print(
    f"Average detections/frame : "
    f"{average_detections_per_frame:.2f}"
)
print(
    f"Average confidence       : "
    f"{average_confidence:.4f}"
)

print()
print("Generated files:")
print(
    os.path.join(
        METRICS_DIR,
        "yolo11m_onnx_metrics.csv"
    )
)
print(
    os.path.join(
        METRICS_DIR,
        "detections_per_frame.csv"
    )
)
print(
    os.path.join(
        METRICS_DIR,
        "class_distribution.csv"
    )
)
print(
    os.path.join(
        METRICS_DIR,
        "confidence_statistics.csv"
    )
)
print(
    os.path.join(
        METRICS_DIR,
        "experiment_config.json"
    )
)
print(report_path)
print(PLOTS_DIR)

print("=" * 70)