from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PT_METRICS = (
    PROJECT_ROOT
    / "outputs"
    / "pt"
    / "metrics"
    / "yolo11m_pt_metrics.json"
)

ONNX_METRICS = (
    PROJECT_ROOT
    / "outputs"
    / "onnx"
    / "metrics"
    / "yolo11m_onnx_metrics.json"
)

MCB_METRICS = (
    PROJECT_ROOT
    / "outputs"
    / "mcbytetrack"
    / "metrics"
    / "mcbytetrack_metrics.json"
)

MCB_ANALYSIS = (
    PROJECT_ROOT
    / "outputs"
    / "mcbytetrack"
    / "metrics"
    / "mcbytetrack_analysis_metrics.json"
)

COMPARISON_DIR = PROJECT_ROOT / "comparison"
PLOTS_DIR = COMPARISON_DIR / "plots"


# ============================================================
# CREATE DIRECTORIES
# ============================================================

COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


pt = load_json(PT_METRICS)
onnx = load_json(ONNX_METRICS)
mcb = load_json(MCB_METRICS)
mcb_analysis = load_json(MCB_ANALYSIS)


# ============================================================
# EXTRACT METRICS
# ============================================================

def extract_experiment(metrics):
    return metrics.get("experiment", {})


def extract_input(metrics):
    return metrics.get("input_video", {})


def extract_processing(metrics):
    return metrics.get("processing", {})


def extract_confidence(metrics):
    return metrics.get("confidence", {})


pt_exp = extract_experiment(pt)
onnx_exp = extract_experiment(onnx)
mcb_exp = extract_experiment(mcb)

pt_input = extract_input(pt)
onnx_input = extract_input(onnx)
mcb_input = extract_input(mcb)

pt_proc = extract_processing(pt)
onnx_proc = extract_processing(onnx)
mcb_proc = extract_processing(mcb)

pt_conf = extract_confidence(pt)
onnx_conf = extract_confidence(onnx)
mcb_conf = extract_confidence(mcb)

mcb_tracking = mcb.get("tracking", {})


# ============================================================
# MAIN COMPARISON TABLE
# ============================================================

comparison_rows = [
    {
        "metric": "Frames processed",
        "YOLO11m PT": pt_proc.get("frames_processed"),
        "YOLO11m ONNX": onnx_proc.get("frames_processed"),
        "YOLO11m + MCByteTrack": mcb_proc.get("frames_processed"),
        "unit": "frames",
    },

    {
        "metric": "Total detections / tracking records",
        "YOLO11m PT": pt_proc.get("total_detections"),
        "YOLO11m ONNX": onnx_proc.get("total_detections"),
        "YOLO11m + MCByteTrack": mcb_proc.get(
            "total_tracking_records"
        ),
        "unit": "records",
    },

    {
        "metric": "Average confidence",
        "YOLO11m PT": pt_conf.get("average"),
        "YOLO11m ONNX": onnx_conf.get("average"),
        "YOLO11m + MCByteTrack": mcb_conf.get("average"),
        "unit": "score",
    },

    {
        "metric": "Total processing time",
        "YOLO11m PT": pt_proc.get(
            "total_processing_time_seconds"
        ),
        "YOLO11m ONNX": onnx_proc.get(
            "total_pipeline_time_seconds"
        ),
        "YOLO11m + MCByteTrack": mcb_proc.get(
            "total_processing_time_seconds"
        ),
        "unit": "seconds",
    },

    {
        "metric": "Average inference time/frame",
        "YOLO11m PT": pt_proc.get(
            "average_inference_time_per_frame_ms"
        ),
        "YOLO11m ONNX": onnx_proc.get(
            "average_inference_time_per_frame_ms"
        ),
        "YOLO11m + MCByteTrack": mcb_proc.get(
            "average_inference_time_per_frame_ms"
        ),
        "unit": "ms/frame",
    },

    {
        "metric": "Inference FPS",
        "YOLO11m PT": pt_proc.get("inference_fps"),
        "YOLO11m ONNX": onnx_proc.get("inference_fps"),
        "YOLO11m + MCByteTrack": mcb_proc.get("inference_fps"),
        "unit": "FPS",
    },

    {
        "metric": "Complete pipeline FPS",
        "YOLO11m PT": pt_proc.get(
            "complete_pipeline_fps"
        ),
        "YOLO11m ONNX": onnx_proc.get(
            "complete_pipeline_fps"
        ),
        "YOLO11m + MCByteTrack": mcb_proc.get(
            "complete_pipeline_fps"
        ),
        "unit": "FPS",
    },

    {
        "metric": "Unique track IDs",
        "YOLO11m PT": None,
        "YOLO11m ONNX": None,
        "YOLO11m + MCByteTrack": mcb_tracking.get(
            "total_unique_track_ids"
        ),
        "unit": "tracks",
    },

    {
        "metric": "Average tracks/frame",
        "YOLO11m PT": None,
        "YOLO11m ONNX": None,
        "YOLO11m + MCByteTrack": mcb_tracking.get(
            "average_tracks_per_frame"
        ),
        "unit": "tracks/frame",
    },

    {
        "metric": "Average track length",
        "YOLO11m PT": None,
        "YOLO11m ONNX": None,
        "YOLO11m + MCByteTrack": mcb_tracking.get(
            "average_track_length_frames"
        ),
        "unit": "frames",
    },

    {
        "metric": "Minimum track length",
        "YOLO11m PT": None,
        "YOLO11m ONNX": None,
        "YOLO11m + MCByteTrack": mcb_tracking.get(
            "minimum_track_length_frames"
        ),
        "unit": "frames",
    },

    {
        "metric": "Maximum track length",
        "YOLO11m PT": None,
        "YOLO11m ONNX": None,
        "YOLO11m + MCByteTrack": mcb_tracking.get(
            "maximum_track_length_frames"
        ),
        "unit": "frames",
    },
]


comparison_df = pd.DataFrame(comparison_rows)


# ============================================================
# SAVE MAIN CSV
# ============================================================

comparison_df.to_csv(
    COMPARISON_DIR / "final_comparison.csv",
    index=False,
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

pt_classes = pt.get("detections_by_class", {})
onnx_classes = onnx.get("detections_by_class", {})

mcb_classes = mcb_tracking.get(
    "tracks_by_class",
    {},
)

all_classes = sorted(
    set(pt_classes)
    | set(onnx_classes)
    | set(mcb_classes)
)

class_rows = []

for class_name in all_classes:

    class_rows.append(
        {
            "class_name": class_name,
            "PT_detections": pt_classes.get(
                class_name,
                0,
            ),
            "ONNX_detections": onnx_classes.get(
                class_name,
                0,
            ),
            "MCByteTrack_unique_tracks": mcb_classes.get(
                class_name,
                0,
            ),
        }
    )


class_df = pd.DataFrame(class_rows)

class_df.to_csv(
    COMPARISON_DIR / "class_comparison.csv",
    index=False,
)


# ============================================================
# GROUND-TRUTH METRICS
# ============================================================

ground_truth_metrics = pd.DataFrame(
    [
        {
            "metric": "Precision",
            "PT": "Ground truth required",
            "ONNX": "Ground truth required",
            "MCByteTrack": "Ground truth required",
        },
        {
            "metric": "Recall",
            "PT": "Ground truth required",
            "ONNX": "Ground truth required",
            "MCByteTrack": "Ground truth required",
        },
        {
            "metric": "F1-score",
            "PT": "Ground truth required",
            "ONNX": "Ground truth required",
            "MCByteTrack": "Ground truth required",
        },
        {
            "metric": "mAP@50",
            "PT": "Ground truth required",
            "ONNX": "Ground truth required",
            "MCByteTrack": "Ground truth required",
        },
        {
            "metric": "mAP@50-95",
            "PT": "Ground truth required",
            "ONNX": "Ground truth required",
            "MCByteTrack": "Ground truth required",
        },
        {
            "metric": "IoU",
            "PT": "Ground truth required",
            "ONNX": "Ground truth required",
            "MCByteTrack": "Ground truth required",
        },
        {
            "metric": "MOTA",
            "PT": "Not applicable",
            "ONNX": "Not applicable",
            "MCByteTrack": "Ground truth required",
        },
        {
            "metric": "MOTP",
            "PT": "Not applicable",
            "ONNX": "Not applicable",
            "MCByteTrack": "Ground truth required",
        },
        {
            "metric": "IDF1",
            "PT": "Not applicable",
            "ONNX": "Not applicable",
            "MCByteTrack": "Ground truth required",
        },
        {
            "metric": "HOTA",
            "PT": "Not applicable",
            "ONNX": "Not applicable",
            "MCByteTrack": "Ground truth required",
        },
    ]
)

ground_truth_metrics.to_csv(
    COMPARISON_DIR / "ground_truth_metrics.csv",
    index=False,
)


# ============================================================
# PERFORMANCE DATA
# ============================================================

performance_df = pd.DataFrame(
    [
        {
            "method": "YOLO11m PT",
            "average_inference_time_ms": pt_proc.get(
                "average_inference_time_per_frame_ms"
            ),
            "inference_fps": pt_proc.get(
                "inference_fps"
            ),
            "pipeline_fps": pt_proc.get(
                "complete_pipeline_fps"
            ),
        },
        {
            "method": "YOLO11m ONNX",
            "average_inference_time_ms": onnx_proc.get(
                "average_inference_time_per_frame_ms"
            ),
            "inference_fps": onnx_proc.get(
                "inference_fps"
            ),
            "pipeline_fps": onnx_proc.get(
                "complete_pipeline_fps"
            ),
        },
        {
            "method": "YOLO11m + MCByteTrack",
            "average_inference_time_ms": mcb_proc.get(
                "average_inference_time_per_frame_ms"
            ),
            "inference_fps": mcb_proc.get(
                "inference_fps"
            ),
            "pipeline_fps": mcb_proc.get(
                "complete_pipeline_fps"
            ),
        },
    ]
)

performance_df.to_csv(
    COMPARISON_DIR / "performance_comparison.csv",
    index=False,
)


# ============================================================
# PLOT 1 — RECORD COUNT
# ============================================================

methods = [
    "YOLO11m PT",
    "YOLO11m ONNX",
    "YOLO11m + MCByteTrack",
]

record_values = [
    pt_proc.get("total_detections"),
    onnx_proc.get("total_detections"),
    mcb_proc.get("total_tracking_records"),
]

plt.figure(figsize=(10, 6))

plt.bar(
    methods,
    record_values,
)

plt.ylabel("Records")
plt.title("Detection / Tracking Record Count")
plt.xticks(rotation=15)
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "record_count_comparison.png",
    dpi=150,
)

plt.close()


# ============================================================
# PLOT 2 — CONFIDENCE
# ============================================================

confidence_values = [
    pt_conf.get("average"),
    onnx_conf.get("average"),
    mcb_conf.get("average"),
]

plt.figure(figsize=(10, 6))

plt.bar(
    methods,
    confidence_values,
)

plt.ylabel("Average Confidence")
plt.title("Average Confidence Comparison")
plt.xticks(rotation=15)
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "confidence_comparison.png",
    dpi=150,
)

plt.close()


# ============================================================
# PLOT 3 — INFERENCE TIME
# ============================================================

inference_time_values = [
    pt_proc.get(
        "average_inference_time_per_frame_ms"
    ),
    onnx_proc.get(
        "average_inference_time_per_frame_ms"
    ),
    mcb_proc.get(
        "average_inference_time_per_frame_ms"
    ),
]

plt.figure(figsize=(10, 6))

plt.bar(
    methods,
    inference_time_values,
)

plt.ylabel("Milliseconds / Frame")
plt.title("Average Inference Time Comparison")
plt.xticks(rotation=15)
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "inference_time_comparison.png",
    dpi=150,
)

plt.close()


# ============================================================
# PLOT 4 — FPS
# ============================================================

fps_values = [
    pt_proc.get("inference_fps"),
    onnx_proc.get("inference_fps"),
    mcb_proc.get("inference_fps"),
]

plt.figure(figsize=(10, 6))

plt.bar(
    methods,
    fps_values,
)

plt.ylabel("FPS")
plt.title("Inference FPS Comparison")
plt.xticks(rotation=15)
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "fps_comparison.png",
    dpi=150,
)

plt.close()


# ============================================================
# PLOT 5 — PIPELINE FPS
# ============================================================

pipeline_fps_values = [
    pt_proc.get("complete_pipeline_fps"),
    onnx_proc.get("complete_pipeline_fps"),
    mcb_proc.get("complete_pipeline_fps"),
]

plt.figure(figsize=(10, 6))

plt.bar(
    methods,
    pipeline_fps_values,
)

plt.ylabel("Pipeline FPS")
plt.title("Complete Pipeline FPS Comparison")
plt.xticks(rotation=15)
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "pipeline_fps_comparison.png",
    dpi=150,
)

plt.close()


# ============================================================
# JSON SUMMARY
# ============================================================

summary = {
    "experiment": {
        "video": "PNNL_Parking_LOT(1).avi",
        "frames": 998,
        "model": "YOLO11m",
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "image_size": 640,
        "device": "CPU",
    },

    "methods": {
        "YOLO11m_PT": {
            "total_detections": pt_proc.get(
                "total_detections"
            ),
            "average_confidence": pt_conf.get(
                "average"
            ),
            "total_processing_time_seconds": pt_proc.get(
                "total_processing_time_seconds"
            ),
            "average_inference_time_ms": pt_proc.get(
                "average_inference_time_per_frame_ms"
            ),
            "inference_fps": pt_proc.get(
                "inference_fps"
            ),
            "pipeline_fps": pt_proc.get(
                "complete_pipeline_fps"
            ),
        },

        "YOLO11m_ONNX": {
            "total_detections": onnx_proc.get(
                "total_detections"
            ),
            "average_confidence": onnx_conf.get(
                "average"
            ),
            "total_inference_time_seconds": onnx_proc.get(
                "total_inference_time_seconds"
            ),
            "total_pipeline_time_seconds": onnx_proc.get(
                "total_pipeline_time_seconds"
            ),
            "average_inference_time_ms": onnx_proc.get(
                "average_inference_time_per_frame_ms"
            ),
            "inference_fps": onnx_proc.get(
                "inference_fps"
            ),
            "pipeline_fps": onnx_proc.get(
                "complete_pipeline_fps"
            ),
        },

        "YOLO11m_MCByteTrack": {
            "total_tracking_records": mcb_proc.get(
                "total_tracking_records"
            ),
            "average_confidence": mcb_conf.get(
                "average"
            ),
            "total_unique_track_ids": mcb_tracking.get(
                "total_unique_track_ids"
            ),
            "person_tracks": mcb_tracking.get(
                "tracks_by_class", {}
            ).get("person"),
            "car_tracks": mcb_tracking.get(
                "tracks_by_class", {}
            ).get("car"),
            "average_tracks_per_frame": mcb_tracking.get(
                "average_tracks_per_frame"
            ),
            "average_track_length_frames": mcb_tracking.get(
                "average_track_length_frames"
            ),
            "minimum_track_length_frames": mcb_tracking.get(
                "minimum_track_length_frames"
            ),
            "maximum_track_length_frames": mcb_tracking.get(
                "maximum_track_length_frames"
            ),
            "total_processing_time_seconds": mcb_proc.get(
                "total_processing_time_seconds"
            ),
            "average_inference_time_ms": mcb_proc.get(
                "average_inference_time_per_frame_ms"
            ),
            "inference_fps": mcb_proc.get(
                "inference_fps"
            ),
            "pipeline_fps": mcb_proc.get(
                "complete_pipeline_fps"
            ),
        },
    },

    "ground_truth_note": (
        "Precision, recall, F1-score, mAP, IoU, MOTA, "
        "MOTP, IDF1 and HOTA require ground-truth annotations."
    ),
}


with open(
    COMPARISON_DIR / "final_comparison.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# TEXT REPORT
# ============================================================

report = []

report.append("YOLO11m PT vs ONNX vs MCByteTrack")
report.append("=" * 80)
report.append("")

report.append("EXPERIMENT CONFIGURATION")
report.append("-" * 80)
report.append("Video: PNNL_Parking_LOT(1).avi")
report.append("Frames: 998")
report.append("Model: YOLO11m")
report.append("Confidence threshold: 0.25")
report.append("IoU threshold: 0.45")
report.append("Image size: 640")
report.append("Device: CPU")
report.append("")

report.append("PERFORMANCE COMPARISON")
report.append("-" * 80)

for _, row in performance_df.iterrows():

    report.append(
        f"{row['method']}: "
        f"{row['average_inference_time_ms']:.2f} ms/frame | "
        f"Inference FPS: {row['inference_fps']:.2f} | "
        f"Pipeline FPS: {row['pipeline_fps']:.2f}"
    )

report.append("")

report.append("DETECTION / TRACKING RECORDS")
report.append("-" * 80)
report.append(
    f"YOLO11m PT: {pt_proc.get('total_detections')}"
)
report.append(
    f"YOLO11m ONNX: {onnx_proc.get('total_detections')}"
)
report.append(
    f"MCByteTrack tracking records: "
    f"{mcb_proc.get('total_tracking_records')}"
)

report.append("")

report.append("CONFIDENCE")
report.append("-" * 80)
report.append(
    f"YOLO11m PT average: "
    f"{pt_conf.get('average'):.6f}"
)
report.append(
    f"YOLO11m ONNX average: "
    f"{onnx_conf.get('average'):.6f}"
)
report.append(
    f"MCByteTrack average: "
    f"{mcb_conf.get('average'):.6f}"
)

report.append("")

report.append("MCByteTrack TRACKING")
report.append("-" * 80)
report.append(
    f"Unique track IDs: "
    f"{mcb_tracking.get('total_unique_track_ids')}"
)
report.append(
    f"Person tracks: "
    f"{mcb_tracking.get('tracks_by_class', {}).get('person')}"
)
report.append(
    f"Car tracks: "
    f"{mcb_tracking.get('tracks_by_class', {}).get('car')}"
)
report.append(
    f"Average tracks/frame: "
    f"{mcb_tracking.get('average_tracks_per_frame'):.4f}"
)
report.append(
    f"Average track length: "
    f"{mcb_tracking.get('average_track_length_frames'):.2f} frames"
)
report.append(
    f"Minimum track length: "
    f"{mcb_tracking.get('minimum_track_length_frames')} frames"
)
report.append(
    f"Maximum track length: "
    f"{mcb_tracking.get('maximum_track_length_frames')} frames"
)

report.append("")

report.append("GROUND-TRUTH DEPENDENT METRICS")
report.append("-" * 80)
report.append("Precision: Ground truth required")
report.append("Recall: Ground truth required")
report.append("F1-score: Ground truth required")
report.append("mAP@50: Ground truth required")
report.append("mAP@50-95: Ground truth required")
report.append("IoU: Ground truth required")
report.append("MOTA: Ground truth required")
report.append("MOTP: Ground truth required")
report.append("IDF1: Ground truth required")
report.append("HOTA: Ground truth required")

report.append("")

report.append("INTERPRETATION NOTES")
report.append("-" * 80)
report.append(
    "PT and ONNX use the same YOLO11m model and common video."
)
report.append(
    "Small detection differences between PT and ONNX are "
    "reported rather than treated as identical."
)
report.append(
    "MCByteTrack adds temporal object association and track IDs."
)
report.append(
    "Tracking statistics are not equivalent to ground-truth "
    "tracking accuracy."
)
report.append(
    "Ground-truth annotations are required for formal detection "
    "and multi-object tracking evaluation metrics."
)


with open(
    COMPARISON_DIR / "final_comparison_report.txt",
    "w",
    encoding="utf-8",
) as f:

    f.write("\n".join(report))


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 80)
print("FINAL COMPARISON COMPLETED")
print("=" * 80)

print()
print("Generated files:")

for path in sorted(COMPARISON_DIR.rglob("*")):

    if path.is_file():

        print(
            f"  {path.relative_to(PROJECT_ROOT)}"
        )

print()
print("Main comparison:")
print(COMPARISON_DIR / "final_comparison.csv")

print()
print("Final report:")
print(COMPARISON_DIR / "final_comparison_report.txt")

print()
print("Final JSON:")
print(COMPARISON_DIR / "final_comparison.json")