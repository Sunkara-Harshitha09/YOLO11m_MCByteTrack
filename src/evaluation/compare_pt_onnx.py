import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

PT_CSV = os.path.join(
    BASE_DIR,
    "outputs",
    "pt",
    "detections",
    "yolo11m_pt_detections.csv"
)

ONNX_CSV = os.path.join(
    BASE_DIR,
    "outputs",
    "onnx",
    "detections",
    "yolo11m_onnx_detections.csv"
)

PT_METRICS = os.path.join(
    BASE_DIR,
    "outputs",
    "pt",
    "metrics",
    "yolo11m_pt_metrics.json"
)

ONNX_METRICS = os.path.join(
    BASE_DIR,
    "outputs",
    "onnx",
    "metrics",
    "yolo11m_onnx_metrics.json"
)

COMPARISON_DIR = os.path.join(
    BASE_DIR,
    "comparison"
)

PLOTS_DIR = os.path.join(
    COMPARISON_DIR,
    "plots"
)

os.makedirs(COMPARISON_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

# IoU used to decide whether a PT detection and ONNX detection
# represent the same detection.
MATCH_IOU_THRESHOLD = 0.50


# ============================================================
# IOU FUNCTION
# ============================================================

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union.

    box format:
    [x1, y1, x2, y2]
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0.0, x2 - x1)
    intersection_height = max(0.0, y2 - y1)

    intersection_area = (
        intersection_width * intersection_height
    )

    area1 = (
        max(0.0, box1[2] - box1[0])
        *
        max(0.0, box1[3] - box1[1])
    )

    area2 = (
        max(0.0, box2[2] - box2[0])
        *
        max(0.0, box2[3] - box2[1])
    )

    union_area = (
        area1
        + area2
        - intersection_area
    )

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("YOLO11m PT vs ONNX COMPARISON")
print("=" * 70)

print()
print("Loading PT detections...")

pt_df = pd.read_csv(PT_CSV)

print(
    f"PT detection records: {len(pt_df)}"
)

print()
print("Loading ONNX detections...")

onnx_df = pd.read_csv(ONNX_CSV)

print(
    f"ONNX detection records: {len(onnx_df)}"
)


# ============================================================
# LOAD METRICS
# ============================================================

with open(
    PT_METRICS,
    "r",
    encoding="utf-8"
) as f:

    pt_metrics = json.load(f)


with open(
    ONNX_METRICS,
    "r",
    encoding="utf-8"
) as f:

    onnx_metrics = json.load(f)


# ============================================================
# BASIC INFORMATION
# ============================================================

pt_total = len(pt_df)

onnx_total = len(onnx_df)

pt_frames = int(
    pt_metrics["processing"]["frames_processed"]
)

onnx_frames = int(
    onnx_metrics["processing"]["frames_processed"]
)

total_frames = min(
    pt_frames,
    onnx_frames
)

print()
print(f"PT frames: {pt_frames}")
print(f"ONNX frames: {onnx_frames}")
print(f"Frames compared: {total_frames}")


# ============================================================
# DETECTION COUNT COMPARISON
# ============================================================

detection_difference = (
    onnx_total - pt_total
)

detection_difference_percent = (
    detection_difference
    / pt_total
    * 100
    if pt_total > 0
    else 0
)


# ============================================================
# CLASS COMPARISON
# ============================================================

pt_class = (
    pt_df
    .groupby("class_name")
    .size()
    .rename("pt_detections")
)

onnx_class = (
    onnx_df
    .groupby("class_name")
    .size()
    .rename("onnx_detections")
)

class_comparison = pd.concat(
    [
        pt_class,
        onnx_class
    ],
    axis=1
).fillna(0)

class_comparison[
    "pt_detections"
] = (
    class_comparison[
        "pt_detections"
    ].astype(int)
)

class_comparison[
    "onnx_detections"
] = (
    class_comparison[
        "onnx_detections"
    ].astype(int)
)

class_comparison["difference"] = (
    class_comparison["onnx_detections"]
    -
    class_comparison["pt_detections"]
)

class_comparison[
    "difference_percent"
] = np.where(
    class_comparison["pt_detections"] > 0,

    (
        class_comparison["difference"]
        /
        class_comparison["pt_detections"]
        *
        100
    ),

    np.nan
)

class_comparison = (
    class_comparison
    .reset_index()
    .rename(
        columns={
            "index": "class_name"
        }
    )
)

class_comparison_path = os.path.join(
    COMPARISON_DIR,
    "pt_vs_onnx_class_comparison.csv"
)

class_comparison.to_csv(
    class_comparison_path,
    index=False
)


# ============================================================
# PER-FRAME COMPARISON
# ============================================================

pt_frame_counts = (
    pt_df
    .groupby("frame_number")
    .size()
    .rename("pt_detections")
)

onnx_frame_counts = (
    onnx_df
    .groupby("frame_number")
    .size()
    .rename("onnx_detections")
)

frame_comparison = pd.concat(
    [
        pt_frame_counts,
        onnx_frame_counts
    ],
    axis=1
).fillna(0)

frame_comparison[
    "pt_detections"
] = (
    frame_comparison[
        "pt_detections"
    ].astype(int)
)

frame_comparison[
    "onnx_detections"
] = (
    frame_comparison[
        "onnx_detections"
    ].astype(int)
)

frame_comparison["difference"] = (
    frame_comparison["onnx_detections"]
    -
    frame_comparison["pt_detections"]
)

frame_comparison[
    "absolute_difference"
] = (
    frame_comparison["difference"]
    .abs()
)

frame_comparison = (
    frame_comparison
    .reset_index()
)

frame_comparison_path = os.path.join(
    COMPARISON_DIR,
    "pt_vs_onnx_frame_comparison.csv"
)

frame_comparison.to_csv(
    frame_comparison_path,
    index=False
)


# ============================================================
# DETECTION MATCHING
# ============================================================

print()
print("=" * 70)
print("MATCHING PT AND ONNX DETECTIONS")
print("=" * 70)

print(
    f"Matching IoU threshold: "
    f"{MATCH_IOU_THRESHOLD}"
)

matched_records = []

pt_only_count = 0
onnx_only_count = 0


# ============================================================
# GROUP BY FRAME
# ============================================================

pt_groups = {
    frame: group
    for frame, group
    in pt_df.groupby("frame_number")
}

onnx_groups = {
    frame: group
    for frame, group
    in onnx_df.groupby("frame_number")
}


# ============================================================
# FRAME-BY-FRAME MATCHING
# ============================================================

for frame_number in range(
    1,
    total_frames + 1
):

    pt_frame = pt_groups.get(
        frame_number,
        pd.DataFrame()
    )

    onnx_frame = onnx_groups.get(
        frame_number,
        pd.DataFrame()
    )

    used_onnx_indices = set()

    # --------------------------------------------------------
    # MATCH PT DETECTIONS
    # --------------------------------------------------------

    for pt_index, pt_row in pt_frame.iterrows():

        best_match_index = None
        best_iou = 0.0

        pt_box = [
            float(pt_row["x1"]),
            float(pt_row["y1"]),
            float(pt_row["x2"]),
            float(pt_row["y2"])
        ]

        # Only compare detections belonging to the same class.
        candidates = onnx_frame[
            onnx_frame["class_id"]
            ==
            pt_row["class_id"]
        ]

        for onnx_index, onnx_row in candidates.iterrows():

            if onnx_index in used_onnx_indices:
                continue

            onnx_box = [
                float(onnx_row["x1"]),
                float(onnx_row["y1"]),
                float(onnx_row["x2"]),
                float(onnx_row["y2"])
            ]

            current_iou = calculate_iou(
                pt_box,
                onnx_box
            )

            if current_iou > best_iou:

                best_iou = current_iou
                best_match_index = onnx_index

        # ----------------------------------------------------
        # MATCH FOUND
        # ----------------------------------------------------

        if (
            best_match_index is not None
            and
            best_iou >= MATCH_IOU_THRESHOLD
        ):

            onnx_row = onnx_frame.loc[
                best_match_index
            ]

            used_onnx_indices.add(
                best_match_index
            )

            confidence_difference = (
                float(onnx_row["confidence"])
                -
                float(pt_row["confidence"])
            )

            matched_records.append({

                "frame_number":
                    int(frame_number),

                "class_id":
                    int(pt_row["class_id"]),

                "class_name":
                    pt_row["class_name"],

                "pt_confidence":
                    float(pt_row["confidence"]),

                "onnx_confidence":
                    float(onnx_row["confidence"]),

                "confidence_difference":
                    confidence_difference,

                "absolute_confidence_difference":
                    abs(confidence_difference),

                "bbox_iou":
                    float(best_iou),

                "pt_x1":
                    float(pt_row["x1"]),

                "pt_y1":
                    float(pt_row["y1"]),

                "pt_x2":
                    float(pt_row["x2"]),

                "pt_y2":
                    float(pt_row["y2"]),

                "onnx_x1":
                    float(onnx_row["x1"]),

                "onnx_y1":
                    float(onnx_row["y1"]),

                "onnx_x2":
                    float(onnx_row["x2"]),

                "onnx_y2":
                    float(onnx_row["y2"])
            })

        else:

            pt_only_count += 1

    # --------------------------------------------------------
    # REMAINING ONNX DETECTIONS
    # --------------------------------------------------------

    for onnx_index in onnx_frame.index:

        if onnx_index not in used_onnx_indices:

            onnx_only_count += 1


# ============================================================
# MATCHED DATAFRAME
# ============================================================

matched_df = pd.DataFrame(
    matched_records
)

matched_path = os.path.join(
    COMPARISON_DIR,
    "pt_vs_onnx_matched_detections.csv"
)

matched_df.to_csv(
    matched_path,
    index=False
)


# ============================================================
# MATCHING STATISTICS
# ============================================================

matched_count = len(
    matched_df
)

if matched_count > 0:

    average_bbox_iou = float(
        matched_df["bbox_iou"].mean()
    )

    median_bbox_iou = float(
        matched_df["bbox_iou"].median()
    )

    minimum_bbox_iou = float(
        matched_df["bbox_iou"].min()
    )

    maximum_bbox_iou = float(
        matched_df["bbox_iou"].max()
    )

    average_confidence_difference = float(
        matched_df[
            "confidence_difference"
        ].mean()
    )

    mean_absolute_confidence_difference = float(
        matched_df[
            "absolute_confidence_difference"
        ].mean()
    )

else:

    average_bbox_iou = 0.0
    median_bbox_iou = 0.0
    minimum_bbox_iou = 0.0
    maximum_bbox_iou = 0.0

    average_confidence_difference = 0.0
    mean_absolute_confidence_difference = 0.0


# ============================================================
# MATCHING RATES
# ============================================================

pt_matching_rate = (
    matched_count
    /
    pt_total
    *
    100
    if pt_total > 0
    else 0
)

onnx_matching_rate = (
    matched_count
    /
    onnx_total
    *
    100
    if onnx_total > 0
    else 0
)


# ============================================================
# PERFORMANCE COMPARISON
# ============================================================

# IMPORTANT:
# Compare inference time with inference time.
# Do NOT compare PT complete pipeline time with ONNX
# inference time.

pt_avg_inference_ms = float(
    pt_metrics["processing"][
        "average_inference_time_per_frame_ms"
    ]
)

onnx_avg_inference_ms = float(
    onnx_metrics["processing"][
        "average_inference_time_per_frame_ms"
    ]
)

pt_fps = float(
    pt_metrics["processing"][
        "inference_fps"
    ]
)

onnx_fps = float(
    onnx_metrics["processing"][
        "inference_fps"
    ]
)

pt_pipeline_fps = float(
    pt_metrics["processing"][
        "complete_pipeline_fps"
    ]
)

onnx_pipeline_fps = float(
    onnx_metrics["processing"][
        "complete_pipeline_fps"
    ]
)

inference_time_difference_ms = (
    onnx_avg_inference_ms
    -
    pt_avg_inference_ms
)

inference_time_difference_percent = (
    inference_time_difference_ms
    /
    pt_avg_inference_ms
    *
    100
    if pt_avg_inference_ms > 0
    else 0
)

fps_difference = (
    onnx_fps
    -
    pt_fps
)

fps_difference_percent = (
    fps_difference
    /
    pt_fps
    *
    100
    if pt_fps > 0
    else 0
)

pipeline_fps_difference = (
    onnx_pipeline_fps
    -
    pt_pipeline_fps
)

pipeline_fps_difference_percent = (
    pipeline_fps_difference
    /
    pt_pipeline_fps
    *
    100
    if pt_pipeline_fps > 0
    else 0
)


# ============================================================
# CONFIDENCE COMPARISON
# ============================================================

pt_average_confidence = float(
    pt_metrics["confidence"]["average"]
)

onnx_average_confidence = float(
    onnx_metrics["confidence"]["average"]
)

confidence_difference = (
    onnx_average_confidence
    -
    pt_average_confidence
)


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {

    "experiment": {

        "model":
            "YOLO11m",

        "pt_format":
            "PyTorch",

        "onnx_format":
            "ONNX",

        "runtime":
            "ONNX Runtime",

        "video":
            "PNNL_Parking_LOT(1).avi",

        "frames_compared":
            total_frames,

        "confidence_threshold":
            0.25,

        "nms_iou_threshold":
            0.45,

        "matching_iou_threshold":
            MATCH_IOU_THRESHOLD
    },

    "detection_counts": {

        "pt":
            pt_total,

        "onnx":
            onnx_total,

        "difference_onnx_minus_pt":
            detection_difference,

        "difference_percent":
            detection_difference_percent
    },

    "matching": {

        "matched_detections":
            matched_count,

        "pt_only_detections":
            pt_only_count,

        "onnx_only_detections":
            onnx_only_count,

        "pt_matching_rate_percent":
            pt_matching_rate,

        "onnx_matching_rate_percent":
            onnx_matching_rate,

        "average_bbox_iou":
            average_bbox_iou,

        "median_bbox_iou":
            median_bbox_iou,

        "minimum_bbox_iou":
            minimum_bbox_iou,

        "maximum_bbox_iou":
            maximum_bbox_iou,

        "average_confidence_difference":
            average_confidence_difference,

        "mean_absolute_confidence_difference":
            mean_absolute_confidence_difference
    },

    "performance": {

        "pt_average_inference_time_ms":
            pt_avg_inference_ms,

        "onnx_average_inference_time_ms":
            onnx_avg_inference_ms,

        "inference_time_difference_ms":
            inference_time_difference_ms,

        "inference_time_difference_percent":
            inference_time_difference_percent,

        "pt_inference_fps":
            pt_fps,

        "onnx_inference_fps":
            onnx_fps,

        "fps_difference":
            fps_difference,

        "fps_difference_percent":
            fps_difference_percent,

        "pt_pipeline_fps":
            pt_pipeline_fps,

        "onnx_pipeline_fps":
            onnx_pipeline_fps,

        "pipeline_fps_difference":
            pipeline_fps_difference,

        "pipeline_fps_difference_percent":
            pipeline_fps_difference_percent
    },

    "confidence": {

        "pt_average":
            pt_average_confidence,

        "onnx_average":
            onnx_average_confidence,

        "difference":
            confidence_difference
    },

    "class_comparison":
        class_comparison.to_dict(
            orient="records"
        ),

    "evaluation_metrics": {

        "precision":
            "Ground truth required",

        "recall":
            "Ground truth required",

        "f1_score":
            "Ground truth required",

        "map50":
            "Ground truth required",

        "map50_95":
            "Ground truth required",

        "iou":
            "Ground truth required"
    }
}


summary_json_path = os.path.join(
    COMPARISON_DIR,
    "pt_vs_onnx_summary.json"
)

with open(
    summary_json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# TEXT REPORT
# ============================================================

report_path = os.path.join(
    COMPARISON_DIR,
    "pt_vs_onnx_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "YOLO11m PT vs ONNX COMPARISON REPORT\n"
    )

    f.write("=" * 70 + "\n\n")

    # --------------------------------------------------------
    # EXPERIMENT
    # --------------------------------------------------------

    f.write("EXPERIMENT\n")
    f.write("-" * 70 + "\n")

    f.write(
        "Model: YOLO11m\n"
    )

    f.write(
        "Video: PNNL_Parking_LOT(1).avi\n"
    )

    f.write(
        f"Frames compared: {total_frames}\n"
    )

    f.write(
        "Confidence threshold: 0.25\n"
    )

    f.write(
        "NMS IoU threshold: 0.45\n"
    )

    f.write(
        f"Detection matching IoU: "
        f"{MATCH_IOU_THRESHOLD}\n\n"
    )

    # --------------------------------------------------------
    # DETECTION COUNT
    # --------------------------------------------------------

    f.write("DETECTION COUNT\n")
    f.write("-" * 70 + "\n")

    f.write(
        f"PT detections: "
        f"{pt_total}\n"
    )

    f.write(
        f"ONNX detections: "
        f"{onnx_total}\n"
    )

    f.write(
        f"Difference: "
        f"{detection_difference}\n"
    )

    f.write(
        f"Difference (%): "
        f"{detection_difference_percent:.4f}%\n\n"
    )

    # --------------------------------------------------------
    # MATCHING
    # --------------------------------------------------------

    f.write("DETECTION MATCHING\n")
    f.write("-" * 70 + "\n")

    f.write(
        f"Matched detections: "
        f"{matched_count}\n"
    )

    f.write(
        f"PT-only detections: "
        f"{pt_only_count}\n"
    )

    f.write(
        f"ONNX-only detections: "
        f"{onnx_only_count}\n"
    )

    f.write(
        f"PT matching rate: "
        f"{pt_matching_rate:.4f}%\n"
    )

    f.write(
        f"ONNX matching rate: "
        f"{onnx_matching_rate:.4f}%\n"
    )

    f.write(
        f"Average matched bbox IoU: "
        f"{average_bbox_iou:.6f}\n"
    )

    f.write(
        f"Median matched bbox IoU: "
        f"{median_bbox_iou:.6f}\n"
    )

    f.write(
        f"Minimum matched bbox IoU: "
        f"{minimum_bbox_iou:.6f}\n"
    )

    f.write(
        f"Maximum matched bbox IoU: "
        f"{maximum_bbox_iou:.6f}\n"
    )

    f.write(
        f"Average confidence difference: "
        f"{average_confidence_difference:.6f}\n"
    )

    f.write(
        f"Mean absolute confidence difference: "
        f"{mean_absolute_confidence_difference:.6f}\n\n"
    )

    # --------------------------------------------------------
    # PERFORMANCE
    # --------------------------------------------------------

    f.write("PERFORMANCE\n")
    f.write("-" * 70 + "\n")

    f.write(
        f"PT average inference time: "
        f"{pt_avg_inference_ms:.4f} ms/frame\n"
    )

    f.write(
        f"ONNX average inference time: "
        f"{onnx_avg_inference_ms:.4f} ms/frame\n"
    )

    f.write(
        f"Inference time difference: "
        f"{inference_time_difference_ms:.4f} ms/frame\n"
    )

    f.write(
        f"Inference time difference (%): "
        f"{inference_time_difference_percent:.4f}%\n"
    )

    f.write(
        f"PT inference FPS: "
        f"{pt_fps:.4f}\n"
    )

    f.write(
        f"ONNX inference FPS: "
        f"{onnx_fps:.4f}\n"
    )

    f.write(
        f"FPS difference: "
        f"{fps_difference:.4f}\n"
    )

    f.write(
        f"FPS difference (%): "
        f"{fps_difference_percent:.4f}%\n"
    )

    f.write(
        f"PT pipeline FPS: "
        f"{pt_pipeline_fps:.4f}\n"
    )

    f.write(
        f"ONNX pipeline FPS: "
        f"{onnx_pipeline_fps:.4f}\n"
    )

    f.write(
        f"Pipeline FPS difference: "
        f"{pipeline_fps_difference:.4f}\n"
    )

    f.write(
        f"Pipeline FPS difference (%): "
        f"{pipeline_fps_difference_percent:.4f}%\n\n"
    )

    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    f.write("CONFIDENCE\n")
    f.write("-" * 70 + "\n")

    f.write(
        f"PT average confidence: "
        f"{pt_average_confidence:.6f}\n"
    )

    f.write(
        f"ONNX average confidence: "
        f"{onnx_average_confidence:.6f}\n"
    )

    f.write(
        f"Difference: "
        f"{confidence_difference:.6f}\n\n"
    )

    # --------------------------------------------------------
    # CLASS COMPARISON
    # --------------------------------------------------------

    f.write("CLASS COMPARISON\n")
    f.write("-" * 70 + "\n")

    for _, row in class_comparison.iterrows():

        difference_percent = row[
            "difference_percent"
        ]

        if pd.isna(difference_percent):

            difference_percent_text = "N/A"

        else:

            difference_percent_text = (
                f"{difference_percent:.4f}%"
            )

        f.write(
            f"{row['class_name']}: "
            f"PT={int(row['pt_detections'])}, "
            f"ONNX={int(row['onnx_detections'])}, "
            f"Difference={int(row['difference'])}, "
            f"Difference%={difference_percent_text}\n"
        )

    f.write("\n")

    # --------------------------------------------------------
    # EVALUATION METRICS
    # --------------------------------------------------------

    f.write("EVALUATION METRICS\n")
    f.write("-" * 70 + "\n")

    f.write(
        "Precision: Ground truth required\n"
    )

    f.write(
        "Recall: Ground truth required\n"
    )

    f.write(
        "F1-score: Ground truth required\n"
    )

    f.write(
        "mAP@50: Ground truth required\n"
    )

    f.write(
        "mAP@50-95: Ground truth required\n"
    )

    f.write(
        "IoU: Ground truth required\n"
    )


# ============================================================
# PLOT 1 — DETECTION COUNT
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.bar(
    ["PT", "ONNX"],
    [
        pt_total,
        onnx_total
    ]
)

plt.ylabel(
    "Total Detections"
)

plt.title(
    "YOLO11m PT vs ONNX Detection Count"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "detection_count_comparison.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# PLOT 2 — CONFIDENCE
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.bar(
    ["PT", "ONNX"],
    [
        pt_average_confidence,
        onnx_average_confidence
    ]
)

plt.ylabel(
    "Average Confidence"
)

plt.title(
    "YOLO11m PT vs ONNX Average Confidence"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "confidence_comparison.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# PLOT 3 — INFERENCE FPS
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.bar(
    ["PT", "ONNX"],
    [
        pt_fps,
        onnx_fps
    ]
)

plt.ylabel(
    "Inference FPS"
)

plt.title(
    "YOLO11m PT vs ONNX Inference FPS"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "inference_performance_comparison.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# PLOT 4 — AVERAGE INFERENCE TIME
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.bar(
    ["PT", "ONNX"],
    [
        pt_avg_inference_ms,
        onnx_avg_inference_ms
    ]
)

plt.ylabel(
    "Average Inference Time (ms/frame)"
)

plt.title(
    "YOLO11m PT vs ONNX Average Inference Time"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOTS_DIR,
        "average_inference_time_comparison.png"
    ),
    dpi=150
)

plt.close()


# ============================================================
# FINAL CONSOLE OUTPUT
# ============================================================

print()
print("=" * 70)
print("PT vs ONNX COMPARISON COMPLETED")
print("=" * 70)

print()
print("DETECTION RESULTS")
print("-" * 70)

print(
    f"PT detections          : "
    f"{pt_total}"
)

print(
    f"ONNX detections        : "
    f"{onnx_total}"
)

print(
    f"Detection difference   : "
    f"{detection_difference}"
)

print(
    f"Difference percentage  : "
    f"{detection_difference_percent:.4f}%"
)

print()
print("MATCHING RESULTS")
print("-" * 70)

print(
    f"Matched detections     : "
    f"{matched_count}"
)

print(
    f"PT-only detections     : "
    f"{pt_only_count}"
)

print(
    f"ONNX-only detections   : "
    f"{onnx_only_count}"
)

print(
    f"PT matching rate       : "
    f"{pt_matching_rate:.2f}%"
)

print(
    f"ONNX matching rate     : "
    f"{onnx_matching_rate:.2f}%"
)

print(
    f"Average matched IoU    : "
    f"{average_bbox_iou:.4f}"
)

print()
print("PERFORMANCE")
print("-" * 70)

print(
    f"PT inference time      : "
    f"{pt_avg_inference_ms:.2f} ms/frame"
)

print(
    f"ONNX inference time    : "
    f"{onnx_avg_inference_ms:.2f} ms/frame"
)

print(
    f"PT inference FPS       : "
    f"{pt_fps:.2f}"
)

print(
    f"ONNX inference FPS     : "
    f"{onnx_fps:.2f}"
)

print(
    f"PT pipeline FPS        : "
    f"{pt_pipeline_fps:.2f}"
)

print(
    f"ONNX pipeline FPS      : "
    f"{onnx_pipeline_fps:.2f}"
)

print()
print("OUTPUT FILES")
print("-" * 70)

print(class_comparison_path)
print(frame_comparison_path)
print(matched_path)
print(summary_json_path)
print(report_path)
print(PLOTS_DIR)

print()
print("=" * 70)