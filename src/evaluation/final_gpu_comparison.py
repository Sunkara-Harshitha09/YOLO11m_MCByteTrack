from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
FINAL = OUTPUTS / "final"

METRICS = {
    "PT": OUTPUTS / "pt" / "metrics" / "yolo11m_pt_metrics.json",
    "ONNX": OUTPUTS / "onnx" / "metrics" / "yolo11m_onnx_metrics.json",
    "TensorRT": OUTPUTS / "tensorrt" / "metrics" / "yolo11m_tensorrt_metrics.json",
    "CPU_MCByteTrack": OUTPUTS / "mcbytetrack" / "metrics" / "mcbytetrack_metrics.json",
    "GPU_MCByteTrack": OUTPUTS / "mcbytetrack_gpu" / "metrics" / "tensorrt_mcbytetrack_metrics.json",
}

def load_json(path):
    if not path.exists():
        print(f"WARNING: Missing metrics file: {path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def get_metrics(path, label):
    d = load_json(path)
    processing = d.get("processing", {})
    tracking = d.get("tracking", {})
    confidence = d.get("confidence", {})

    frames = processing.get("frames_processed")
    detections = processing.get("total_detections")
    records = processing.get("tracking_records")

    return {
        "system": label,
        "frames_processed": frames,
        "detection_count": detections,
        "tracking_records": records,
        "average_confidence": confidence.get("average"),
        "average_inference_ms": processing.get(
            "average_inference_time_per_frame_ms"
        ),
        "inference_fps": processing.get("inference_fps"),
        "pipeline_fps": processing.get("complete_pipeline_fps"),
        "total_time_seconds": processing.get(
            "total_pipeline_time_seconds",
            processing.get("total_processing_time_seconds"),
        ),
        "unique_track_ids": tracking.get("total_unique_track_ids"),
        "average_tracks_per_frame": tracking.get("average_tracks_per_frame"),
        "average_track_length_frames": tracking.get(
            "average_track_length_frames"
        ),
        "minimum_track_length_frames": tracking.get(
            "minimum_track_length_frames"
        ),
        "maximum_track_length_frames": tracking.get(
            "maximum_track_length_frames"
        ),
    }

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def write_report(path, title, rows, notes):
    lines = [title, "=" * len(title), ""]
    for row in rows:
        lines.append(f"System: {row['system']}")
        for key, value in row.items():
            if key != "system":
                lines.append(f"  {key}: {value}")
        lines.append("")
    lines.extend(["Notes:", *[f"- {n}" for n in notes], ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")

def make_plot(path, rows, metric, title, ylabel):
    valid = [
        row for row in rows
        if isinstance(row.get(metric), (int, float))
    ]
    if not valid:
        print(f"Skipping empty plot: {path.name}")
        return

    plt.figure(figsize=(9, 5))
    plt.bar([r["system"] for r in valid], [r[metric] for r in valid])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()

def main():
    backend = FINAL / "backend_comparison"
    cpu_gpu = FINAL / "cpu_gpu_comparison"

    for base in [backend, cpu_gpu]:
        for folder in ["tables", "reports", "plots"]:
            (base / folder).mkdir(parents=True, exist_ok=True)

    pt = get_metrics(METRICS["PT"], "YOLO11m PT (CPU)")
    onnx = get_metrics(METRICS["ONNX"], "YOLO11m ONNX")
    trt = get_metrics(METRICS["TensorRT"], "YOLO11m TensorRT (GPU)")
    cpu_track = get_metrics(
        METRICS["CPU_MCByteTrack"], "YOLO11m + MCByteTrack (CPU)"
    )
    gpu_track = get_metrics(
        METRICS["GPU_MCByteTrack"], "YOLO11m TensorRT + ByteTrack (GPU)"
    )

    # Comparison 1: Detection backends.
    backend_rows = [pt, onnx, trt]
    write_csv(backend / "tables" / "pt_onnx_tensorrt_comparison.csv", backend_rows)
    (backend / "tables" / "pt_onnx_tensorrt_comparison.json").write_text(
        json.dumps(backend_rows, indent=2), encoding="utf-8"
    )
    write_report(
        backend / "reports" / "pt_onnx_tensorrt_report.txt",
        "YOLO11m PT vs ONNX vs TensorRT",
        backend_rows,
        [
            "PT, ONNX and TensorRT are detection-backend experiments.",
            "PT was run on CPU in the recorded baseline; TensorRT uses the GPU.",
            "Different hardware and precision can affect timing and numerical results.",
            "Detection count and confidence are not ground-truth accuracy metrics.",
            "Precision, recall and mAP require labelled ground-truth boxes.",
        ],
    )
    make_plot(
        backend / "plots" / "inference_time_ms.png",
        backend_rows, "average_inference_ms",
        "Average Inference Time", "Milliseconds per frame",
    )
    make_plot(
        backend / "plots" / "inference_fps.png",
        backend_rows, "inference_fps",
        "Inference Throughput", "Frames per second",
    )
    make_plot(
        backend / "plots" / "pipeline_fps.png",
        backend_rows, "pipeline_fps",
        "Complete Pipeline Throughput", "Frames per second",
    )
    make_plot(
        backend / "plots" / "confidence.png",
        backend_rows, "average_confidence",
        "Average Detection Confidence", "Confidence",
    )

    # Comparison 2: CPU vs GPU for detection and tracking.
    cpu_gpu_rows = [
        pt,
        trt,
        cpu_track,
        gpu_track,
    ]
    write_csv(cpu_gpu / "tables" / "cpu_gpu_comparison.csv", cpu_gpu_rows)
    (cpu_gpu / "tables" / "cpu_gpu_comparison.json").write_text(
        json.dumps(cpu_gpu_rows, indent=2), encoding="utf-8"
    )
    write_report(
        cpu_gpu / "reports" / "cpu_gpu_report.txt",
        "CPU vs GPU Detection and Tracking",
        cpu_gpu_rows,
        [
            "CPU detection: YOLO11m PT; GPU detection: YOLO11m TensorRT.",
            "CPU tracking: existing YOLO11m PT + MCByteTrack.",
            "GPU tracking: YOLO11m TensorRT + ByteTrack.",
            "Tracking statistics are meaningful only with their tracker configuration.",
            "GPU model, precision, driver and software versions should be recorded.",
            "For fair timing, use the same video and image size, warm up each model, "
            "and compare both inference-only and complete pipeline timings.",
            "MOTA, MOTP, IDF1 and HOTA require ground-truth tracking annotations.",
        ],
    )
    make_plot(
        cpu_gpu / "plots" / "pipeline_fps.png",
        cpu_gpu_rows, "pipeline_fps",
        "CPU vs GPU Pipeline FPS", "Frames per second",
    )
    make_plot(
        cpu_gpu / "plots" / "inference_time_ms.png",
        cpu_gpu_rows, "average_inference_ms",
        "CPU vs GPU Inference Time", "Milliseconds per frame",
    )
    make_plot(
        cpu_gpu / "plots" / "average_confidence.png",
        cpu_gpu_rows, "average_confidence",
        "Average Confidence by Pipeline", "Confidence",
    )

    print("=" * 70)
    print("GPU COMPARISONS GENERATED")
    print("=" * 70)
    print("Backend comparison:", backend)
    print("CPU/GPU comparison:", cpu_gpu)
    print("Missing GPU metrics are reported as warnings; run GPU scripts first.")

if __name__ == "__main__":
    main()