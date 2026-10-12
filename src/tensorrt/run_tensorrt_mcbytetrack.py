from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import time

import cv2
import numpy as np
import torch
import supervision as sv
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "input" / "videos" / "PNNL_Parking_LOT(1).avi"
ENGINE = ROOT / "models" / "tensorrt" / "yolo11m.engine"
OUT = ROOT / "outputs" / "mcbytetrack_gpu"

VIDEO_OUT = OUT / "videos" / "yolo11m_tensorrt_mcbytetrack_output.mp4"
CSV_OUT = OUT / "tracking" / "tensorrt_mcbytetrack_tracks.csv"
JSON_OUT = OUT / "tracking" / "tensorrt_mcbytetrack_tracks.json"
METRICS_OUT = OUT / "metrics" / "tensorrt_mcbytetrack_metrics.json"
SUMMARY_OUT = OUT / "metrics" / "tensorrt_mcbytetrack_summary.txt"

CONF = 0.25
IOU = 0.45
IMGSZ = 640
DEVICE = 0

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")
    if not VIDEO.exists():
        raise FileNotFoundError(f"Video missing: {VIDEO}")
    if not ENGINE.exists():
        raise FileNotFoundError(f"TensorRT engine missing: {ENGINE}")

    for p in [VIDEO_OUT, CSV_OUT, JSON_OUT, METRICS_OUT, SUMMARY_OUT]:
        p.parent.mkdir(parents=True, exist_ok=True)

    print("TensorRT + GPU MCByteTrack")
    print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO(str(ENGINE), task="detect")
    tracker = sv.ByteTrack()

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 29.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(VIDEO_OUT),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create video: {VIDEO_OUT}")

    fields = [
        "frame_number", "timestamp_seconds", "track_id", "class_id",
        "class_name", "confidence", "x1", "y1", "x2", "y2",
        "box_width", "box_height",
    ]
    rows = []
    confidences = []
    class_counts = Counter()
    track_frames = defaultdict(set)
    inference_times = []
    frames = 0
    total_start = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frames += 1

            # Synchronize CUDA to measure completed GPU inference time.
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            result = model.predict(
                source=frame,
                conf=CONF,
                iou=IOU,
                imgsz=IMGSZ,
                device=DEVICE,
                verbose=False,
            )[0]

            torch.cuda.synchronize()
            inference_times.append(time.perf_counter() - t0)

            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                detections = sv.Detections.empty()
            else:
                detections = sv.Detections(
                    xyxy=boxes.xyxy.cpu().numpy(),
                    confidence=boxes.conf.cpu().numpy(),
                    class_id=boxes.cls.cpu().numpy().astype(int),
                )

            tracked = tracker.update_with_detections(detections)
            annotated = frame.copy()

            if tracked.tracker_id is not None and len(tracked) > 0:
                for i, xyxy in enumerate(tracked.xyxy):
                    track_id = int(tracked.tracker_id[i])
                    class_id = (
                        int(tracked.class_id[i])
                        if tracked.class_id is not None else -1
                    )
                    confidence = (
                        float(tracked.confidence[i])
                        if tracked.confidence is not None else None
                    )
                    class_name = result.names.get(
                        class_id, str(class_id)
                    )

                    x1, y1, x2, y2 = map(float, xyxy)
                    cv2.rectangle(
                        annotated,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),
                        2,
                    )
                    label = f"{class_name} ID:{track_id}"
                    cv2.putText(
                        annotated,
                        label,
                        (int(x1), max(20, int(y1) - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )

                    rows.append({
                        "frame_number": frames,
                        "timestamp_seconds": (frames - 1) / fps,
                        "track_id": track_id,
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "box_width": max(0.0, x2 - x1),
                        "box_height": max(0.0, y2 - y1),
                    })
                    track_frames[track_id].add(frames)
                    class_counts[class_name] += 1
                    if confidence is not None:
                        confidences.append(confidence)

            writer.write(annotated)

            if frames % 100 == 0:
                print(f"Processed {frames}/{expected or '?'} frames")

    finally:
        cap.release()
        writer.release()

    total_seconds = time.perf_counter() - total_start
    avg_infer = sum(inference_times) / len(inference_times) if inference_times else 0
    lengths = [len(v) for v in track_frames.values()]

    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    JSON_OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    metrics = {
        "experiment": {
            "model": "YOLO11m",
            "model_format": "TensorRT engine",
            "tracker": "supervision.ByteTrack",
            "device": torch.cuda.get_device_name(0),
            "confidence_threshold": CONF,
            "iou_threshold": IOU,
            "image_size": IMGSZ,
        },
        "input_video": {
            "filename": VIDEO.name,
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames_expected": expected,
        },
        "processing": {
            "frames_processed": frames,
            "tracking_records": len(rows),
            "total_processing_time_seconds": total_seconds,
            "average_inference_time_per_frame_seconds": avg_infer,
            "average_inference_time_per_frame_ms": avg_infer * 1000,
            "inference_fps": 1 / avg_infer if avg_infer else 0,
            "complete_pipeline_fps": frames / total_seconds if total_seconds else 0,
        },
        "tracking": {
            "total_unique_track_ids": len(track_frames),
            "average_tracks_per_frame": len(rows) / frames if frames else 0,
            "average_track_length_frames": (
                sum(lengths) / len(lengths) if lengths else 0
            ),
            "minimum_track_length_frames": min(lengths) if lengths else 0,
            "maximum_track_length_frames": max(lengths) if lengths else 0,
            "tracks_by_class": dict(
                Counter(
                    next(
                        (r["class_name"] for r in rows
                         if r["track_id"] == tid),
                        "unknown",
                    )
                    for tid in track_frames
                )
            ),
        },
        "confidence": {
            "average": sum(confidences) / len(confidences) if confidences else None,
        },
        "evaluation_metrics": {
            "mota": None, "motp": None, "idf1": None, "hota": None,
            "reason": "Ground-truth tracking annotations were not provided.",
        },
    }

    METRICS_OUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    SUMMARY_OUT.write_text(
        "\n".join([
            "TensorRT + MCByteTrack GPU Summary",
            f"GPU: {torch.cuda.get_device_name(0)}",
            f"Frames: {frames}",
            f"Tracking records: {len(rows)}",
            f"Unique tracks: {len(track_frames)}",
            f"Average track length: {metrics['tracking']['average_track_length_frames']:.2f}",
            f"Average inference time (ms/frame): {avg_infer * 1000:.3f}",
            f"Inference FPS: {metrics['processing']['inference_fps']:.4f}",
            f"Pipeline FPS: {metrics['processing']['complete_pipeline_fps']:.4f}",
            "Ground-truth tracking metrics: unavailable",
        ]),
        encoding="utf-8",
    )

    print("\nGPU TRACKING COMPLETE")
    print("Frames:", frames)
    print("Records:", len(rows))
    print("Unique tracks:", len(track_frames))
    print("Output:", OUT)

if __name__ == "__main__":
    main()