from pathlib import Path
from collections import Counter
import csv
import json
import time
import platform

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "input" / "videos" / "PNNL_Parking_LOT(1).avi"
ENGINE = ROOT / "models" / "tensorrt" / "yolo11m.engine"

OUT = ROOT / "outputs" / "tensorrt"
VIDEO_OUT = OUT / "videos" / "yolo11m_tensorrt_output.mp4"
CSV_OUT = OUT / "detections" / "yolo11m_tensorrt_detections.csv"
JSON_OUT = OUT / "detections" / "yolo11m_tensorrt_detections.json"
METRICS_OUT = OUT / "metrics" / "yolo11m_tensorrt_metrics.json"
SUMMARY_OUT = OUT / "metrics" / "yolo11m_tensorrt_summary.txt"

CONF = 0.25
IOU = 0.45
IMGSZ = 640
DEVICE = 0

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")
    if not ENGINE.exists():
        raise FileNotFoundError(f"TensorRT engine missing: {ENGINE}")
    if not VIDEO.exists():
        raise FileNotFoundError(f"Input video missing: {VIDEO}")

    for p in [VIDEO_OUT, CSV_OUT, JSON_OUT, METRICS_OUT, SUMMARY_OUT]:
        p.parent.mkdir(parents=True, exist_ok=True)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Engine:", ENGINE)
    print("Video:", VIDEO)

    model = YOLO(str(ENGINE), task="detect")
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 29.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # MP4V is used for broad OpenCV compatibility. Convert to H.264 later
    # if the target machine's video preview requires it.
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
        "frame_number", "timestamp_seconds", "class_id", "class_name",
        "confidence", "x1", "y1", "x2", "y2", "box_width", "box_height"
    ]
    rows = []
    class_counts = Counter()
    confidences = []
    inference_times = []
    frames_with_detections = 0
    frame_number = 0
    total_start = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_number += 1
            t0 = time.perf_counter()
            result = model.predict(
                source=frame,
                conf=CONF,
                iou=IOU,
                imgsz=IMGSZ,
                device=DEVICE,
                verbose=False,
            )[0]
            inference_times.append(time.perf_counter() - t0)

            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                frames_with_detections += 1
                xyxy = boxes.xyxy.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                scores = boxes.conf.cpu().numpy()

                for box, cls_id, score in zip(xyxy, cls_ids, scores):
                    x1, y1, x2, y2 = map(float, box)
                    name = result.names.get(int(cls_id), str(cls_id))
                    rows.append({
                        "frame_number": frame_number,
                        "timestamp_seconds": (frame_number - 1) / fps,
                        "class_id": int(cls_id),
                        "class_name": name,
                        "confidence": float(score),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "box_width": max(0.0, x2 - x1),
                        "box_height": max(0.0, y2 - y1),
                    })
                    class_counts[name] += 1
                    confidences.append(float(score))

            annotated = result.plot()
            writer.write(annotated)

            if frame_number % 100 == 0:
                print(f"Processed {frame_number}/{expected or '?'} frames")

    finally:
        cap.release()
        writer.release()

    total_seconds = time.perf_counter() - total_start
    frames = frame_number
    total_detections = len(rows)
    avg_infer = sum(inference_times) / len(inference_times) if inference_times else 0

    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    JSON_OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    metrics = {
        "experiment": {
            "model": "YOLO11m",
            "model_format": "TensorRT engine",
            "device": torch.cuda.get_device_name(0),
            "precision": "Engine-configured FP16/FP32; see export metadata",
            "confidence_threshold": CONF,
            "iou_threshold": IOU,
            "image_size": IMGSZ,
            "platform": platform.platform(),
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
            "frames_with_detections": frames_with_detections,
            "total_detections": total_detections,
            "total_pipeline_time_seconds": total_seconds,
            "average_inference_time_per_frame_seconds": avg_infer,
            "average_inference_time_per_frame_ms": avg_infer * 1000,
            "inference_fps": 1 / avg_infer if avg_infer else 0,
            "complete_pipeline_fps": frames / total_seconds if total_seconds else 0,
        },
        "confidence": {
            "average": sum(confidences) / len(confidences) if confidences else None,
            "minimum": min(confidences) if confidences else None,
            "maximum": max(confidences) if confidences else None,
        },
        "detections_by_class": dict(class_counts),
        "evaluation_metrics": {
            "precision": None, "recall": None, "f1_score": None,
            "map50": None, "map50_95": None, "iou": None,
            "reason": "Ground-truth annotations were not provided.",
        },
    }

    METRICS_OUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    SUMMARY_OUT.write_text(
        "\n".join([
            "YOLO11m TensorRT Detection Summary",
            f"GPU: {torch.cuda.get_device_name(0)}",
            f"Frames processed: {frames}",
            f"Total detections: {total_detections}",
            f"Average confidence: {metrics['confidence']['average']}",
            f"Average inference time (ms/frame): {avg_infer * 1000:.3f}",
            f"Inference FPS: {metrics['processing']['inference_fps']:.4f}",
            f"Pipeline FPS: {metrics['processing']['complete_pipeline_fps']:.4f}",
            "Ground-truth accuracy metrics: unavailable",
        ]),
        encoding="utf-8",
    )

    print("\nTENSORRT DETECTION COMPLETE")
    print("Frames:", frames)
    print("Detections:", total_detections)
    print("Video:", VIDEO_OUT)
    print("Metrics:", METRICS_OUT)

if __name__ == "__main__":
    main()