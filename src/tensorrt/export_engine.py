from pathlib import Path
import json
import platform
import time

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
PT_MODEL = ROOT / "models" / "pt" / "yolo11m.pt"
ENGINE_DIR = ROOT / "models" / "tensorrt"
ENGINE_PATH = ENGINE_DIR / "yolo11m.engine"
METRICS_DIR = ROOT / "outputs" / "tensorrt" / "metrics"

IMAGE_SIZE = 640
HALF_PRECISION = True
WORKSPACE_GB = 4

def main():
    ENGINE_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    if not PT_MODEL.exists():
        raise FileNotFoundError(
            f"Missing PyTorch model: {PT_MODEL}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Install a compatible NVIDIA driver "
            "and CUDA-enabled PyTorch on the GPU system first."
        )

    print("=" * 70)
    print("YOLO11m -> TensorRT ENGINE EXPORT")
    print("=" * 70)
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("Image size:", IMAGE_SIZE)
    print("FP16 requested:", HALF_PRECISION)

    model = YOLO(str(PT_MODEL))
    start = time.perf_counter()

    exported = model.export(
        format="engine",
        imgsz=IMAGE_SIZE,
        device=0,
        half=HALF_PRECISION,
        workspace=WORKSPACE_GB,
        batch=1,
        dynamic=False,
    )

    elapsed = time.perf_counter() - start
    exported_path = Path(exported)

    if not exported_path.exists():
        raise FileNotFoundError(
            f"Ultralytics reported an engine path that does not exist: "
            f"{exported_path}"
        )

    # Keep the engine in the project's dedicated models/tensorrt folder.
    if exported_path.resolve() != ENGINE_PATH.resolve():
        import shutil
        shutil.copy2(exported_path, ENGINE_PATH)

    try:
        import tensorrt as trt
        trt_version = trt.__version__
    except ImportError:
        trt_version = "Unknown"

    report = {
        "model": "YOLO11m",
        "source_model": str(PT_MODEL.relative_to(ROOT)),
        "engine_path": str(ENGINE_PATH.relative_to(ROOT)),
        "engine_size_bytes": ENGINE_PATH.stat().st_size,
        "export_time_seconds": elapsed,
        "image_size": IMAGE_SIZE,
        "batch_size": 1,
        "precision_requested": "FP16" if HALF_PRECISION else "FP32",
        "device": torch.cuda.get_device_name(0),
        "pytorch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "tensorrt_version": trt_version,
        "platform": platform.platform(),
        "note": (
            "Engine compatibility depends on the GPU, TensorRT version, "
            "CUDA/runtime and platform. Re-export on the target GPU system."
        ),
    }

    out = METRICS_DIR / "tensorrt_engine_export.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nENGINE EXPORT COMPLETE")
    print("Engine:", ENGINE_PATH)
    print("Size (MB):", round(ENGINE_PATH.stat().st_size / 1e6, 2))
    print("Export time (s):", round(elapsed, 2))
    print("Export metadata:", out)

if __name__ == "__main__":
    main()