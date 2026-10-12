# YOLO11m Object Detection, ONNX, TensorRT and MCByteTrack

A computer vision project for object detection, multi-object tracking, model-format comparison, and CPU-versus-GPU performance evaluation using YOLO11m, ONNX Runtime, NVIDIA TensorRT, and ByteTrack.

The project preserves the original CPU experiments and adds a separate TensorRT GPU pipeline, GPU-based tracking, and two independent comparison workflows.

## Project Objectives

- Run YOLO11m object detection using the PyTorch model.
- Convert the YOLO11m model into ONNX format.
- Run detection using ONNX Runtime.
- Compare PyTorch, ONNX, and TensorRT detection backends.
- Implement MCByteTrack using YOLO11m detections.
- Run a separate GPU tracking pipeline using TensorRT and ByteTrack.
- Compare CPU detection and tracking results against GPU results.
- Measure inference time, processing time, FPS, confidence, detection counts, and tracking statistics.
- Store outputs in CSV, JSON, TXT, and PNG formats.
- Maintain separate output directories for each experiment.
- Preserve the original CPU experiment results without overwriting them.

## Project Workflow

The project consists of five major experiment stages:

1. YOLO11m PyTorch detection on CPU.
2. YOLO11m ONNX conversion and inference.
3. YOLO11m plus MCByteTrack tracking on CPU.
4. YOLO11m TensorRT detection and TensorRT plus ByteTrack tracking on GPU.
5. Final model-backend and CPU-versus-GPU comparisons.

The same source video and nominal detection configuration are used across experiments wherever supported. Actual execution hardware, precision, runtime versions, and timing methodology must be recorded for fair comparison.

## Common Input Video

Video filename: PNNL_Parking_LOT(1).avi

Video properties:

- Resolution: 1920 × 1080
- Original frame rate: 29 FPS
- Total frames: 998
- Duration: approximately 34.41 seconds
- Codec: FMP4

The video is used as the common input for the detection and tracking experiments.

The input video is excluded from the Git repository because it is a binary dataset file.

Expected local path: input/videos/PNNL_Parking_LOT(1).avi

## Model Configuration

Model: YOLO11m

Model formats:

- PyTorch: yolo11m.pt
- ONNX: yolo11m.onnx
- TensorRT: yolo11m.engine

Nominal detection settings:

- Confidence threshold: 0.25
- IoU threshold: 0.45
- Input image size: 640
- Batch size for TensorRT engine export: 1
- Detection classes: COCO pretrained model classes
- CPU baseline: PyTorch and ONNX Runtime CPU execution
- GPU experiment: TensorRT using a compatible NVIDIA GPU

The TensorRT engine is generated on the target GPU system. TensorRT engine compatibility depends on the target platform, GPU, TensorRT version, and runtime configuration.

## Project Structure

YOLO11m_MCByteTrack/

- input/
  - videos/
- models/
  - pt/
  - onnx/
  - tensorrt/
- src/
  - detection/
  - conversion/
  - tracking/
  - evaluation/
  - tensorrt/
- outputs/
  - pt/
    - detections/
    - videos/
    - metrics/
  - onnx/
    - detections/
    - videos/
    - metrics/
  - mcbytetrack/
    - tracking/
    - videos/
    - metrics/
  - tensorrt/
    - detections/
    - videos/
    - metrics/
  - mcbytetrack_gpu/
    - tracking/
    - videos/
    - metrics/
  - final/
    - reports/
    - tables/
    - plots/
    - summaries/
    - backend_comparison/
      - tables/
      - reports/
      - plots/
    - cpu_gpu_comparison/
      - tables/
      - reports/
      - plots/
- requirements.txt
- .gitignore
- README.md

The exact contents of the directories depend on which experiment scripts have been executed successfully.

## Environment and Dependencies

The original CPU experiments were performed using Python 3.12.10 and CPU-only PyTorch.

Main libraries used in the project include:

- Python
- PyTorch
- Ultralytics
- OpenCV
- ONNX
- ONNX Runtime
- NVIDIA TensorRT
- NumPy
- Pandas
- Matplotlib
- Supervision

The original CPU environment reported PyTorch 2.14.0+cpu, CUDA unavailable, and no TensorRT installation. GPU execution therefore requires a separate compatible environment on a machine with an NVIDIA GPU and supported NVIDIA drivers.

The exact compatible CUDA, PyTorch, TensorRT, and driver versions must be verified on the target GPU system before execution.

## Installation and Setup

First, clone or copy the project to the machine on which the experiment will run.

Create a Python virtual environment if appropriate for that environment, activate it, and install the dependencies required for the selected backend.

For CPU experiments, use the project's compatible CPU dependency setup.

For TensorRT GPU experiments, configure a compatible NVIDIA driver, CUDA-enabled PyTorch installation, TensorRT runtime, and required Python dependencies. Do not install a CPU-only PyTorch build for the GPU experiment.

The dependency installation procedure may differ between Windows and Linux and must match the target GPU system.

## Stage 1: YOLO11m PyTorch Detection

The common video is processed using the YOLO11m PyTorch model.

The detection pipeline records:

- Frame number
- Timestamp
- Class ID
- Class name
- Detection confidence
- Bounding-box coordinates
- Bounding-box width and height
- Inference timing
- Processing performance

Results are stored under outputs/pt/.

Generated results include detection CSV and JSON files, metrics, summaries, reports, plots, and annotated video files.

## Stage 2: YOLO11m ONNX Detection

The YOLO11m PyTorch model is converted into ONNX format and executed using ONNX Runtime.

The same common video and nominal detection configuration are used.

The ONNX experiment measures:

- Frames processed
- Detection count
- Class distribution
- Average confidence
- Average inference time
- Inference FPS
- Complete pipeline FPS

Results are stored under outputs/onnx/.

The ONNX experiment is compared against PyTorch to study numerical and performance differences between the two inference backends.

## Stage 3: CPU MCByteTrack

The original CPU tracking experiment combines YOLO11m PyTorch detections with the ByteTrack implementation provided by Supervision.

Tracking output includes:

- Frame number
- Track ID
- Class ID
- Class name
- Confidence
- Bounding-box coordinates
- Track length
- Tracking statistics

Results are stored under outputs/mcbytetrack/.

The CPU tracking outputs are retained independently from the new GPU tracking results.

## Stage 4: TensorRT Engine Export

The YOLO11m PyTorch model is exported to a TensorRT engine on the target GPU system.

Expected engine path: models/tensorrt/yolo11m.engine

The export configuration requests FP16 precision, fixed input size 640, and batch size 1. Actual precision and engine capabilities depend on the TensorRT export configuration and target hardware.

The export process records available environment information and export timing.

The TensorRT engine should be generated and tested in the target GPU environment rather than assumed to be portable between arbitrary systems.

## Stage 5: TensorRT GPU Detection

The TensorRT engine processes the common input video using GPU inference.

The pipeline records:

- Frames processed
- Frames containing detections
- Total detections
- Class distribution
- Average detection confidence
- Average inference time
- Inference FPS
- Complete pipeline FPS
- GPU model and experiment configuration

Results are stored separately under outputs/tensorrt/.

Expected output categories include detection CSV and JSON files, metrics JSON, summary TXT, and annotated videos.

The TensorRT output folder does not overwrite the existing PyTorch or ONNX results.

## Stage 6: TensorRT GPU MCByteTrack

A separate GPU tracking pipeline uses TensorRT for YOLO11m detection and ByteTrack for tracking.

The tracker implementation is kept consistent with the original Supervision ByteTrack approach where possible, while the detector backend changes from PyTorch to TensorRT.

The GPU tracking pipeline records:

- Frame number
- Track ID
- Class ID
- Class name
- Confidence
- Bounding-box coordinates
- Track length
- Number of unique track IDs
- Average tracks per frame
- Average, minimum, and maximum track length
- Inference and pipeline performance

Results are stored under outputs/mcbytetrack_gpu/.

These results remain separate from the original CPU MCByteTrack experiment.

## Comparison 1: PyTorch vs ONNX vs TensorRT

The first comparison evaluates the three detection backends:

- YOLO11m PyTorch
- YOLO11m ONNX
- YOLO11m TensorRT

Output directory: outputs/final/backend_comparison/

The comparison includes:

- Frames processed
- Detection count
- Average confidence
- Average inference time per frame
- Inference FPS
- Complete pipeline FPS
- Total processing time, when available
- Performance charts
- CSV comparison table
- JSON comparison data
- TXT report

This comparison studies backend differences and runtime performance. Since the recorded baseline uses CPU execution for PyTorch and ONNX while TensorRT uses a GPU, the result reflects both backend and hardware differences. It must not be interpreted as a controlled, same-hardware backend benchmark.

## Comparison 2: CPU vs GPU

The second comparison evaluates the original CPU workflows against the GPU workflows.

Output directory: outputs/final/cpu_gpu_comparison/

Detection comparison:

- CPU YOLO11m PyTorch
- GPU YOLO11m TensorRT

Tracking comparison:

- CPU YOLO11m PyTorch plus MCByteTrack
- GPU YOLO11m TensorRT plus ByteTrack

Comparison parameters include:

- Frames processed
- Detection count or tracking-record count, clearly distinguished
- Average confidence
- Average inference time
- Inference FPS
- Complete pipeline FPS
- Total processing time
- Unique track IDs
- Average tracks per frame
- Average track length
- Minimum and maximum track length
- Performance plots and reports

CPU and GPU comparisons must include the hardware and software configurations. The same video, nominal confidence threshold, IoU threshold, image size, and tracking settings should be maintained where applicable.

GPU timing should be measured after suitable warm-up and with GPU execution synchronized where needed. End-to-end processing time and inference-only time must be reported separately.

## Experimental Results: Existing CPU Baseline

The following results were measured in the existing CPU experiments.

### YOLO11m PyTorch

- Frames processed: 998
- Total detections: 12,735
- Average confidence: approximately 0.7277
- Average inference time: approximately 752.63 ms per frame
- Inference FPS: approximately 1.33
- Complete pipeline FPS: approximately 1.24

### YOLO11m ONNX

- Frames processed: 998
- Total detections: 12,892
- Average confidence: approximately 0.7411
- Average inference time: approximately 1,025.71 ms per frame
- Inference FPS: approximately 0.97
- Complete pipeline FPS: approximately 0.92

### YOLO11m + MCByteTrack on CPU

- Frames processed: 998
- Tracking records: 12,302
- Unique track IDs: 37
- Person tracks: 31
- Car tracks: 6
- Average confidence: approximately 0.7427
- Average inference time: approximately 684.30 ms per frame
- Inference FPS: approximately 1.46
- Complete pipeline FPS: approximately 1.42
- Average track length: approximately 332.49 frames
- Minimum track length: 2 frames
- Maximum track length: 998 frames

These are measurements from the existing experiment. The GPU results must be generated on the target GPU system and must not be fabricated or inferred from CPU results.

## Evaluation Metrics and Ground Truth

Ground-truth detection and tracking annotations were not available in the existing experiments.

Therefore, the following accuracy metrics cannot be calculated reliably without suitable ground truth:

- Precision
- Recall
- F1-score
- mAP@50
- mAP@50-95
- IoU-based detection evaluation
- MOTA
- MOTP
- IDF1
- HOTA

These metrics must not be assigned invented values.

Detection counts, confidence values, track counts, and track lengths are descriptive statistics. They are not substitutes for ground-truth accuracy metrics.

For formal detection and tracking accuracy evaluation, labelled bounding boxes and object identities must be supplied for the evaluated video.

## Output Formats

The experiment scripts produce results in several formats.

Detection outputs:

- CSV
- JSON

Tracking outputs:

- CSV
- JSON

Metrics and reports:

- JSON
- TXT
- CSV

Visualizations:

- PNG plots

Annotated videos:

- MP4, with codec determined by the output configuration

Final comparison outputs:

- CSV tables
- JSON comparison data
- TXT reports
- PNG performance plots

## Reproducibility

To reproduce the experiments:

1. Obtain the project source code.
2. Install a compatible environment for the desired CPU or GPU backend.
3. Place the common video in input/videos/.
4. Place the YOLO11m PyTorch weights in models/pt/.
5. Run PyTorch detection.
6. Convert the model to ONNX and run ONNX inference.
7. Run the original CPU MCByteTrack pipeline.
8. On a compatible NVIDIA GPU system, export the TensorRT engine.
9. Run TensorRT GPU detection.
10. Run the separate TensorRT plus ByteTrack tracking pipeline.
11. Generate the backend comparison.
12. Generate the CPU-versus-GPU comparison.
13. Verify that all outputs and experiment configurations have been saved.

Record the GPU name, driver version, CUDA version, PyTorch version, TensorRT version, precision, image size, thresholds, batch size, and timing methodology for each GPU run.

## Hardware and Performance Considerations

The original experiments were performed using CPU execution. GPU experiments require a supported NVIDIA GPU and compatible software stack.

TensorRT performance depends on GPU architecture, precision, engine configuration, input dimensions, batch size, warm-up, and preprocessing/postprocessing overhead.

Inference FPS and complete pipeline FPS measure different aspects of performance. Both should be reported to distinguish raw inference throughput from end-to-end application performance.

Comparisons across different hardware should clearly state that hardware differences affect the measured results.

## Git and Large Files

The repository is intended to contain source code, configuration, experiment metrics, reports, tables, and plots.

Large or environment-specific files are excluded from normal Git tracking where appropriate, including:

- Python virtual environments
- Input video files
- Model weight files
- ONNX model binaries
- TensorRT engine files
- Generated annotated videos
- Temporary files

The input video and model files must be supplied locally before running the experiments.

TensorRT engines may depend on the target environment and should be generated on the machine where they will be used.

## Author

Harshitha Sunkara
