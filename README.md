# YOLO11m + MCByteTrack

A computer vision object detection and multi-object tracking project using YOLO11m and MCByteTrack/ByteTrack.

The project uses the same common video for all experiments and evaluates three stages:

1. YOLO11m PyTorch detection
2. YOLO11m ONNX detection
3. YOLO11m + MCByteTrack tracking

The project also includes detailed detection, tracking, performance and comparison analysis.

---

## Project Objective

The objective of this project is to:

- Run YOLO11m using the PyTorch model
- Convert YOLO11m from PyTorch to ONNX
- Run the same video using ONNX Runtime
- Compare PyTorch and ONNX detection results
- Implement MCByteTrack using YOLO11m detections
- Analyze tracking results
- Compare PyTorch, ONNX and MCByteTrack performance
- Store experiment results in structured CSV, JSON, TXT and PNG formats

The same video and common detection configuration are used wherever applicable to maintain consistent experimental conditions.

---

## Common Input Video

Video:

PNNL_Parking_LOT(1).avi

Video properties:

- Resolution: 1920 × 1080
- FPS: 29
- Total frames: 998
- Duration: approximately 34.41 seconds
- Codec: FMP4
- File size: approximately 13.77 MB

The input video is not included in the Git repository because it is a large binary file.

Place the video locally at:

input/videos/PNNL_Parking_LOT(1).avi

---

## Model

Model:

YOLO11m

Model formats:

- PyTorch (.pt)
- ONNX (.onnx)

The model files are excluded from Git because they are large binary files.

Expected local paths:

models/pt/yolo11m.pt

models/onnx/yolo11m.onnx

---

## Experimental Configuration

Common YOLO11m configuration:

- Confidence threshold: 0.25
- IoU threshold: 0.45
- Image size: 640
- Device: CPU

Tracking:

- Tracker: MCByteTrack / ByteTrack
- Detection model: YOLO11m
- Input video: PNNL_Parking_LOT(1).avi

---

## Project Structure

YOLO11m_MCByteTrack/

    input/
        videos/

    models/
        pt/
        onnx/

    src/
        detection/
        conversion/
        tracking/
        evaluation/

    outputs/
        pt/
            videos/
            detections/
            metrics/

        onnx/
            videos/
            detections/
            metrics/

        mcbytetrack/
            videos/
            tracking/
            metrics/

        final/
            reports/
            tables/
            plots/
            summaries/

    requirements.txt
    .gitignore
    README.md

---

## Environment

Python:

3.12.10

Main libraries:

- PyTorch
- Ultralytics
- OpenCV
- ONNX
- ONNX Runtime
- NumPy
- Pandas
- Matplotlib
- Supervision

The complete dependency list is available in requirements.txt.

---

## Installation

Create the virtual environment:

python -m venv .venv

Activate the environment:

Windows PowerShell:

.\.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

---

## Workflow

### 1. YOLO11m PyTorch Detection

The common video is processed using the YOLO11m PyTorch model.

The pipeline records:

- Frame number
- Timestamp
- Class ID
- Class name
- Confidence
- Bounding box
- Bounding box dimensions
- Inference time
- Processing performance

Results are stored under:

outputs/pt/

---

### 2. YOLO11m PyTorch Analysis

The PyTorch detection results are analyzed to generate:

- Detection metrics
- Detection count per frame
- Class distribution
- Confidence statistics
- Experiment configuration
- Detailed report
- Confidence distribution plot
- Detection-per-frame plot
- Class distribution plot

Results are stored under:

outputs/pt/metrics/

---

### 3. YOLO11m PT to ONNX Conversion

The YOLO11m PyTorch model is converted to ONNX.

The ONNX model is stored locally under:

models/onnx/yolo11m.onnx

The ONNX model is validated before inference.

---

### 4. YOLO11m ONNX Detection

The same common video is processed using:

- YOLO11m ONNX
- ONNX Runtime
- CPUExecutionProvider
- Confidence threshold: 0.25
- IoU threshold: 0.45
- Image size: 640

Results are stored under:

outputs/onnx/

---

### 5. PT vs ONNX Comparison

The PyTorch and ONNX results are compared using:

- Total detection count
- Class distribution
- Confidence
- Bounding-box matching
- Frame-level detection comparison
- Average inference time
- Inference FPS
- Pipeline FPS

The comparison results are stored under:

outputs/final/

---

### 6. MCByteTrack

YOLO11m PyTorch detections are passed to the MCByteTrack/ByteTrack tracking stage.

The tracking pipeline records:

- Frame number
- Track ID
- Class ID
- Class name
- Confidence
- Bounding box
- Track duration
- Tracking statistics

Results are stored under:

outputs/mcbytetrack/

---

### 7. MCByteTrack Analysis

Tracking results are analyzed to generate:

- Track count
- Unique track IDs
- Track length statistics
- Tracks per frame
- Class distribution
- Confidence statistics
- Experiment configuration
- Detailed tracking report
- Tracking plots

Results are stored under:

outputs/mcbytetrack/metrics/

---

### 8. Final Comparison

The final comparison combines the results from:

- YOLO11m PyTorch
- YOLO11m ONNX
- YOLO11m + MCByteTrack

Final results are stored under:

outputs/final/

The final output contains:

- Reports
- Tables
- JSON summaries
- Comparison plots

---

## Experimental Results

### YOLO11m PyTorch

Frames processed:

998

Total detections:

12,735

Average confidence:

0.7277

Average inference time:

752.63 ms/frame

Inference FPS:

1.33

Complete pipeline FPS:

1.24

---

### YOLO11m ONNX

Frames processed:

998

Total detections:

12,892

Average confidence:

0.7411

Average inference time:

1,025.71 ms/frame

Inference FPS:

0.97

Complete pipeline FPS:

0.92

---

### YOLO11m + MCByteTrack

Frames processed:

998

Tracking records:

12,302

Unique track IDs:

37

Average confidence:

0.7427

Average inference time:

684.30 ms/frame

Inference FPS:

1.46

Complete pipeline FPS:

1.42

Average tracks per frame:

12.33

Average track length:

332.49 frames

Minimum track length:

2 frames

Maximum track length:

998 frames

Person tracks:

31

Car tracks:

6

---

## Class Tracking Results

MCByteTrack produced:

- 31 person tracks
- 6 car tracks
- 37 unique track IDs in total

The tracking records represent detections associated with track identities and should not be interpreted as detection accuracy.

---

## Ground-Truth Evaluation

Ground-truth detection and tracking annotations were not available for this experiment.

Therefore, the following ground-truth-dependent metrics are not reported:

- Precision
- Recall
- F1-score
- mAP@50
- mAP@50-95
- IoU
- MOTA
- MOTP
- IDF1
- HOTA

Detection counts, confidence values, FPS and tracking statistics are reported as experiment measurements and should not be interpreted as ground-truth accuracy metrics.

---

## Output Formats

The project stores results in multiple formats.

Detection outputs:

- CSV
- JSON

Tracking outputs:

- CSV
- JSON

Metrics:

- CSV
- JSON
- TXT

Visualizations:

- PNG

Final comparison:

- CSV tables
- JSON summaries
- TXT reports
- PNG plots

---

## Final Output Organization

The final comparison results are organized as:

outputs/final/

    reports/
        final_comparison_report.txt
        pt_vs_onnx_report.txt

    tables/
        final_comparison.csv
        class_comparison.csv
        ground_truth_metrics.csv
        performance_comparison.csv
        pt_vs_onnx_class_comparison.csv
        pt_vs_onnx_frame_comparison.csv
        pt_vs_onnx_matched_detections.csv

    plots/
        average_inference_time_comparison.png
        confidence_comparison.png
        detection_count_comparison.png
        fps_comparison.png
        inference_performance_comparison.png
        inference_time_comparison.png
        pipeline_fps_comparison.png
        record_count_comparison.png

    summaries/
        final_comparison.json
        pt_vs_onnx_summary.json

---

## Video Outputs

Annotated videos are generated for the PyTorch, ONNX and MCByteTrack pipelines.

The generated videos are intentionally excluded from Git because of their file size.

H.264 versions were also generated for easier video preview and playback.

---

## Reproducibility

To reproduce the project:

1. Clone the repository.
2. Create the Python virtual environment.
3. Install requirements.txt.
4. Place PNNL_Parking_LOT(1).avi inside input/videos/.
5. Place yolo11m.pt inside models/pt/.
6. Run YOLO11m PyTorch detection.
7. Analyze the PyTorch results.
8. Convert YOLO11m to ONNX.
9. Run ONNX inference.
10. Analyze the ONNX results.
11. Run MCByteTrack.
12. Analyze the tracking results.
13. Run the final comparison.

---

## Hardware

The experiments were performed using CPU execution.

GPU acceleration was not available during the experiment.

---

## Git Repository

Large binary files are intentionally excluded from the repository:

- Input video
- PyTorch model weights
- ONNX model
- Generated videos
- Python virtual environment

Experiment metrics, reports, tables, plots and source code are included.

---

## Author

Harshitha Sunkara

B.Tech Computer Science and Engineering - AI/ML

Alliance University