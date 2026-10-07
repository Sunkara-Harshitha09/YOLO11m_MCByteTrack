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
- Provide a Docker-based environment for reproducible execution
- Provide a standalone MCByteTrack application for easier video processing

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

### Tracker Implementation Clarification

The current MCByteTrack implementation uses the ByteTrack tracker provided by the Supervision library.

The implementation uses:

- YOLO11m for object detection
- Supervision ByteTrack for multi-object tracking
- OpenCV for video processing
- FFmpeg for H.264 video encoding

The current implementation does not use:

- BoxMOT
- BoT-SORT
- SAM
- DeepSORT

The project name remains MCByteTrack, while the actual tracker implementation is based on Supervision ByteTrack.

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
            logs/

        final/
            reports/
            tables/
            plots/
            summaries/

    MCByteTrack_App/
        tracker/
            mcbytetrack.py

        model/
            yolo11m.pt

        output/
            videos/
            tracking/
            metrics/
            logs/

        main.py

    requirements.txt
    Dockerfile
    docker-compose.yml
    .gitignore
    README.md

---

# MCByteTrack Standalone Application

A standalone application has been added inside:

MCByteTrack_App/

The purpose of this application is to provide a simple entry point for running YOLO11m + MCByteTrack on an input video.

The application contains its own tracker, model and output directories.

### Application Structure

MCByteTrack_App/

    tracker/
        mcbytetrack.py

    model/
        yolo11m.pt

    output/
        videos/
        tracking/
        metrics/
        logs/

    main.py

---

## Application Components

### main.py

main.py is the main entry point of the standalone application.

It is responsible for:

- Starting the application
- Displaying application information
- Locating the YOLO11m model
- Locating the output directory
- Asking the user for an input video path
- Validating the input video
- Starting the MCByteTrack pipeline
- Displaying processing progress
- Displaying completion or error information

Run the application from the MCByteTrack_App directory using:

python .\main.py

---

### tracker/mcbytetrack.py

This file contains the detection and tracking pipeline.

Responsibilities include:

- Loading YOLO11m
- Initializing ByteTrack
- Reading the input video
- Processing frames
- Running YOLO11m detection
- Converting detections to Supervision format
- Updating ByteTrack
- Assigning tracking IDs
- Drawing bounding boxes and labels
- Generating tracking information
- Generating metrics
- Generating logs
- Encoding the final output video using H.264

---

### model/yolo11m.pt

This directory contains the YOLO11m PyTorch model used by the standalone application.

Expected path:

MCByteTrack_App/model/yolo11m.pt

---

### output/

All standalone application outputs are stored inside:

MCByteTrack_App/output/

The output directory is organized into:

output/
    videos/
    tracking/
    metrics/
    logs/

---

## Standalone Application Processing Flow

The application follows this flow:

Input Video

↓

YOLO11m Detection

↓

Supervision ByteTrack

↓

Tracking IDs

↓

Frame Annotation

↓

H.264 Encoding

↓

Output Video

↓

Tracking Data + Metrics + Logs

---

## Standalone Application Execution

Navigate to:

MCByteTrack_App

Then run:

python .\main.py

The application displays:

- Application folder
- Tracker
- Model
- Output folder

It then asks the user to enter the input video path.

Example:

C:\path\to\video.avi

The application validates the path before processing starts.

---

## Standalone Application Output

All standalone application results are stored under:

MCByteTrack_App/output/

### output/videos/

Contains the final annotated video.

The final video is encoded using H.264 through FFmpeg.

### output/tracking/

Contains frame-level tracking information.

Tracking information can include:

- Frame number
- Track ID
- Class ID
- Class name
- Confidence
- Bounding box coordinates
- Tracking information

### output/metrics/

Contains generated tracking and performance statistics.

Metrics can include:

- Total frames
- Processed frames
- Total detections
- Unique track IDs
- Average active tracks
- Maximum active tracks
- Average track lifetime
- Longest track
- Processing time
- FPS
- Latency statistics

### output/logs/

Contains execution and FFmpeg logs.

These logs can be used for:

- Debugging
- Checking FFmpeg execution
- Checking encoding problems
- Reviewing application execution

---

## FFmpeg and H.264 Encoding

The standalone application uses FFmpeg for final video encoding.

The required video encoder is:

libx264

Verify FFmpeg:

ffmpeg -version

Verify H.264 support:

ffmpeg -encoders | findstr libx264

The FFmpeg executable must be available through the system PATH.

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
- psutil

The complete dependency list is available in requirements.txt.

---

## Installation

### Python Environment

Create the virtual environment:

python -m venv .venv

Activate the environment on Windows PowerShell:

.\.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

---

## Docker Environment

The project also provides Docker support so that the project can be reproduced without manually configuring the Python environment.

### Docker Requirements

Install:

- Docker Desktop
- WSL 2 on Windows if required by Docker Desktop
- Linux containers enabled in Docker Desktop

Verify Docker:

docker --version

Verify Docker Compose:

docker compose version

---

## Docker Input Requirements

Because the input video and model weights are large binary files, they are not included in Git.

Before running the Docker pipeline, make sure the following files exist locally:

input/videos/PNNL_Parking_LOT(1).avi

models/pt/yolo11m.pt

The project mounts the local input, models and outputs directories into the Docker container.

---

## Docker Build

From the project root directory:

docker compose build

This builds the Docker image using:

Dockerfile

and installs the dependencies specified in:

requirements.txt

---

## Docker Run

After the Docker image is built, run:

docker compose up

The container executes the MCByteTrack pipeline automatically.

The configured Docker service is:

mcbytetrack

The Docker container runs:

src/tracking/run_mcbytetrack.py

---

## Docker Run with Automatic Rebuild

If the Dockerfile or dependencies have been changed, rebuild and run using:

docker compose up --build

---

## Docker Output

Docker writes the generated results back to the local project because the following directories are mounted:

./input → /app/input

./models → /app/models

./outputs → /app/outputs

The MCByteTrack outputs are available locally under:

outputs/mcbytetrack/

Important output directories include:

outputs/mcbytetrack/metrics/

outputs/mcbytetrack/tracking/

outputs/mcbytetrack/videos/

outputs/mcbytetrack/logs/

The generated videos are excluded from Git because of their large size.

---

## Docker Configuration

The project uses:

Dockerfile

docker-compose.yml

The Docker image is based on:

python:3.12-slim

The Docker environment installs the required system dependencies and Python packages before running the tracking pipeline.

The Docker image is currently configured for CPU execution.

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

## 2. YOLO11m PyTorch Analysis

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

## 3. YOLO11m PT to ONNX Conversion

The YOLO11m PyTorch model is converted to ONNX.

The ONNX model is stored locally under:

models/onnx/yolo11m.onnx

The ONNX model is validated before inference.

---

## 4. YOLO11m ONNX Detection

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

## 5. PT vs ONNX Comparison

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

## 6. MCByteTrack

YOLO11m detections are passed to the MCByteTrack/ByteTrack tracking stage.

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

## 7. MCByteTrack Analysis

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

## 8. Final Comparison

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

## Docker MCByteTrack Execution Results

The Dockerized MCByteTrack pipeline was also executed successfully using the same input video and YOLO11m model.

### Docker Direct Run

Results from the Docker image execution:

- Frames processed: 998
- Total detections: 12,735
- Unique track IDs: 37
- Average active tracks: 12.33
- Maximum active tracks: 18
- Average track lifetime: 332.49 frames
- Longest track lifetime: 998 frames
- Average FPS: approximately 1.65
- Average latency: approximately 605.78 ms/frame
- P95 latency: approximately 809.74 ms
- Total processing time: approximately 621.99 seconds

### Docker Compose Run

Results from Docker Compose execution:

- Frames processed: 998
- Total detections: 12,735
- Unique track IDs: 37
- Average active tracks: 12.33
- Maximum active tracks: 18
- Average track lifetime: 332.49 frames
- Longest track lifetime: 998 frames
- Average FPS: approximately 1.51
- Average latency: approximately 663.01 ms/frame
- P95 latency: approximately 1010.18 ms
- Total processing time: approximately 680.43 seconds

The small performance difference between direct Docker execution and Docker Compose execution is expected because of runtime and container execution conditions.

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

### Detection Outputs

- CSV
- JSON

### Tracking Outputs

- CSV
- JSON

### Metrics

- CSV
- JSON
- TXT

### Visualizations

- PNG

### Final Comparison

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

## MCByteTrack Output Organization

The MCByteTrack results are organized under:

outputs/mcbytetrack/

    videos/
        annotated tracking videos

    tracking/
        tracking CSV outputs

    metrics/
        tracking metrics JSON
        tracking statistics CSV
        summary TXT
        tracking plots

    logs/
        execution logs

---

## Standalone MCByteTrack App Output Organization

The standalone application results are organized separately under:

MCByteTrack_App/output/

    videos/
        final H.264 annotated videos

    tracking/
        tracking CSV and tracking information

    metrics/
        performance and tracking metrics

    logs/
        application and FFmpeg logs

This keeps the standalone application outputs separate from the original experiment outputs under:

outputs/

---

## Video Outputs

Annotated videos are generated for the PyTorch, ONNX and MCByteTrack pipelines.

The generated videos are intentionally excluded from Git because of their file size.

H.264 versions are also generated for easier video preview and playback.

---

## Reproducibility

The project can be reproduced using either a local Python environment or Docker.

### Python Reproducibility

To reproduce the project using Python:

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

### Standalone Application Reproducibility

To run the standalone MCByteTrack application:

1. Navigate to MCByteTrack_App/.
2. Ensure model/yolo11m.pt exists.
3. Ensure FFmpeg is installed and available in PATH.
4. Activate the Python environment.
5. Run main.py.
6. Provide the input video path.
7. Check the results under MCByteTrack_App/output/.

---

## Docker Reproducibility

To reproduce the Dockerized MCByteTrack environment:

1. Clone the repository.
2. Install Docker Desktop.
3. Make sure Docker is running.
4. Place PNNL_Parking_LOT(1).avi inside input/videos/.
5. Place yolo11m.pt inside models/pt/.
6. Build the Docker image.
7. Run the Docker Compose service.
8. Check the generated results under outputs/mcbytetrack/.

### Docker Commands

Build:

docker compose build

Run:

docker compose up

Build and run after changes:

docker compose up --build

Stop the running service:

Ctrl + C

Check containers:

docker ps -a

---

## Docker Volume Mapping

The Docker Compose configuration uses the following directory mappings:

Local Project                  Docker Container

./input                        → /app/input

./models                       → /app/models

./outputs                      → /app/outputs

This allows the container to read the local video and model and write the generated results back to the project directory.

---

## Docker Image

Docker image name:

yolo11m-mcbytetrack:latest

Docker container name:

yolo11m-mcbytetrack

The image uses Python 3.12 and installs the dependencies defined in requirements.txt.

---

## Hardware

The experiments were performed using CPU execution.

GPU acceleration was not available during the experiment.

Docker execution is also configured for CPU processing.

---

## Git Repository

Large binary files are intentionally excluded from the repository:

- Input video
- PyTorch model weights
- ONNX model
- Generated videos
- Python virtual environment

Experiment metrics, reports, tables, plots, Docker configuration and source code are included.

The repository contains:

- Source code
- Requirements
- Dockerfile
- Docker Compose configuration
- Metrics
- Reports
- Tables
- Plots
- README documentation

---

## .gitignore

The project uses .gitignore to prevent large files, local environments, generated videos and temporary files from being committed.

Important excluded items include:

.venv/

input/videos/

models/pt/*.pt

models/onnx/*.onnx

outputs/pt/videos/

outputs/onnx/videos/

outputs/mcbytetrack/videos/

runs/

Metrics and reports are retained in Git.

---

## Requirements

The project dependencies are pinned in:

requirements.txt

Current main dependencies include:

ultralytics==8.4.167

opencv-python-headless==4.13.0.92

numpy==2.2.6

pandas==2.3.3

matplotlib==3.10.8

psutil==7.1.0

supervision==0.27.0

---

## Project Execution Summary

The overall project workflow is:

Input Video

↓

YOLO11m PyTorch Detection

↓

PyTorch Metrics

↓

PT → ONNX Conversion

↓

YOLO11m ONNX Detection

↓

ONNX Metrics

↓

PT vs ONNX Comparison

↓

YOLO11m Detections

↓

MCByteTrack / ByteTrack

↓

Tracking Metrics

↓

Final Comparison

↓

Reports + Tables + JSON + Plots

The standalone application provides an additional simplified execution path:

Input Video

↓

MCByteTrack_App/main.py

↓

YOLO11m

↓

Supervision ByteTrack

↓

H.264 Video

↓

MCByteTrack_App/output/

---

## Conclusion

This project evaluates YOLO11m object detection using PyTorch and ONNX Runtime and performs multi-object tracking using the MCByteTrack/ByteTrack tracking stage.

The project maintains common experimental settings where applicable and stores structured results for analysis and comparison.

The project also provides Docker support to make the MCByteTrack execution environment reproducible across different machines.

A standalone MCByteTrack application is also provided under MCByteTrack_App for simplified video processing. The standalone application keeps its model, tracker and generated outputs organized separately.

The current experiment was performed using CPU execution with:

- YOLO11m
- 1920 × 1080 input video
- 998 frames
- Confidence threshold: 0.25
- IoU threshold: 0.45
- Image size: 640
- CPU execution

Ground-truth annotations were not available, so the reported measurements focus on detections, confidence, processing performance and tracking statistics rather than accuracy metrics requiring annotated ground truth.

---

## Author

Harshitha Sunkara