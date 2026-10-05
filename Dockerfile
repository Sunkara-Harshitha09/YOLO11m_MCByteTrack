# ============================================================
# YOLO11m + MCByteTrack
# CPU Docker Image
# ============================================================

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics

# ============================================================
# System dependencies
# ============================================================

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Working directory
# ============================================================

WORKDIR /app

# ============================================================
# Install Python dependencies
# ============================================================

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================
# Copy project source code
# ============================================================

COPY src/ ./src/

# ============================================================
# Create runtime directories
# ============================================================

RUN mkdir -p \
    /app/input/videos \
    /app/models/pt \
    /app/models/onnx \
    /app/outputs/mcbytetrack/videos \
    /app/outputs/mcbytetrack/tracking \
    /app/outputs/mcbytetrack/metrics \
    /app/outputs/mcbytetrack/logs

# ============================================================
# Run MCByteTrack
# ============================================================

CMD ["python", "src/tracking/run_mcbytetrack.py"]