# Hugging Face Spaces Docker template for FastAPI + TensorFlow
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Create model directory
RUN mkdir -p model

# Copy model file (you'll need to upload this separately or use HF Hub)
# The model will be downloaded from HF Hub or mounted as a volume
# COPY best_unet_resnet50_large.keras model/

# Set environment variables
ENV MODEL_PATH=/app/model/best_unet_resnet50_large.keras
ENV PYTHONUNBUFFERED=1

# Expose port (HF Spaces uses 7860)
EXPOSE 7860

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
