"""
Hugging Face Spaces entry point for Golf Course Segmentation API

This file is for HF Spaces deployment. It loads the model from HF Hub
and exposes the FastAPI app.
"""

import os
from huggingface_hub import hf_hub_download

# Download model from HF Hub if not present
MODEL_REPO = os.environ.get("MODEL_REPO", "your-username/golf-segmentation-model")
MODEL_FILENAME = "best_unet_resnet50_large.keras"
MODEL_DIR = "model"

def download_model():
    """Download model from Hugging Face Hub."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)

    if not os.path.exists(model_path):
        print(f"Downloading model from {MODEL_REPO}...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=MODEL_FILENAME,
                local_dir=MODEL_DIR
            )
            print(f"Model downloaded to {downloaded_path}")
        except Exception as e:
            print(f"Could not download from HF Hub: {e}")
            print("Please ensure model file is present in /app/model/")

# Download model on import
download_model()

# Set model path environment variable
os.environ["MODEL_PATH"] = os.path.join(MODEL_DIR, MODEL_FILENAME)

# Import and expose the FastAPI app
from main import app
