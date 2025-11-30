"""
FastAPI backend for Golf Course Segmentation
Runs U-Net ResNet50 model trained on Danish Golf Courses dataset
"""

import os
import io
import base64
import numpy as np
from PIL import Image
from typing import Optional

import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Configuration
MODEL_PATH = os.environ.get("MODEL_PATH", "model/best_unet_resnet50_large.keras")
IMAGE_SIZE = (512, 832)  # Height x Width (paper's resolution)
NUM_CLASSES = 6

# Class colors for visualization (RGB)
CLASS_COLORS = np.array([
    [0, 0, 0],          # 0: Background - Black
    [0, 140, 0],        # 1: Fairway - Dark green
    [0, 255, 0],        # 2: Green - Bright green
    [255, 0, 0],        # 3: Tee - Red
    [217, 230, 122],    # 4: Bunker - Sandy yellow
    [7, 15, 247]        # 5: Water - Blue
], dtype=np.uint8)

CLASS_NAMES = ["Background", "Fairway", "Green", "Tee", "Bunker", "Water"]

# Initialize FastAPI
app = FastAPI(
    title="Golf Course Segmentation API",
    description="Semantic segmentation of golf course aerial imagery using U-Net with ResNet50 encoder",
    version="1.0.0"
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None


def load_model():
    """Load the trained Keras model."""
    global model
    if model is None:
        print(f"Loading model from {MODEL_PATH}...")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        model = keras.models.load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully!")
    return model


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess image for model inference.

    Args:
        image: PIL Image

    Returns:
        Preprocessed numpy array of shape (1, 512, 832, 3)
    """
    # Convert to RGB if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Store original size for later resizing
    original_size = image.size  # (width, height)

    # Resize to model's expected input size
    image = image.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)

    # Convert to numpy and normalize to [0, 1]
    img_array = np.array(image, dtype=np.float32) / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, original_size


def postprocess_mask(logits: np.ndarray, original_size: tuple) -> tuple:
    """
    Convert model logits to segmentation mask.

    Args:
        logits: Model output of shape (1, H, W, num_classes)
        original_size: Original image size (width, height)

    Returns:
        Tuple of (class_mask, rgb_mask, statistics)
    """
    # Get class predictions (argmax)
    class_mask = np.argmax(logits[0], axis=-1).astype(np.uint8)

    # Create RGB visualization
    h, w = class_mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(NUM_CLASSES):
        rgb_mask[class_mask == class_id] = CLASS_COLORS[class_id]

    # Calculate class statistics
    total_pixels = class_mask.size
    statistics = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        pixel_count = np.sum(class_mask == class_id)
        percentage = (pixel_count / total_pixels) * 100
        statistics[class_name] = {
            "pixels": int(pixel_count),
            "percentage": round(percentage, 2)
        }

    # Resize masks back to original size
    class_mask_pil = Image.fromarray(class_mask)
    class_mask_pil = class_mask_pil.resize(original_size, Image.NEAREST)

    rgb_mask_pil = Image.fromarray(rgb_mask)
    rgb_mask_pil = rgb_mask_pil.resize(original_size, Image.NEAREST)

    return np.array(class_mask_pil), np.array(rgb_mask_pil), statistics


def mask_to_base64(mask: np.ndarray) -> str:
    """Convert numpy mask to base64 PNG string."""
    img = Image.fromarray(mask)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class SegmentationResponse(BaseModel):
    """Response model for segmentation endpoint."""
    success: bool
    overlayData: str  # Base64 encoded RGB mask
    classMask: str    # Base64 encoded class indices
    statistics: dict  # Per-class pixel counts and percentages
    message: Optional[str] = None


class Base64ImageRequest(BaseModel):
    """Request model for base64 image input."""
    imageData: str  # Base64 encoded image (with or without data URL prefix)


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    try:
        load_model()
    except Exception as e:
        print(f"Warning: Could not load model on startup: {e}")
        print("Model will be loaded on first request.")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": "U-Net ResNet50",
        "classes": CLASS_NAMES,
        "input_size": f"{IMAGE_SIZE[1]}x{IMAGE_SIZE[0]}",
        "description": "Golf course semantic segmentation API"
    }


@app.get("/health")
async def health():
    """Health check for container orchestration."""
    return {"status": "ok"}


@app.post("/segment", response_model=SegmentationResponse)
async def segment_image(file: UploadFile = File(...)):
    """
    Segment a golf course image uploaded as a file.

    Args:
        file: Image file (JPEG, PNG)

    Returns:
        Segmentation result with overlay, class mask, and statistics
    """
    try:
        # Load model if not already loaded
        model = load_model()

        # Read and validate image
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Preprocess
        img_array, original_size = preprocess_image(image)

        # Run inference
        logits = model.predict(img_array, verbose=0)

        # Postprocess
        class_mask, rgb_mask, statistics = postprocess_mask(logits, original_size)

        # Convert to base64
        overlay_b64 = mask_to_base64(rgb_mask)
        class_mask_b64 = mask_to_base64(class_mask)

        return SegmentationResponse(
            success=True,
            overlayData=f"data:image/png;base64,{overlay_b64}",
            classMask=f"data:image/png;base64,{class_mask_b64}",
            statistics=statistics,
            message="Segmentation completed successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


@app.post("/segment-base64", response_model=SegmentationResponse)
async def segment_base64(request: Base64ImageRequest):
    """
    Segment a golf course image from base64 string.

    Args:
        request: JSON with imageData field containing base64 image

    Returns:
        Segmentation result with overlay, class mask, and statistics
    """
    try:
        # Load model if not already loaded
        model = load_model()

        # Parse base64 image
        image_data = request.imageData

        # Remove data URL prefix if present
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]

        # Decode base64
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")

        # Preprocess
        img_array, original_size = preprocess_image(image)

        # Run inference
        logits = model.predict(img_array, verbose=0)

        # Postprocess
        class_mask, rgb_mask, statistics = postprocess_mask(logits, original_size)

        # Convert to base64
        overlay_b64 = mask_to_base64(rgb_mask)
        class_mask_b64 = mask_to_base64(class_mask)

        return SegmentationResponse(
            success=True,
            overlayData=f"data:image/png;base64,{overlay_b64}",
            classMask=f"data:image/png;base64,{class_mask_b64}",
            statistics=statistics,
            message="Segmentation completed successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
