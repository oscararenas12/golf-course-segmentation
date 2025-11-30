---
title: Golf Course Segmentation API
emoji: ⛳
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Golf Course Segmentation API

Semantic segmentation of golf course aerial imagery using U-Net with ResNet50 encoder.

## Model

- **Architecture**: U-Net with ResNet50 encoder (ImageNet pre-trained)
- **Input Size**: 832×512 pixels
- **Classes**: Background, Fairway, Green, Tee, Bunker, Water
- **Dataset**: Danish Golf Courses Orthophotos

## API Endpoints

### `GET /`
Health check and API info.

### `GET /health`
Simple health check for container orchestration.

### `POST /segment`
Segment an uploaded image file.

**Request**: `multipart/form-data` with `file` field containing JPEG/PNG image

**Response**:
```json
{
  "success": true,
  "overlayData": "data:image/png;base64,...",
  "classMask": "data:image/png;base64,...",
  "statistics": {
    "Background": {"pixels": 12345, "percentage": 30.5},
    "Fairway": {"pixels": 23456, "percentage": 45.2},
    ...
  },
  "message": "Segmentation completed successfully"
}
```

### `POST /segment-base64`
Segment a base64-encoded image.

**Request**:
```json
{
  "imageData": "data:image/jpeg;base64,..."
}
```

**Response**: Same as `/segment`

## Usage Example

```python
import requests

# File upload
with open("golf_course.jpg", "rb") as f:
    response = requests.post(
        "https://your-space.hf.space/segment",
        files={"file": f}
    )
    result = response.json()

# Base64 input
import base64
with open("golf_course.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "https://your-space.hf.space/segment-base64",
    json={"imageData": f"data:image/jpeg;base64,{b64}"}
)
```

## Class Colors

| Class | Color | RGB |
|-------|-------|-----|
| Background | Black | (0, 0, 0) |
| Fairway | Dark Green | (0, 140, 0) |
| Green | Bright Green | (0, 255, 0) |
| Tee | Red | (255, 0, 0) |
| Bunker | Sandy Yellow | (217, 230, 122) |
| Water | Blue | (7, 15, 247) |

## Local Development

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Deployment

This Space uses Docker. The model file must be placed in `/app/model/best_unet_resnet50_large.keras`.

For large models (>10GB LFS limit), upload the model to a separate HF repo and set `MODEL_REPO` environment variable.
