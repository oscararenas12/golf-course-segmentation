# Golf Course Segmentation

Deep learning semantic segmentation of golf course features from aerial imagery, with an interactive Next.js web application.

## Overview

This project combines machine learning and web development for golf course analysis:

**Machine Learning:**
- U-Net with ResNet50 encoder for 6-class segmentation
- MobileNetV2 binary classifier for golf course detection
- Trained on Danish golf course orthophotos + UC Merced dataset
- Jupyter notebooks with Colab support

**Web Application:**
- Next.js 15 + TypeScript + Tailwind CSS
- Google Maps satellite imagery integration
- Three workflows: Dataset Creation, Classification Labeling, Segmentation Analysis
- Real-time AI segmentation via HuggingFace Spaces API

## Segmentation Classes

| Class | Color | Description |
|-------|-------|-------------|
| Background | Black | Non-golf areas |
| Fairway | Dark Green | Main playing surface |
| Green | Bright Green | Putting surface |
| Tee | Red | Teeing areas |
| Bunker | Sandy Yellow | Sand traps |
| Water | Blue | Water hazards |

## Project Structure

```
golf-course-segmentation/
├── app/                          # Next.js pages
│   ├── page.tsx                  # Home - workflow selector
│   ├── dataset/page.tsx          # Segmentation dataset creation
│   ├── classification/page.tsx   # Binary labeling workflow
│   └── segmentation/page.tsx     # AI segmentation analysis
├── components/                   # React components
│   ├── AnnotationCanvas.tsx      # Drawing tool for ground truth
│   ├── MapArea.tsx               # Google Maps integration
│   ├── DatasetCapture.tsx        # Dataset capture UI
│   ├── ClassificationCapture.tsx # Binary labeling UI
│   └── ui/                       # Shadcn UI components
├── api/                          # FastAPI backend (HuggingFace Spaces)
│   └── main.py                   # Segmentation API
├── notebooks/                    # ML training
│   └── unet_resnet50_large.ipynb # U-Net segmentation model
├── classifier/                   # Classification model
│   └── train_classifier.ipynb    # Binary golf classifier
├── utils/                        # Utilities
│   └── datasetStorage.ts         # IndexedDB operations
└── types/                        # TypeScript types
```

## Getting Started

### Web Application

1. **Install dependencies:**
```bash
npm install
```

2. **Set up environment variables:**
```bash
cp .env.example .env.local
```

Add your Google Maps API key:
```
NEXT_PUBLIC_GOOGLE_MAPS_API_KEY=your_api_key
NEXT_PUBLIC_SEGMENTATION_API_URL=https://elo0oo0-golf-segmentation-api.hf.space
```

3. **Run development server:**
```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

### ML Training (Local or Colab)

Both notebooks support Google Colab and local execution:

**U-Net Segmentation Model:**
```bash
# Open in Jupyter or upload to Colab
notebooks/unet_resnet50_large.ipynb
```

**Binary Classifier:**
```bash
classifier/train_classifier.ipynb
```

**Dependencies (auto-installed on Colab):**
```bash
pip install tensorflow kagglehub datasets pillow numpy matplotlib
```

## Workflows

### 1. Segmentation Dataset Creation (`/dataset`)
- Capture satellite imagery from Google Maps
- Annotate ground truth masks with 6-class brush tool
- Auto-segment using trained U-Net model
- Export ZIP with images, masks, and metadata CSV

### 2. Classification Dataset Creation (`/classification`)
- Quick capture and binary labeling (Golf / Not Golf)
- Track class balance statistics
- Export ZIP with labeled folders

### 3. Segmentation Analysis (`/segmentation`)
- Search any golf course location
- One-click AI segmentation
- Overlay visualization with opacity control
- Per-class pixel statistics

## Model Architecture

### U-Net (Segmentation)
- **Encoder**: ResNet50 (ImageNet pretrained)
- **Decoder**: Transposed convolutions with skip connections
- **Input**: 832×512 RGB
- **Output**: 6-class probability map
- **Training**: Mixed precision (float16), AdamW optimizer

### MobileNetV2 (Classification)
- **Base**: MobileNetV2 (frozen, ImageNet pretrained)
- **Head**: GlobalAveragePooling → Dense(128) → Dense(1, sigmoid)
- **Input**: 224×224 RGB
- **Output**: Binary probability (golf / not golf)

## API

The segmentation API is deployed on HuggingFace Spaces:

**Endpoint:** `https://elo0oo0-golf-segmentation-api.hf.space`

```bash
# Health check
GET /

# Segment image (file upload)
POST /segment
Content-Type: multipart/form-data

# Segment image (base64)
POST /segment-base64
Content-Type: application/json
{"imageData": "data:image/jpeg;base64,..."}
```

## Technology Stack

**Frontend:**
- Next.js 15 (App Router)
- TypeScript
- Tailwind CSS
- Radix UI / Shadcn
- Google Maps JavaScript API
- IndexedDB (client-side storage)

**Backend:**
- FastAPI
- TensorFlow/Keras
- HuggingFace Spaces

**ML:**
- TensorFlow 2.x
- Mixed precision training
- Kagglehub for dataset loading

## Datasets

1. **Danish Golf Courses Orthophotos** (Kaggle)
   - 1,123 orthophotos from 107 golf courses
   - 6-class pixel-level annotations
   - Used for segmentation training

2. **UC Merced Land Use Dataset** (HuggingFace)
   - 21 land use classes including golf
   - Used for classification training

## References

- [Danish Golf Courses Dataset](https://www.kaggle.com/datasets/jacotaco/danish-golf-courses-orthophotos)
- [UC Merced Dataset](https://huggingface.co/datasets/blanchon/UC_Merced)
- [U-Net Segmentation Notebook](https://www.kaggle.com/code/viniciussmatthiesen/semantic-segmentation-of-danish-golf-courses-u-net)
- [MobileNets Paper](https://arxiv.org/abs/1704.04861)

## License

MIT

## Contact

For access requests: oscar.arenas01@student.csulb.edu
