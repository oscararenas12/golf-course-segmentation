# Golf Course Segmentation - Complete Documentation

A comprehensive web application for semantic segmentation of golf course aerial imagery using deep learning. This tool enables dataset creation, binary classification labeling, and real-time segmentation analysis of golf courses from satellite imagery.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Technology Stack](#technology-stack)
4. [Architecture](#architecture)
5. [Pages & Workflows](#pages--workflows)
6. [Components](#components)
7. [API Backend](#api-backend)
8. [Data Storage](#data-storage)
9. [Model Details](#model-details)
10. [Scripts](#scripts)
11. [Security](#security)
12. [References](#references)

---

## Project Overview

This application provides three main workflows for golf course image analysis:

1. **Segmentation Dataset Creation** - Capture satellite imagery and create annotated ground truth masks for training semantic segmentation models
2. **Classification Dataset Creation** - Binary labeling of images as "Golf" or "Not Golf" for training classification models
3. **Segmentation Analysis** - Real-time semantic segmentation of golf course imagery using a trained U-Net model

The segmentation model identifies 6 classes in golf course imagery:
- **Background** (Black) - Non-golf course areas
- **Fairway** (Dark Green) - Main playing surface
- **Green** (Bright Green) - Putting surfaces
- **Tee** (Red) - Teeing areas
- **Bunker** (Sandy Yellow) - Sand traps
- **Water** (Blue) - Water hazards

---

## Features

### 1. Segmentation Dataset Workflow (`/dataset`)

- **Map Integration**: Google Maps satellite view with search functionality
- **Image Capture**: Capture current map view as high-resolution satellite imagery
- **Annotation Tool**: Full-featured annotation canvas for creating ground truth masks
  - 6-class brush tool with configurable brush sizes (5-100px)
  - Eraser tool for corrections
  - Undo/Redo functionality
  - Auto-segment using the trained U-Net model
  - Real-time overlay preview with adjustable opacity
  - Keyboard shortcuts (1-6 for classes, E for eraser, Z for undo)
- **Entry Management**:
  - Course name and metadata input
  - View, edit, and delete saved entries
  - Thumbnail previews with annotation status
- **Dataset Export**:
  - ZIP file export with organized folder structure
  - Separate folders for satellite images, masks, and ground truth
  - CSV metadata file with coordinates and timestamps
  - Multiple format support (Training-ready, TensorFlow, COCO)

### 2. Classification Dataset Workflow (`/classification`)

- **Quick Capture**: Streamlined image capture for binary labeling
- **Two-Button Labeling**: Simple "Golf" or "Not Golf" classification
- **Statistics Dashboard**:
  - Real-time count of labeled images per class
  - Class balance indicator with visual progress bar
  - Total dataset size tracking
- **Dataset Export**:
  - ZIP file with `golf/` and `not_golf/` folders
  - Labels CSV file with coordinates and timestamps

### 3. Segmentation Analysis Workflow (`/segmentation`)

- **Search Any Location**: Find golf courses worldwide using Google Maps search
- **AI-Powered Segmentation**: One-click analysis using trained U-Net model
- **Results Visualization**:
  - Overlay mask on satellite imagery
  - Adjustable overlay opacity slider
  - Per-class statistics (pixel count and percentage)
- **Class Legend**: Color-coded legend for all 6 segmentation classes

### Common Features

- **Password Protection**: Gated access with configurable password
- **Responsive Design**: Full-screen map with collapsible sidebar
- **Toast Notifications**: Real-time feedback for all operations
- **Dark Theme**: Modern slate color scheme optimized for long sessions

---

## Technology Stack

### Frontend
- **Framework**: Next.js 15 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS with tailwindcss-animate
- **UI Components**: Radix UI primitives (Dialog, Slider, ScrollArea, etc.)
- **Icons**: Lucide React
- **Notifications**: Sonner toast library
- **Maps**: Google Maps JavaScript API (@googlemaps/js-api-loader)
- **Screenshot**: html2canvas for map capture
- **ZIP Export**: JSZip for dataset packaging

### Backend (API)
- **Framework**: FastAPI (Python)
- **ML Framework**: TensorFlow/Keras
- **Image Processing**: Pillow (PIL), NumPy
- **Deployment**: Hugging Face Spaces

### Data Storage
- **Client-side**: IndexedDB for large dataset storage
- **Migration**: Automatic localStorage to IndexedDB migration

---

## Architecture

```
golf-course-segmentation/
├── app/                          # Next.js App Router pages
│   ├── layout.tsx                # Root layout with PasswordGate
│   ├── page.tsx                  # Home page with workflow cards
│   ├── dataset/page.tsx          # Segmentation dataset workflow
│   ├── classification/page.tsx   # Binary classification workflow
│   └── segmentation/page.tsx     # Segmentation analysis workflow
├── components/
│   ├── AnnotationCanvas.tsx      # Full annotation tool with drawing
│   ├── ClassificationCapture.tsx # Binary labeling component
│   ├── DatasetCapture.tsx        # Dataset creation component
│   ├── DatasetViewer.tsx         # View/edit saved entries
│   ├── ExportDatasetDialog.tsx   # Export configuration dialog
│   ├── Footer.tsx                # Logout and contact info
│   ├── Header.tsx                # Navigation header
│   ├── Legend.tsx                # Segmentation class legend
│   ├── MapArea.tsx               # Google Maps integration
│   ├── PasswordGate.tsx          # Authentication gate
│   ├── Sidebar.tsx               # Main sidebar wrapper
│   ├── Statistics.tsx            # Segmentation statistics display
│   └── ui/                       # Radix UI component library
├── api/                          # FastAPI backend
│   └── main.py                   # Segmentation API endpoints
├── utils/
│   └── datasetStorage.ts         # IndexedDB operations
├── types/
│   ├── dataset.ts                # Dataset entry interfaces
│   └── segmentation.ts           # Segmentation result types
└── scripts/
    └── visualize_dataset_examples.py  # Paper figure generation
```

---

## Pages & Workflows

### Home Page (`/`)
- Workflow selector with 3 card options
- About section with project description
- References & Data Sources with clickable links
- Footer with contact information

### Segmentation Dataset (`/dataset`)
- Full-screen Google Maps with search
- Right sidebar with:
  - Capture button
  - Current entry panel (thumbnail, metadata, actions)
  - Dataset statistics (count, size)
  - Export and Clear buttons
  - Saved entries list with View/Edit/Delete

### Classification Dataset (`/classification`)
- Full-screen Google Maps with search
- Right sidebar with:
  - Capture button
  - Image preview with Golf/Not Golf buttons
  - Dataset statistics with class balance
  - Export and Clear buttons

### Segmentation Analysis (`/segmentation`)
- Full-screen Google Maps with search
- Right sidebar with:
  - Analyze button (triggers API call)
  - Loading state with spinner
  - Results with overlay toggle and opacity slider
  - Statistics per class
  - Class legend

---

## Components

### AnnotationCanvas.tsx
Full-featured annotation tool for creating ground truth masks.

**Features:**
- Canvas-based drawing with configurable brush
- 6 class colors matching model training data
- Eraser tool for corrections
- Undo/Redo stack (50 states max)
- Auto-segment integration with HuggingFace API
- Overlay preview with opacity control
- Keyboard shortcuts:
  - `1-6`: Select class
  - `E`: Eraser tool
  - `Z`: Undo
  - `Y`: Redo
- Touch support for mobile devices

**Class Colors (RGBA):**
```javascript
Background: [0, 0, 0, 255]        // Black
Fairway:    [0, 140, 0, 255]      // Dark Green
Green:      [0, 255, 0, 255]      // Bright Green
Tee:        [255, 0, 0, 255]      // Red
Bunker:     [217, 230, 122, 255]  // Sandy Yellow
Water:      [7, 15, 247, 255]     // Blue
```

### MapArea.tsx
Google Maps integration component.

**Features:**
- Satellite view with high zoom levels
- Search box with autocomplete
- Map capture using html2canvas
- Configurable initial center and zoom
- Responsive container sizing

### DatasetViewer.tsx
Saved entries management component.

**Features:**
- Scrollable list of saved entries
- Thumbnail preview with annotation status badges
- View mode with full-size image display
- Edit annotation functionality
- Delete with confirmation dialog
- Refresh trigger for real-time updates

### ExportDatasetDialog.tsx
Dataset export configuration.

**Export Formats:**
- **Training-ready**: Simple folder structure
- **TensorFlow**: TFRecord-compatible structure
- **COCO**: COCO annotation format

**Export Contents:**
- `images/` - Satellite imagery (JPEG)
- `masks/` - Segmentation masks (PNG)
- `ground_truth/` - Manual annotations (PNG)
- `metadata.csv` - Entry metadata

---

## API Backend

### Endpoints

#### `GET /`
Health check with model information.

**Response:**
```json
{
  "status": "healthy",
  "model": "U-Net ResNet50",
  "classes": ["Background", "Fairway", "Green", "Tee", "Bunker", "Water"],
  "input_size": "832x512",
  "description": "Golf course semantic segmentation API"
}
```

#### `POST /segment`
Segment an uploaded image file.

**Request:** Multipart form with image file
**Response:**
```json
{
  "success": true,
  "overlayData": "data:image/png;base64,...",
  "classMask": "data:image/png;base64,...",
  "statistics": {
    "Background": {"pixels": 12345, "percentage": 25.5},
    "Fairway": {"pixels": 23456, "percentage": 48.2},
    ...
  },
  "message": "Segmentation completed successfully"
}
```

#### `POST /segment-base64`
Segment a base64-encoded image.

**Request:**
```json
{
  "imageData": "data:image/jpeg;base64,..."
}
```

**Response:** Same as `/segment`

### Model Configuration
- **Input Size**: 832 x 512 pixels (Width x Height)
- **Output**: 6-class probability map
- **Preprocessing**: RGB normalization to [0, 1]
- **Postprocessing**: Argmax + resize to original dimensions

---

## Data Storage

### IndexedDB Schema

**Database:** `golf_course_db`
**Store:** `dataset`

**Entry Structure:**
```typescript
interface DatasetEntry {
  id: string;                    // Unique identifier
  filename: string;              // Generated filename
  courseName: string;            // User-provided name
  timestamp: string;             // ISO timestamp
  location: {
    name: string;
    lat: number;
    lng: number;
    zoomLevel: number;
  };
  captureBox: {
    width: number;               // 1664px default
    height: number;              // 1024px default
    bounds: {
      north: number;
      south: number;
      east: number;
      west: number;
    };
  };
  images: {
    satellite?: string;          // Base64 satellite image
    mask?: string;               // Base64 model prediction
    groundTruth?: string;        // Base64 manual annotation
  };
  segmentation?: {
    model: string;
    version: string;
    hasPrediction: boolean;
    hasGroundTruth: boolean;
    classDistribution: object;
  };
}
```

### Classification Database

**Database:** `classification_dataset_db`
**Store:** `classification_entries`

**Entry Structure:**
```typescript
interface ClassificationEntry {
  id: string;
  imageData: string;             // Base64 image
  label: 'golf' | 'not_golf';
  timestamp: string;
  location: {
    lat: number;
    lng: number;
  };
}
```

---

## Model Details

### Architecture
- **Base**: U-Net with ResNet50 encoder
- **Convolutions**: MobileNetV2-style depthwise separable convolutions
- **Input**: 832 x 512 x 3 (RGB)
- **Output**: 832 x 512 x 6 (class probabilities)

### Training Data
- **Dataset**: Danish Golf Courses Orthophotos (Kaggle)
- **Classes**: 6 (Background, Fairway, Green, Tee, Bunker, Water)
- **Annotations**: Pixel-level semantic masks

### Deployment
- **Platform**: Hugging Face Spaces
- **Framework**: FastAPI + TensorFlow
- **URL**: `https://elo0oo0-golf-segmentation-api.hf.space`

---

## Scripts

### visualize_dataset_examples.py
Generates visualization figures for papers and documentation.

**Figures Generated:**
1. `fig0_danish_golf_segmentation.png` - Danish golf courses with segmentation masks (3 random examples)
2. `fig1_danish_golf_courses.png` - Danish golf course examples
3. `fig2_ucmerced_golf_courses.png` - UC Merced golf examples
4. `fig3_challenging_negatives.png` - Challenging negative examples
5. `fig4_easy_negatives.png` - Easy negative examples
6. `fig5_golf_vs_challenging_comparison.png` - Side-by-side comparison
7. `fig6_ucmerced_all_classes.png` - All 21 UC Merced classes
8. `fig7_augmentation_examples.png` - Data augmentation demo
9. `fig8_segmentation_augmentation.png` - Synchronized image+mask augmentation

**Usage:**
```bash
python scripts/visualize_dataset_examples.py
```

**Output:** `paper_figures/` directory

---

## Security

### Password Protection
- Gate component wraps entire application
- Password stored in component (configurable)
- Session persistence via localStorage
- Logout functionality in footer

### Contact
Access requests: oscar.arenas01@student.csulb.edu

---

## References

### Datasets
1. **Danish Golf Courses Orthophotos**
   - URL: https://www.kaggle.com/datasets/jacotaco/danish-golf-courses-orthophotos
   - High-resolution aerial imagery with semantic masks
   - Used for segmentation model training

2. **UC Merced Land Use Dataset**
   - URL: https://huggingface.co/datasets/blanchon/UC_Merced
   - 21 land use classes including golf courses
   - Used for classification model training

### Code & Notebooks
3. **Semantic Segmentation of Danish Golf Courses (U-Net)**
   - URL: https://www.kaggle.com/code/viniciussmatthiesen/semantic-segmentation-of-danish-golf-courses-u-net
   - Model architecture and training reference

### Papers
4. **MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications**
   - URL: https://arxiv.org/abs/1704.04861
   - Depthwise separable convolution architecture

5. **IEEE Semantic Segmentation Methods**
   - URL: https://ieeexplore.ieee.org/document/10221980
   - Deep learning approaches for semantic segmentation

---

## Environment Variables

```env
NEXT_PUBLIC_GOOGLE_MAPS_API_KEY=your_google_maps_api_key
NEXT_PUBLIC_SEGMENTATION_API_URL=https://elo0oo0-golf-segmentation-api.hf.space
```

---

## Development

### Prerequisites
- Node.js 18+
- npm or yarn
- Google Maps API key

### Installation
```bash
npm install
```

### Development Server
```bash
npm run dev
```

### Production Build
```bash
npm run build
npm start
```

### API Development (Python)
```bash
cd api
pip install -r requirements.txt
python main.py
```

---

## License

This project is for educational and research purposes.

---

*Documentation last updated: November 2024*
