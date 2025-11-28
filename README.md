# Golf Course Segmentation

Deep learning project for semantic segmentation of golf course features from aerial imagery, with an interactive Next.js web application.

## Overview

This project combines machine learning and web development to create a comprehensive golf course analysis system:

**Machine Learning:**
- U-Net architecture with ResNet50 encoder (TensorFlow/Keras)
- Trained on 1,123 Danish golf course orthophotos
- 6-class segmentation: Fairway, Green, Tee, Bunker, Water, Background
- Resolution: 832×512 pixels (matching Aalborg University paper)

**Web Application:**
- Next.js 15 (App Router) + TypeScript
- Google Maps API integration for satellite imagery
- Two workflows: Dataset Creation & Segmentation Analysis
- Real-time AI-powered golf course feature detection

## Project Structure

```
golf-course-segmentation/
├── app/                      # Next.js App Router
│   ├── layout.tsx
│   ├── page.tsx
│   └── globals.css
├── components/               # React components
│   ├── Header.tsx
│   ├── Sidebar.tsx
│   ├── MapArea.tsx
│   ├── WorkflowSelector.tsx
│   ├── DatasetCapture.tsx
│   ├── Legend.tsx
│   └── ui/                   # ShadCN UI components
├── types/                    # TypeScript types
├── utils/                    # Utility functions
├── public/                   # Static assets
├── notebooks/                # ML training scripts
│   ├── unet_golf_segmentation_tf.py
│   └── paper_replication/
│       ├── unet_resnet50_large.py
│       ├── evaluate_resnet50_testset.py
│       ├── test_resnet50_on_ucmerced.py
│       └── test_custom_images.py
├── models/                   # Trained .keras models
├── predictions/              # Model outputs
├── classifier/               # Binary golf/non-golf classifier
└── api/                      # FastAPI backend (coming soon)
```

## Getting Started

### Prerequisites

**For Web App:**
- Node.js 18+ and npm/yarn/pnpm
- Google Maps API key (optional but recommended)

**For ML Training:**
- Python 3.10+
- TensorFlow 2.15+
- GPU recommended (CUDA support)

### Web Application Setup

1. **Install dependencies:**
```bash
npm install
```

2. **Set up environment variables:**
```bash
cp .env.local.example .env.local
```

Edit `.env.local` and add your Google Maps API key:
```
NEXT_PUBLIC_GOOGLE_MAPS_API_KEY=your_api_key_here
```

3. **Run development server:**
```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

4. **Build for production:**
```bash
npm run build
npm start
```

### ML Model Training

1. **Install Python dependencies:**
```bash
# Using pip
pip install tensorflow kagglehub pillow opencv-python numpy pandas matplotlib

# Or using uv (recommended)
uv sync
```

2. **Train ResNet50 U-Net:**
```bash
cd notebooks/paper_replication
python unet_resnet50_large.py
```

3. **Evaluate on test set:**
```bash
python evaluate_resnet50_testset.py
```

## Features

### Web Application

**Dataset Creation Mode:**
- Capture satellite imagery from Google Maps
- Build custom training datasets
- Add metadata (course name, location)
- Export as ZIP for model training

**Segmentation Analysis Mode:**
- Real-time golf course feature detection
- Interactive overlay with adjustable opacity
- Statistics and class distribution
- Color-coded legend

### ML Model

**Architecture:**
- Encoder: ResNet50 (ImageNet pre-trained)
- Decoder: Upsampling blocks with skip connections
- Input: 832×512 RGB images
- Output: 6-class segmentation mask

**Performance:**
- Target: 42% mean IoU (ResNet50 baseline from paper)
- Goal: Match paper's 69.6% mIoU with advanced encoders

## Technology Stack

**Frontend:**
- Next.js 15 (App Router)
- TypeScript
- Tailwind CSS v4
- ShadCN/UI components
- Google Maps JavaScript API
- Lucide React icons
- Recharts

**Backend (Coming Soon):**
- FastAPI
- TensorFlow Serving
- Python 3.10+

**ML Framework:**
- TensorFlow/Keras 2.15+
- Mixed precision training (float16)
- GPU acceleration (CUDA)

## Dataset

**Danish Golf Courses Orthophotos**
- Source: [Kaggle Dataset](https://www.kaggle.com/datasets/jacotaco/danish-golf-courses-orthophotos)
- 1,123 orthophotos from 107 Danish golf courses
- Scale: 1:1000
- 6-class semantic segmentation masks

## Usage

### Creating a Dataset

1. Select "Create a Dataset" from landing page
2. Search or pan to a golf course
3. Click "Capture Current View"
4. Add metadata (name, location, notes)
5. Click "Save to Dataset"
6. Export as ZIP when ready

### Running Segmentation

1. Select "Run Segmentation" from landing page
2. Position golf course within capture box
3. Click "Segment Course"
4. View overlay and statistics
5. Adjust opacity/settings as needed

## UI Design

The web application UI was designed with Figma Make AI and converted to Next.js code.
Original Figma design: https://www.figma.com/design/jKEfDiUpROQtVHkJj2bcgx/Golf-Course-Segmentation-UI

## Development Roadmap

- [x] Train ResNet50 U-Net model
- [x] Build Next.js frontend with dual workflows
- [x] Integrate Google Maps API
- [x] Implement dataset capture & export
- [ ] Create FastAPI backend for model serving
- [ ] Connect frontend to backend API
- [ ] Deploy to production (Vercel)
- [ ] Expand to global golf courses
- [ ] Improve model performance (target 69.6% mIoU)

## Credits

**Based on research by:**
- Aalborg University: "Semantic Segmentation of Golf Courses for Course Rating Assistance" (IEEE ICMEW 2023)

**Dataset:**
- Kaggle: [Danish Golf Courses Orthophotos](https://www.kaggle.com/datasets/jacotaco/danish-golf-courses-orthophotos)

**UI Design:**
- Generated with Figma Make AI

## License

MIT

## Contributing

Contributions welcome! Please open an issue or submit a PR.
