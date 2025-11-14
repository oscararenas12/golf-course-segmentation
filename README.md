# Golf Course Segmentation

Deep learning project for semantic segmentation of golf course features from aerial imagery.

## Overview

This project uses a U-Net architecture to identify and segment different features of golf courses from orthophotos (aerial images):
- **Fairway** - Main playing areas
- **Green** - Putting greens
- **Tee** - Tee boxes
- **Bunker** - Sand traps
- **Water** - Water hazards
- **Background** - Everything else

## Dataset

**Danish Golf Courses Orthophotos**
- Source: [Kaggle Dataset](https://www.kaggle.com/datasets/jacotaco/danish-golf-courses-orthophotos)
- 1,123 orthophotos from 107 Danish golf courses
- Resolution: 1600x900 pixels (resized to 256x256 for training)
- Scale: 1:1000
- Annotations: Semantic segmentation masks with 6 classes

## Model

**Architecture**: U-Net (PyTorch Lightning)
- Encoder-decoder with skip connections
- Input: RGB images (256x256)
- Output: 6-class segmentation mask
- Loss: CrossEntropyLoss
- Optimizer: AdamW (lr=1e-4)

**Training:**
- 50 epochs (~30-60 min on T4 x2 GPU)
- Batch size: 16
- Train/Val/Test split: 70/20/10

## Future Goals

- **Web Application**: Interactive map where users can zoom into golf courses and see automatic segmentation
- **Global Coverage**: Expand beyond Danish courses to worldwide golf courses using satellite imagery APIs
- **Real-time Inference**: Segment courses on-demand when users zoom to certain map levels

## Setup

```bash
# Clone repository
git clone <repo-url>
cd golf-course-segmentation

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset (requires kagglehub or manual download)
python download_dataset.py
```

## Project Structure

```
golf-course-segmentation/
├── data/               # Dataset directory (not tracked)
├── models/             # Saved model weights
├── notebooks/          # Jupyter notebooks
├── src/                # Source code
│   ├── model.py       # U-Net model definition
│   ├── dataset.py     # Dataset and DataModule classes
│   └── train.py       # Training script
├── requirements.txt
└── README.md
```

## License

Dataset: Database: Open Database, Contents: © Original Authors (Open Data Commons)

## Credits

Based on Kaggle notebook by [viniciussmatthiesen](https://www.kaggle.com/code/viniciussmatthiesen/semantic-segmentation-of-danish-golf-courses-u-net)