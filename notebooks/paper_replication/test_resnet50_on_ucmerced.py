# %%
# Test ResNet50 U-Net Segmentation Model on UC Merced Golf Courses
# Evaluates how well the larger, transfer-learning model generalizes to different golf courses

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from datasets import load_dataset

# %%
# Configuration
MODEL_PATH = '../../models/best_unet_resnet50_large.keras'  # or 'final_unet_resnet50_large.keras'
IMAGE_SIZE = (512, 832)  # Height×Width - matches ResNet50 training
OUTPUT_DIR = '../../predictions/resnet50_ucmerced'
NUM_CLASSES = 6  # Background, Fairway, Green, Tee, Bunker, Water

# Class colors for visualization
CLASS_COLORS = np.array([
    [0, 0, 0],                      # Background - Black
    [0, 140, 0],                    # Fairway - Dark green
    [0, 255, 0],                    # Green - Bright green
    [255, 0, 0],                    # Tee - Red
    [217, 230, 122],                # Bunker - Sandy yellow
    [7, 15, 247]                    # Water - Blue
], dtype=np.float32) / 255.0  # Normalize to [0, 1]

CLASS_NAMES = ['Background', 'Fairway', 'Green', 'Tee', 'Bunker', 'Water']

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def mask_to_rgb(mask):
    """Convert single-channel class mask to RGB visualization."""
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.float32)

    for class_id in range(NUM_CLASSES):
        rgb_mask[mask == class_id] = CLASS_COLORS[class_id]

    return rgb_mask

# %%
# Load trained ResNet50 U-Net model
print("=" * 60)
print("LOADING RESNET50 U-NET MODEL")
print("=" * 60)

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found: {MODEL_PATH}")
    print("Please train the model first using notebooks/paper_replication/unet_resnet50_large.py")
    exit(1)

print(f"\nLoading model from: {MODEL_PATH}")
print(f"Expected image size: {IMAGE_SIZE[1]}×{IMAGE_SIZE[0]} (W×H)")

# ResNet50 U-Net doesn't use custom layers, so no custom_objects needed
model = keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# %%
# Load UC Merced golf courses
print("\n" + "=" * 60)
print("LOADING UC MERCED GOLF COURSES")
print("=" * 60)

print("\nDownloading UC Merced dataset...")
ucmerced = load_dataset("blanchon/UC_Merced", split="train")

# Extract only golf course images (class 9)
golf_samples = [item for item in ucmerced if item['label'] == 9]
print(f"✅ Found {len(golf_samples)} golf course images (class 9)")

# %%
# Prepare images for inference
print("\n" + "=" * 60)
print("PREPARING IMAGES")
print("=" * 60)

def prepare_image(pil_image, target_size=(512, 832)):
    """Resize and normalize image for ResNet50 U-Net inference."""
    img = pil_image.resize((target_size[1], target_size[0]))  # PIL uses (W, H)
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    return img_array.astype(np.float32)

print(f"\nPreparing {len(golf_samples)} images to {IMAGE_SIZE[1]}×{IMAGE_SIZE[0]}...")
images = []
for sample in golf_samples:
    img = prepare_image(sample['image'], IMAGE_SIZE)
    images.append(img)

images = np.array(images)
print(f"✅ Images prepared: {images.shape}")
print(f"   Note: Larger image size captures more detail than original U-Net (256×256)")

# %%
# Run inference
print("\n" + "=" * 60)
print("RUNNING SEGMENTATION")
print("=" * 60)

print("\nRunning ResNet50 U-Net inference on all 100 images...")
print("⚠️  This may take longer due to larger image size and deeper model")
predictions = model.predict(images, batch_size=4, verbose=1)  # Smaller batch for large images

# Convert predictions to class masks (argmax over channel dimension)
pred_masks = np.argmax(predictions, axis=-1)  # Shape: (100, 512, 832)

print(f"✅ Predictions complete!")
print(f"   Logits shape: {predictions.shape}")
print(f"   Class masks shape: {pred_masks.shape}")
print(f"   Predicted classes: {np.unique(pred_masks)}")

# %%
# Calculate statistics
print("\n" + "=" * 60)
print("PREDICTION STATISTICS")
print("=" * 60)

# Calculate percentage of pixels predicted as golf course features (non-background)
golf_percentages = (pred_masks > 0).mean(axis=(1, 2)) * 100  # % of non-background pixels

print(f"\nGolf course feature coverage (non-background):")
print(f"  Mean:   {golf_percentages.mean():.1f}% of pixels")
print(f"  Median: {np.median(golf_percentages):.1f}% of pixels")
print(f"  Min:    {golf_percentages.min():.1f}% of pixels")
print(f"  Max:    {golf_percentages.max():.1f}% of pixels")

# Class distribution
print(f"\nClass distribution across all images:")
for class_id in range(NUM_CLASSES):
    class_pixels = (pred_masks == class_id).sum()
    total_pixels = pred_masks.size
    percentage = (class_pixels / total_pixels) * 100
    print(f"  {CLASS_NAMES[class_id]:12s}: {percentage:5.2f}% of all pixels")

# %%
# Visualize sample predictions
print("\n" + "=" * 60)
print("VISUALIZING RESULTS")
print("=" * 60)

# Select diverse samples (spread across the dataset)
num_samples = 20
sample_indices = np.linspace(0, len(images) - 1, num_samples, dtype=int)

fig, axes = plt.subplots(num_samples, 3, figsize=(18, num_samples * 4))

for i, idx in enumerate(sample_indices):
    img = images[idx]
    pred_mask = pred_masks[idx]
    pred_rgb = mask_to_rgb(pred_mask)

    # Original image
    axes[i, 0].imshow(img)
    axes[i, 0].set_title(f"Original #{idx}", fontsize=10)
    axes[i, 0].axis('off')

    # Predicted mask
    axes[i, 1].imshow(pred_rgb)
    axes[i, 1].set_title(f"Prediction ({golf_percentages[idx]:.1f}%)", fontsize=10)
    axes[i, 1].axis('off')

    # Overlay
    axes[i, 2].imshow(img)
    axes[i, 2].imshow(pred_rgb, alpha=0.5)
    axes[i, 2].set_title("Overlay", fontsize=10)
    axes[i, 2].axis('off')

plt.suptitle('ResNet50 U-Net Segmentation on UC Merced Golf Courses\n'
             f'(Image size: {IMAGE_SIZE[1]}×{IMAGE_SIZE[0]} - Transfer Learning from ImageNet)\n'
             '(Green=Fairway/Green, Yellow=Bunker, Blue=Water, Red=Tee)',
             fontsize=14, weight='bold', y=0.998)
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, 'ucmerced_resnet50_samples.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Sample visualization saved: {output_path}")
plt.show()

# %%
# Create grid visualization of all predictions
print("\nCreating grid visualization of all 100 predictions...")

# Create 10x10 grid
fig, axes = plt.subplots(10, 10, figsize=(25, 15))
axes = axes.flatten()

for i in range(100):
    # Show overlay
    pred_rgb = mask_to_rgb(pred_masks[i])
    axes[i].imshow(images[i])
    axes[i].imshow(pred_rgb, alpha=0.5)
    axes[i].set_title(f"#{i}\n{golf_percentages[i]:.0f}%", fontsize=6)
    axes[i].axis('off')

plt.suptitle('All 100 UC Merced Golf Courses - ResNet50 U-Net Predictions',
             fontsize=16, weight='bold')
plt.tight_layout()

grid_path = os.path.join(OUTPUT_DIR, 'ucmerced_resnet50_grid.png')
plt.savefig(grid_path, dpi=150, bbox_inches='tight')
print(f"✅ Grid visualization saved: {grid_path}")
plt.show()

# %%
# Find interesting cases
print("\n" + "=" * 60)
print("INTERESTING CASES")
print("=" * 60)

# High confidence predictions (>70% golf features)
high_conf_indices = np.where(golf_percentages > 70)[0]
print(f"\nHigh confidence ({len(high_conf_indices)} images):")
print(f"  Images with >70% golf features: {list(high_conf_indices[:5])}...")

# Low confidence predictions (<20% golf features)
low_conf_indices = np.where(golf_percentages < 20)[0]
print(f"\nLow confidence ({len(low_conf_indices)} images):")
print(f"  Images with <20% golf features: {list(low_conf_indices[:5])}...")

# Medium confidence (40-60% - most typical)
medium_conf_indices = np.where((golf_percentages >= 40) & (golf_percentages <= 60))[0]
print(f"\nMedium confidence ({len(medium_conf_indices)} images):")
print(f"  Images with 40-60% golf features: {list(medium_conf_indices[:5])}...")

# %%
# Visualize extreme cases
fig, axes = plt.subplots(2, 6, figsize=(24, 8))

# Top 3 highest confidence
for i in range(3):
    idx = np.argsort(golf_percentages)[-i-1]
    pred_rgb = mask_to_rgb(pred_masks[idx])

    axes[0, i*2].imshow(images[idx])
    axes[0, i*2].set_title(f"High #{idx}\n{golf_percentages[idx]:.1f}%", fontsize=10)
    axes[0, i*2].axis('off')

    axes[0, i*2+1].imshow(images[idx])
    axes[0, i*2+1].imshow(pred_rgb, alpha=0.5)
    axes[0, i*2+1].set_title("Overlay", fontsize=10)
    axes[0, i*2+1].axis('off')

# Top 3 lowest confidence
for i in range(3):
    idx = np.argsort(golf_percentages)[i]
    pred_rgb = mask_to_rgb(pred_masks[idx])

    axes[1, i*2].imshow(images[idx])
    axes[1, i*2].set_title(f"Low #{idx}\n{golf_percentages[idx]:.1f}%", fontsize=10)
    axes[1, i*2].axis('off')

    axes[1, i*2+1].imshow(images[idx])
    axes[1, i*2+1].imshow(pred_rgb, alpha=0.5)
    axes[1, i*2+1].set_title("Overlay", fontsize=10)
    axes[1, i*2+1].axis('off')

plt.suptitle('Extreme Cases: Highest vs Lowest Confidence Predictions',
             fontsize=14, weight='bold')
plt.tight_layout()

extreme_path = os.path.join(OUTPUT_DIR, 'ucmerced_resnet50_extremes.png')
plt.savefig(extreme_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Extreme cases saved: {extreme_path}")
plt.show()

# %%
# Summary
print("\n" + "=" * 60)
print("SUMMARY - RESNET50 U-NET EVALUATION")
print("=" * 60)

print(f"\n✅ Tested ResNet50 U-Net on {len(golf_samples)} UC Merced golf courses")
print(f"   Model: ResNet50 encoder with ImageNet pre-training")
print(f"   Image size: {IMAGE_SIZE[1]}×{IMAGE_SIZE[0]} (vs 256×256 for original U-Net)")
print(f"   Average coverage: {golf_percentages.mean():.1f}% golf features detected")
print(f"\n📁 Results saved to: {OUTPUT_DIR}/")
print(f"   - ucmerced_resnet50_samples.png (20 detailed samples)")
print(f"   - ucmerced_resnet50_grid.png (10×10 grid of all 100)")
print(f"   - ucmerced_resnet50_extremes.png (highest/lowest confidence)")

print("\n" + "=" * 60)
print("COMPARISON WITH ORIGINAL U-NET")
print("=" * 60)
print("\nExpected improvements with ResNet50:")
print("  ✓ Better feature extraction (ImageNet pre-training)")
print("  ✓ Higher resolution (832×512 vs 256×256)")
print("  ✓ More detailed detection of small features (tees, bunkers)")
print("\nTo compare:")
print("  1. Run the original test: test_unet_on_ucmerced.py")
print("  2. Compare class distributions and confidence percentages")
print("  3. Look for improvements in small feature detection")

print("\n" + "=" * 60)
print("NEXT STEPS")
print("=" * 60)
print("\n1. Review the visualizations to assess model generalization")
print("2. Compare with original U-Net results (ucmerced_predictions/)")
print("3. Check if larger resolution helps with tee and bunker detection")
print("4. Evaluate if ImageNet pre-training improves robustness")
print("5. Consider calculating per-class IoU metrics on test set")

print("\n✅ Evaluation complete!")
print("=" * 60)

# %%
