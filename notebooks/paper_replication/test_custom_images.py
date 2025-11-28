# %%
# Test ResNet50 U-Net on Custom Golf Course Images
# Works with any size images from Google Maps, Google Earth, or other sources

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# %%
# Configuration
MODEL_PATH = '../../models/best_unet_resnet50_large.keras'
IMAGE_SIZE = (512, 832)  # Model's expected size (H, W)
CUSTOM_IMAGES_DIR = '../../data/custom_golf_images'  # Put your images here
OUTPUT_DIR = '../../predictions/custom_images'
NUM_CLASSES = 6

# Resize method: 'crop', 'pad', or 'stretch'
# - 'crop': Center crop to 1.625:1 ratio (RECOMMENDED - no distortion)
# - 'pad': Add black borders to match ratio (preserves full image)
# - 'stretch': Force resize (may distort if aspect ratio differs)
RESIZE_METHOD = 'crop'

# Class colors for visualization
CLASS_COLORS = np.array([
    [0, 0, 0],                      # Background - Black
    [0, 140, 0],                    # Fairway - Dark green
    [0, 255, 0],                    # Green - Bright green
    [255, 0, 0],                    # Tee - Red
    [217, 230, 122],                # Bunker - Sandy yellow
    [7, 15, 247]                    # Water - Blue
], dtype=np.float32) / 255.0

CLASS_NAMES = ['Background', 'Fairway', 'Green', 'Tee', 'Bunker', 'Water']

# Create directories
os.makedirs(CUSTOM_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("TEST RESNET50 U-NET ON CUSTOM IMAGES")
print("=" * 60)
print(f"\n📁 Place your golf course images in: {CUSTOM_IMAGES_DIR}")
print(f"   Supported formats: .jpg, .jpeg, .png")
print(f"   Any resolution works - will be resized to {IMAGE_SIZE[1]}×{IMAGE_SIZE[0]}")

# %%
# Load model
print("\n" + "=" * 60)
print("LOADING MODEL")
print("=" * 60)

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found: {MODEL_PATH}")
    print("Train the model first using unet_resnet50_large.py")
    exit(1)

print(f"\nLoading ResNet50 U-Net from: {MODEL_PATH}")
model = keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# %%
# Find and load custom images
print("\n" + "=" * 60)
print("LOADING CUSTOM IMAGES")
print("=" * 60)

# Get all image files
image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
image_files = [f for f in os.listdir(CUSTOM_IMAGES_DIR) if f.endswith(image_extensions)]

if len(image_files) == 0:
    print(f"\n❌ No images found in {CUSTOM_IMAGES_DIR}")
    print("\n📋 Instructions:")
    print("   1. Open Google Maps in satellite view")
    print("   2. Find a golf course (try: Pebble Beach, St Andrews, Augusta National)")
    print("   3. Take screenshot of the course")
    print(f"   4. Save to: {CUSTOM_IMAGES_DIR}")
    print("   5. Run this script again")
    exit(1)

print(f"\n✅ Found {len(image_files)} images:")
for i, filename in enumerate(image_files, 1):
    print(f"   {i}. {filename}")

# %%
# Prepare images for inference
print("\n" + "=" * 60)
print("PREPARING IMAGES")
print("=" * 60)

def prepare_image(image_path, target_size=(512, 832), method='crop'):
    """
    Load and prepare image for inference with aspect ratio preservation.

    Args:
        image_path: Path to image file
        target_size: (H, W) target size for model
        method: 'crop', 'pad', or 'stretch'
            - 'crop': Center crop to match aspect ratio (recommended)
            - 'pad': Add padding to match aspect ratio
            - 'stretch': Stretch/squash to fit (may distort)

    Returns:
        original image, prepared array, original size
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    original_size = img.size  # (W, H)

    print(f"   Original size: {original_size[0]}×{original_size[1]}")

    # Target aspect ratio
    target_h, target_w = target_size
    target_ratio = target_w / target_h  # 832/512 = 1.625

    # Current aspect ratio
    current_w, current_h = original_size
    current_ratio = current_w / current_h

    print(f"   Aspect ratio: {current_ratio:.3f} (target: {target_ratio:.3f})")

    if method == 'stretch':
        # Simple resize (may distort)
        img_resized = img.resize((target_w, target_h))
        if abs(current_ratio - target_ratio) > 0.1:
            print(f"   ⚠️  Warning: Image will be distorted (aspect ratio mismatch)")

    elif method == 'crop':
        # Center crop to match aspect ratio, then resize
        if current_ratio > target_ratio:
            # Image is wider - crop width
            new_width = int(current_h * target_ratio)
            left = (current_w - new_width) // 2
            img_cropped = img.crop((left, 0, left + new_width, current_h))
            print(f"   ✂️  Cropped to: {new_width}×{current_h}")
        else:
            # Image is taller - crop height
            new_height = int(current_w / target_ratio)
            top = (current_h - new_height) // 2
            img_cropped = img.crop((0, top, current_w, top + new_height))
            print(f"   ✂️  Cropped to: {current_w}×{new_height}")

        img_resized = img_cropped.resize((target_w, target_h))
        print(f"   ✅ Resized to: {target_w}×{target_h} (no distortion)")

    elif method == 'pad':
        # Pad to match aspect ratio, then resize
        if current_ratio > target_ratio:
            # Image is wider - pad height
            new_height = int(current_w / target_ratio)
            padding = (new_height - current_h) // 2
            img_padded = Image.new('RGB', (current_w, new_height), (0, 0, 0))
            img_padded.paste(img, (0, padding))
            print(f"   📏 Padded to: {current_w}×{new_height}")
        else:
            # Image is taller - pad width
            new_width = int(current_h * target_ratio)
            padding = (new_width - current_w) // 2
            img_padded = Image.new('RGB', (new_width, current_h), (0, 0, 0))
            img_padded.paste(img, (padding, 0))
            print(f"   📏 Padded to: {new_width}×{current_h}")

        img_resized = img_padded.resize((target_w, target_h))
        print(f"   ✅ Resized to: {target_w}×{target_h} (with padding)")

    else:
        raise ValueError(f"Unknown method: {method}")

    # Convert to array and normalize
    img_array = np.array(img_resized) / 255.0
    return img, img_array.astype(np.float32), original_size

images = []
originals = []
original_sizes = []
filenames = []

for filename in image_files:
    print(f"\nProcessing: {filename}")
    image_path = os.path.join(CUSTOM_IMAGES_DIR, filename)
    original, prepared, orig_size = prepare_image(image_path, IMAGE_SIZE, method=RESIZE_METHOD)

    originals.append(original)
    images.append(prepared)
    original_sizes.append(orig_size)
    filenames.append(filename)

images_array = np.array(images)
print(f"\n✅ Prepared {len(images)} images for inference")
print(f"   Input shape: {images_array.shape}")

# %%
# Run inference
print("\n" + "=" * 60)
print("RUNNING SEGMENTATION")
print("=" * 60)

print("\nRunning ResNet50 U-Net inference...")
predictions = model.predict(images_array, batch_size=1, verbose=1)

# Convert to class predictions
pred_masks = np.argmax(predictions, axis=-1)

print(f"✅ Inference complete!")
print(f"   Predictions shape: {pred_masks.shape}")

# %%
# Calculate statistics per image
print("\n" + "=" * 60)
print("PREDICTION STATISTICS")
print("=" * 60)

def mask_to_rgb(mask):
    """Convert class mask to RGB."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for class_id in range(NUM_CLASSES):
        rgb[mask == class_id] = CLASS_COLORS[class_id]
    return rgb

for i, filename in enumerate(filenames):
    pred_mask = pred_masks[i]

    print(f"\n{filename}:")

    # Calculate percentage of each class
    for class_id in range(NUM_CLASSES):
        pixels = (pred_mask == class_id).sum()
        percentage = (pixels / pred_mask.size) * 100

        if percentage > 1.0:  # Only show classes with >1% coverage
            print(f"   {CLASS_NAMES[class_id]:<12}: {percentage:5.1f}%")

    # Golf feature coverage
    golf_pixels = (pred_mask > 0).sum()
    golf_percentage = (golf_pixels / pred_mask.size) * 100
    print(f"   {'Total Golf':<12}: {golf_percentage:5.1f}%")

# %%
# Visualize all predictions
print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

num_images = len(images)

# Create figure with all images
fig, axes = plt.subplots(num_images, 3, figsize=(18, num_images * 5))

if num_images == 1:
    axes = axes.reshape(1, -1)

for i in range(num_images):
    pred_mask = pred_masks[i]
    pred_rgb = mask_to_rgb(pred_mask)
    golf_pct = (pred_mask > 0).mean() * 100

    # Original image
    axes[i, 0].imshow(images[i])
    axes[i, 0].set_title(f"{filenames[i]}\n(Original: {original_sizes[i][0]}×{original_sizes[i][1]})", fontsize=10)
    axes[i, 0].axis('off')

    # Prediction mask
    axes[i, 1].imshow(pred_rgb)
    axes[i, 1].set_title(f"Segmentation\n({golf_pct:.1f}% golf features)", fontsize=10)
    axes[i, 1].axis('off')

    # Overlay
    axes[i, 2].imshow(images[i])
    axes[i, 2].imshow(pred_rgb, alpha=0.5)
    axes[i, 2].set_title("Overlay", fontsize=10)
    axes[i, 2].axis('off')

plt.suptitle('ResNet50 U-Net Predictions on Custom Golf Course Images\n'
             'Green=Fairway, Bright Green=Green, Yellow=Bunker, Blue=Water, Red=Tee',
             fontsize=14, weight='bold', y=0.998)
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, 'custom_predictions.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Visualization saved: {output_path}")
plt.show()

# %%
# Save individual predictions
print("\nSaving individual predictions...")

for i, filename in enumerate(filenames):
    pred_rgb = mask_to_rgb(pred_masks[i])

    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(images[i])
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(pred_rgb)
    axes[1].set_title("Segmentation", fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(images[i])
    axes[2].imshow(pred_rgb, alpha=0.5)
    axes[2].set_title("Overlay", fontsize=12)
    axes[2].axis('off')

    # Add class distribution as text
    class_text = "Detected:\n"
    for class_id in range(1, NUM_CLASSES):  # Skip background
        pixels = (pred_masks[i] == class_id).sum()
        percentage = (pixels / pred_masks[i].size) * 100
        if percentage > 1.0:
            class_text += f"{CLASS_NAMES[class_id]}: {percentage:.1f}%\n"

    fig.text(0.02, 0.5, class_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'{filename}', fontsize=14, weight='bold')
    plt.tight_layout()

    # Save
    output_filename = os.path.splitext(filename)[0] + '_prediction.png'
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   Saved: {output_filename}")

# %%
# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

avg_golf_coverage = [(pred_masks[i] > 0).mean() * 100 for i in range(num_images)]
avg_golf = np.mean(avg_golf_coverage)

print(f"\n✅ Processed {num_images} custom golf course images")
print(f"   Average golf feature coverage: {avg_golf:.1f}%")
print(f"\n📁 Results saved to: {OUTPUT_DIR}/")
print(f"   - custom_predictions.png (all images)")
print(f"   - [filename]_prediction.png (individual results)")

print("\n" + "=" * 60)
print("INTERPRETATION GUIDE")
print("=" * 60)
print("\nWhat to look for:")
print("  ✅ Good predictions:")
print("     - Fairways (dark green) cover most playable areas")
print("     - Greens (bright green) detected on putting surfaces")
print("     - Bunkers (yellow) on sand traps")
print("     - Water (blue) on ponds/lakes")
print("\n  ⚠️  Signs of poor generalization:")
print("     - Mostly background (black) detected")
print("     - Random scattered predictions")
print("     - Wrong features (e.g., trees marked as fairway)")
print("\n  💡 Different courses will vary:")
print("     - Desert courses: May have less fairway coverage")
print("     - Links courses: Natural terrain, harder to segment")
print("     - Parkland courses: Should perform best (similar to Danish)")

print("\n" + "=" * 60)
print("NEXT STEPS")
print("=" * 60)
print("\n1. Try different golf courses:")
print("   - Pebble Beach (coastal)")
print("   - TPC Scottsdale (desert)")
print("   - St Andrews (links)")
print("   - Augusta National (parkland)")
print("\n2. Compare results with UC Merced and Danish test set")
print("\n3. If results are poor across all custom images:")
print("   - Model needs more training epochs")
print("   - Need stronger data augmentation")
print("   - Consider training on more diverse datasets")

print("\n✅ Testing complete!")
print("=" * 60)

# %%
