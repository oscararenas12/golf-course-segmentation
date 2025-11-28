# %%
# Evaluate ResNet50 U-Net on Danish Golf Course Test Set
# Calculate per-class IoU metrics to match paper's evaluation

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import kagglehub

# %%
# Configuration
MODEL_PATH = '../../models/best_unet_resnet50_large.keras'
IMAGE_SIZE = (512, 832)
BATCH_SIZE = 2
NUM_CLASSES = 6

CLASS_NAMES = ['Background', 'Fairway', 'Green', 'Tee', 'Bunker', 'Water']
CLASS_COLORS = np.array([
    [0, 0, 0],                      # Background - Black
    [0, 140, 0],                    # Fairway - Dark green
    [0, 255, 0],                    # Green - Bright green
    [255, 0, 0],                    # Tee - Red
    [217, 230, 122],                # Bunker - Sandy yellow
    [7, 15, 247]                    # Water - Blue
], dtype=np.float32) / 255.0

OUTPUT_DIR = '../../predictions/resnet50_testset'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("EVALUATING RESNET50 U-NET ON TEST SET")
print("=" * 60)

# %%
# Load dataset
print("\nDownloading Danish Golf Course dataset...")
dataset_path = kagglehub.dataset_download('jacotaco/danish-golf-courses-orthophotos')

IMAGES_DIR = os.path.join(dataset_path, '1. orthophotos')
LABELMASKS_DIR = os.path.join(dataset_path, '3. class masks')

print(f"Dataset path: {dataset_path}")

# %%
# Data loading functions

def load_and_preprocess_image(image_path, mask_path):
    """Load and preprocess a single image and mask pair."""
    # Load image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    # Load mask
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, IMAGE_SIZE, method='nearest')
    mask = tf.cast(mask, tf.float32)
    mask = tf.squeeze(mask, axis=-1)

    return image, mask


def create_test_dataset():
    """Create test dataset (10% split, same as training)."""
    image_filenames = sorted(os.listdir(IMAGES_DIR))
    image_paths = [os.path.join(IMAGES_DIR, fname) for fname in image_filenames]
    mask_paths = [os.path.join(LABELMASKS_DIR, fname.replace('.jpg', '.png')) for fname in image_filenames]

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.shuffle(buffer_size=len(image_paths), seed=42)
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    total_size = len(image_paths)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)

    # Get test set (skip train + val)
    test_ds = dataset.skip(train_size + val_size)
    test_size = total_size - train_size - val_size

    print(f"\nDataset split:")
    print(f"  Total: {total_size} images")
    print(f"  Train: {train_size} (70%)")
    print(f"  Val:   {val_size} (20%)")
    print(f"  Test:  {test_size} (10%)")

    # Batch
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return test_ds, test_size


# %%
# Load model
print("\nLoading ResNet50 U-Net model...")
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found: {MODEL_PATH}")
    print("Train the model first using unet_resnet50_large.py")
    exit(1)

model = keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# %%
# Prepare test dataset
print("\nPreparing test dataset...")
test_ds, test_size = create_test_dataset()

# %%
# Calculate IoU metrics (Intersection over Union)

def calculate_iou_per_class(y_true, y_pred, num_classes=6):
    """
    Calculate IoU for each class.

    IoU = Intersection / Union
        = True Positives / (True Positives + False Positives + False Negatives)
    """
    ious = []

    for class_id in range(num_classes):
        # Get binary masks for this class
        true_mask = (y_true == class_id)
        pred_mask = (y_pred == class_id)

        # Calculate intersection and union
        intersection = np.logical_and(true_mask, pred_mask).sum()
        union = np.logical_or(true_mask, pred_mask).sum()

        # Calculate IoU (handle division by zero)
        if union == 0:
            iou = float('nan')  # No ground truth or predictions for this class
        else:
            iou = intersection / union

        ious.append(iou)

    return ious


def calculate_sensitivity(y_true, y_pred, num_classes=6):
    """
    Calculate Sensitivity (Recall) per class.

    Sensitivity = True Positives / (True Positives + False Negatives)
                = True Positives / All Ground Truth Pixels
    """
    sensitivities = []

    for class_id in range(num_classes):
        true_mask = (y_true == class_id)
        pred_mask = (y_pred == class_id)

        true_positives = np.logical_and(true_mask, pred_mask).sum()
        all_ground_truth = true_mask.sum()

        if all_ground_truth == 0:
            sensitivity = float('nan')
        else:
            sensitivity = true_positives / all_ground_truth

        sensitivities.append(sensitivity)

    return sensitivities


def calculate_ppv(y_true, y_pred, num_classes=6):
    """
    Calculate PPV (Positive Predictive Value / Precision) per class.

    PPV = True Positives / (True Positives + False Positives)
        = True Positives / All Predicted Pixels
    """
    ppvs = []

    for class_id in range(num_classes):
        true_mask = (y_true == class_id)
        pred_mask = (y_pred == class_id)

        true_positives = np.logical_and(true_mask, pred_mask).sum()
        all_predictions = pred_mask.sum()

        if all_predictions == 0:
            ppv = float('nan')
        else:
            ppv = true_positives / all_predictions

        ppvs.append(ppv)

    return ppvs


# %%
# Run inference on test set
print("\n" + "=" * 60)
print("RUNNING INFERENCE ON TEST SET")
print("=" * 60)

all_true_masks = []
all_pred_masks = []

for batch_idx, (images, masks) in enumerate(test_ds):
    print(f"Processing batch {batch_idx + 1}...", end='\r')

    # Predict
    predictions = model.predict(images, verbose=0)
    pred_masks = np.argmax(predictions, axis=-1)

    # Store for metric calculation
    all_true_masks.append(masks.numpy())
    all_pred_masks.append(pred_masks)

print(f"\n✅ Inference complete on {test_size} test images")

# Concatenate all batches
all_true_masks = np.concatenate(all_true_masks, axis=0)
all_pred_masks = np.concatenate(all_pred_masks, axis=0)

print(f"True masks shape: {all_true_masks.shape}")
print(f"Pred masks shape: {all_pred_masks.shape}")

# %%
# Calculate metrics
print("\n" + "=" * 60)
print("CALCULATING METRICS")
print("=" * 60)

# Flatten for metric calculation
y_true_flat = all_true_masks.flatten()
y_pred_flat = all_pred_masks.flatten()

# Calculate per-class metrics
ious = calculate_iou_per_class(y_true_flat, y_pred_flat, NUM_CLASSES)
sensitivities = calculate_sensitivity(y_true_flat, y_pred_flat, NUM_CLASSES)
ppvs = calculate_ppv(y_true_flat, y_pred_flat, NUM_CLASSES)

# Calculate mean metrics (excluding NaN values and background)
valid_ious = [iou for iou in ious[1:] if not np.isnan(iou)]  # Exclude background (class 0)
valid_sens = [s for s in sensitivities[1:] if not np.isnan(s)]
valid_ppvs = [p for p in ppvs[1:] if not np.isnan(p)]

mean_iou = np.mean(valid_ious) if valid_ious else 0.0
mean_sensitivity = np.mean(valid_sens) if valid_sens else 0.0
mean_ppv = np.mean(valid_ppvs) if valid_ppvs else 0.0

# %%
# Print results
print("\n" + "=" * 60)
print("RESULTS - PER-CLASS METRICS")
print("=" * 60)
print(f"\n{'Class':<15} {'IoU':<10} {'Sensitivity':<15} {'PPV (Precision)':<15}")
print("-" * 60)

for i in range(NUM_CLASSES):
    iou_str = f"{ious[i]*100:.1f}%" if not np.isnan(ious[i]) else "N/A"
    sens_str = f"{sensitivities[i]*100:.1f}%" if not np.isnan(sensitivities[i]) else "N/A"
    ppv_str = f"{ppvs[i]*100:.1f}%" if not np.isnan(ppvs[i]) else "N/A"

    print(f"{CLASS_NAMES[i]:<15} {iou_str:<10} {sens_str:<15} {ppv_str:<15}")

print("-" * 60)
print(f"{'Mean (excl. BG)':<15} {mean_iou*100:.1f}%{'':<5} {mean_sensitivity*100:.1f}%{'':<10} {mean_ppv*100:.1f}%")

print("\n" + "=" * 60)
print("COMPARISON WITH PAPER (ResNet50, Model 6)")
print("=" * 60)
print("\nAalborg Paper Results:")
print("  Mean IoU: 41.9%")
print("  Per-class IoU:")
print("    - Bunker:  ~65-80%")
print("    - Green:   ~70-80%")
print("    - Fairway: ~65-75%")
print("    - Water:   ~55-65%")
print("    - Tee:     ~40-50%")
print(f"\nYour Results:")
print(f"  Mean IoU: {mean_iou*100:.1f}%")

if mean_iou < 0.30:
    print("\n⚠️  WARNING: Mean IoU is significantly lower than paper's 41.9%")
    print("Possible issues:")
    print("  1. Model undertrained (only 10 epochs vs paper's likely 50+)")
    print("  2. Need more aggressive data augmentation")
    print("  3. Different preprocessing or normalization")
    print("  4. Model architecture differences")
elif mean_iou < 0.42:
    print("\n✓ Results approaching paper's performance")
    print("  Consider training for more epochs to match 41.9%")
else:
    print("\n✅ Results match or exceed paper's performance!")

# %%
# Pixel accuracy
pixel_accuracy = (y_true_flat == y_pred_flat).mean()
print(f"\nOverall Pixel Accuracy: {pixel_accuracy*100:.2f}%")

# Class distribution in test set
print("\n" + "=" * 60)
print("TEST SET CLASS DISTRIBUTION")
print("=" * 60)
print("\nGround Truth:")
for class_id in range(NUM_CLASSES):
    pixels = (y_true_flat == class_id).sum()
    percentage = (pixels / len(y_true_flat)) * 100
    print(f"  {CLASS_NAMES[class_id]:<12}: {percentage:5.2f}% of pixels")

print("\nPredictions:")
for class_id in range(NUM_CLASSES):
    pixels = (y_pred_flat == class_id).sum()
    percentage = (pixels / len(y_pred_flat)) * 100
    print(f"  {CLASS_NAMES[class_id]:<12}: {percentage:5.2f}% of pixels")

# %%
# Visualize some predictions
print("\n" + "=" * 60)
print("SAVING VISUALIZATIONS")
print("=" * 60)

def mask_to_rgb(mask):
    """Convert class mask to RGB."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for class_id in range(NUM_CLASSES):
        rgb[mask == class_id] = CLASS_COLORS[class_id]
    return rgb

# Get first batch for visualization
test_ds_vis = create_test_dataset()[0]
for images, masks in test_ds_vis.take(1):
    predictions = model.predict(images, verbose=0)
    pred_masks = np.argmax(predictions, axis=-1)

    num_samples = min(BATCH_SIZE, len(images))
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 4))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Original
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')

        # Ground truth
        axes[i, 1].imshow(mask_to_rgb(masks[i].numpy().astype(int)))
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')

        # Prediction
        axes[i, 2].imshow(mask_to_rgb(pred_masks[i]))
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'test_predictions.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization: {save_path}")
    plt.show()

# %%
# Summary
print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)
print(f"\n✅ Evaluated on {test_size} test images from Danish Golf Course dataset")
print(f"   Mean IoU: {mean_iou*100:.1f}% (Paper: 41.9%)")
print(f"   Pixel Accuracy: {pixel_accuracy*100:.2f}%")
print(f"\n📁 Results saved to: {OUTPUT_DIR}/")

print("\n" + "=" * 60)
print("NEXT STEPS")
print("=" * 60)

if mean_iou < 0.30:
    print("\n1. Train for more epochs (currently only 10)")
    print("2. Increase data augmentation strength")
    print("3. Try different augmentation techniques (rotation, color jitter)")
    print("4. Check if model is learning (look at training curves)")
else:
    print("\n1. Model is performing reasonably well on Danish test set")
    print("2. Poor UC Merced performance indicates dataset shift")
    print("3. To improve generalization:")
    print("   - Train on more diverse golf course datasets")
    print("   - Use stronger data augmentation")
    print("   - Try domain adaptation techniques")

print("\n✅ Evaluation complete!")
print("=" * 60)

# %%
