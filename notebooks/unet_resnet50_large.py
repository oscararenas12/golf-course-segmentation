# %%
# TensorFlow/Keras U-Net with ResNet50 Encoder - Aalborg Paper Replication
# Image size: 832×512 (paper's resolution)
# Encoder: ResNet50 with ImageNet pre-training

import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import kagglehub

# %%
# GPU Configuration and Verification
print("=" * 60)
print("GPU SETUP AND VERIFICATION")
print("=" * 60)

# Check TensorFlow version
print(f"\nTensorFlow Version: {tf.__version__}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

# List all physical devices
print("\nAll Physical Devices:")
for device in tf.config.list_physical_devices():
    print(f"  {device}")

# Get GPU devices
gpus = tf.config.list_physical_devices('GPU')
print(f"\nNumber of GPUs Available: {len(gpus)}")

if gpus:
    try:
        # Configure GPU memory growth to avoid OOM errors
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Get detailed GPU info
        for i, gpu in enumerate(gpus):
            print(f"\nGPU {i}:")
            print(f"  Device: {gpu}")
            details = tf.config.experimental.get_device_details(gpu)
            if details:
                for key, value in details.items():
                    print(f"  {key}: {value}")

        # Set the GPU as visible
        tf.config.set_visible_devices(gpus, 'GPU')

        # Enable mixed precision for better memory efficiency and faster training
        keras.mixed_precision.set_global_policy('mixed_float16')
        print("\n✅ Mixed precision training enabled (float16)")

        print("\n✅ GPU Configuration Successful!")
        print(f"✅ Memory growth enabled for {len(gpus)} GPU(s)")

    except RuntimeError as e:
        print(f"\n❌ GPU Configuration Error: {e}")
else:
    print("\n⚠️  WARNING: No GPUs detected! Training will use CPU (very slow)")
    print("    Make sure you're running in WSL2 with CUDA support")

print("=" * 60)

# %%
# Download dataset
jacotaco_danish_golf_courses_orthophotos_path = kagglehub.dataset_download('jacotaco/danish-golf-courses-orthophotos')

print('Data source import complete.')
print(f"Dataset path: {jacotaco_danish_golf_courses_orthophotos_path}")

# %%
# Hyperparameters - Updated to match Aalborg paper
BATCH_SIZE = 2  # Reduced for larger images (832×512)
IMAGE_SIZE = (512, 832)  # Paper's resolution: Height×Width
IN_CHANNELS = 3  # There are 3 channels for RGB
LEARNING_RATE = 1e-4
NUM_CLASSES = 6  # Background, Fairway, Green, Tee, Bunker, Water
MAX_EPOCHS = 10  # Reduced for faster iteration
AUGMENTATION_PROBABILITY = 0.25  # Apply augmentation to 25% of training images

base_path = jacotaco_danish_golf_courses_orthophotos_path
IMAGES_DIR = os.path.join(base_path, '1. orthophotos')
SEGMASKS_DIR = os.path.join(base_path, '2. segmentation masks')
LABELMASKS_DIR = os.path.join(base_path, '3. class masks')

print(f"\nConfiguration:")
print(f"  Image Size: {IMAGE_SIZE[1]}×{IMAGE_SIZE[0]} (W×H)")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  Max Epochs: {MAX_EPOCHS}")
print(f"  Data Augmentation: {AUGMENTATION_PROBABILITY * 100}% of training images")
print(f"  Encoder: ResNet50 (ImageNet pre-trained)")

# %%
# Loading the data
orthophoto_list = os.listdir(IMAGES_DIR)
print(f"\nThere are {len(orthophoto_list)} orthophotos in this dataset!")

# Load image with index of 5 (shows all classes)
idx = 5
golf_image = Image.open(os.path.join(IMAGES_DIR, orthophoto_list[idx]))
golf_segmask = Image.open(os.path.join(SEGMASKS_DIR, orthophoto_list[idx].replace(".jpg", ".png")))

# Plot using matplotlib
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].set_title('Orthophoto')
axes[1].set_title('Segmentation Mask')
axes[0].imshow(golf_image)
axes[1].imshow(golf_segmask)
plt.show()

# %%
# Data loading functions

def load_and_preprocess_image(image_path, mask_path):
    """Load and preprocess a single image and mask pair."""
    # Load image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]

    # Load mask
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, IMAGE_SIZE, method='nearest')  # Use nearest to preserve label values
    mask = tf.cast(mask, tf.float32)
    mask = tf.squeeze(mask, axis=-1)  # Remove channel dimension: (H, W, 1) -> (H, W)

    return image, mask


def augment_image_and_mask(image, mask):
    """
    Apply synchronized augmentation to both image and mask.
    Augmentation is applied with AUGMENTATION_PROBABILITY chance.

    IMPORTANT: Geometric transforms (flip, rotation) must be applied identically
    to both image and mask. Color transforms only apply to image.
    """
    def apply_augmentation():
        # Concatenate image and mask for synchronized geometric transforms
        # Expand mask to 3D: (H, W) -> (H, W, 1)
        mask_expanded = tf.expand_dims(mask, axis=-1)

        # Random horizontal flip (50% chance)
        combined = tf.concat([image, mask_expanded], axis=-1)  # (H, W, 4)
        combined = tf.image.random_flip_left_right(combined)

        # Random rotation (±10 degrees)
        # Note: tf.image doesn't have random_rotation, so we use small angle approximation
        # Or we can skip rotation for now and just use flips

        # Split back
        aug_image = combined[:, :, :3]
        aug_mask = combined[:, :, 3]

        # Color augmentation (only on image, not mask)
        aug_image = tf.image.random_brightness(aug_image, 0.1)  # ±10%
        aug_image = tf.image.random_contrast(aug_image, 0.9, 1.1)  # 90-110%
        aug_image = tf.clip_by_value(aug_image, 0.0, 1.0)  # Keep in valid range

        return tf.cast(aug_image, tf.float32), aug_mask

    def keep_original():
        return tf.cast(image, tf.float32), mask

    # Apply augmentation with AUGMENTATION_PROBABILITY
    should_augment = tf.random.uniform([]) < AUGMENTATION_PROBABILITY

    aug_image, aug_mask = tf.cond(
        should_augment,
        apply_augmentation,
        keep_original
    )

    return aug_image, aug_mask


def create_dataset(images_dir, labelmasks_dir, shuffle=True):
    """Create TensorFlow dataset from image and mask directories."""
    # Get all image paths
    image_filenames = sorted(os.listdir(images_dir))
    image_paths = [os.path.join(images_dir, fname) for fname in image_filenames]
    mask_paths = [os.path.join(labelmasks_dir, fname.replace('.jpg', '.png')) for fname in image_filenames]

    # Create dataset from paths
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), seed=42)

    # Map the load and preprocess function
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset, len(image_paths)


def prepare_datasets():
    """Prepare train, validation, and test datasets with 70/20/10 split."""
    # Create full dataset
    full_dataset, total_size = create_dataset(IMAGES_DIR, LABELMASKS_DIR, shuffle=True)

    # Calculate split sizes
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    print(f"Dataset split: {train_size} train, {val_size} val, {test_size} test")
    print(f"Data augmentation: {AUGMENTATION_PROBABILITY * 100}% of training images")

    # Split the dataset
    train_ds = full_dataset.take(train_size)
    remaining = full_dataset.skip(train_size)
    val_ds = remaining.take(val_size)
    test_ds = remaining.skip(val_size)

    # Apply augmentation ONLY to training set
    train_ds = train_ds.map(augment_image_and_mask, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch for performance
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds

# %%
# U-Net Model Architecture with ResNet50 Encoder

def build_unet_resnet50(input_shape=(512, 832, 3), num_classes=6):
    """
    Build U-Net model with ResNet50 encoder (transfer learning from ImageNet).

    Architecture:
    - Encoder: ResNet50 pre-trained on ImageNet
    - Decoder: Upsampling blocks with skip connections from encoder
    - Output: Segmentation mask with num_classes channels

    This matches the paper's approach of using transfer learning.
    """
    inputs = keras.Input(shape=input_shape)

    # ===== ENCODER: ResNet50 (pre-trained on ImageNet) =====
    base_model = keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )

    # Extract feature maps at different scales for skip connections
    # ResNet50 layer names for different resolutions:
    skip_layer_names = [
        'conv1_relu',           # 1/2 resolution: (H/2, W/2, 64)
        'conv2_block3_out',     # 1/4 resolution: (H/4, W/4, 256)
        'conv3_block4_out',     # 1/8 resolution: (H/8, W/8, 512)
        'conv4_block6_out',     # 1/16 resolution: (H/16, W/16, 1024)
    ]

    skip_connections = [base_model.get_layer(name).output for name in skip_layer_names]
    bottleneck = base_model.get_layer('conv5_block3_out').output  # 1/32 resolution: (H/32, W/32, 2048)

    # ===== DECODER: Upsampling with skip connections =====

    # Upsampling block 1: 2048 -> 1024 channels (1/32 -> 1/16)
    x = layers.Conv2DTranspose(512, kernel_size=2, strides=2, padding='same')(bottleneck)
    x = layers.Concatenate()([x, skip_connections[3]])  # Skip from conv4_block6_out
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)

    # Upsampling block 2: 1024 -> 512 channels (1/16 -> 1/8)
    x = layers.Conv2DTranspose(256, kernel_size=2, strides=2, padding='same')(x)
    x = layers.Concatenate()([x, skip_connections[2]])  # Skip from conv3_block4_out
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)

    # Upsampling block 3: 512 -> 256 channels (1/8 -> 1/4)
    x = layers.Conv2DTranspose(128, kernel_size=2, strides=2, padding='same')(x)
    x = layers.Concatenate()([x, skip_connections[1]])  # Skip from conv2_block3_out
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)

    # Upsampling block 4: 256 -> 128 channels (1/4 -> 1/2)
    x = layers.Conv2DTranspose(64, kernel_size=2, strides=2, padding='same')(x)
    x = layers.Concatenate()([x, skip_connections[0]])  # Skip from conv1_relu
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)

    # Upsampling block 5: 128 -> 64 channels (1/2 -> 1/1, full resolution)
    x = layers.Conv2DTranspose(32, kernel_size=2, strides=2, padding='same')(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)

    # ===== OUTPUT LAYER =====
    # Final 1x1 convolution to produce class logits
    outputs = layers.Conv2D(num_classes, kernel_size=1, padding='same', dtype='float32')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='UNet_ResNet50')
    return model


# %%
# Build and compile the model
print("\nBuilding U-Net model with ResNet50 encoder...")
print("Loading ImageNet pre-trained weights...")
model = build_unet_resnet50(input_shape=(*IMAGE_SIZE, 3), num_classes=NUM_CLASSES)

# Compile the model
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Print model summary
print("\n" + "=" * 60)
print("MODEL SUMMARY")
print("=" * 60)
model.summary()

# %%
# Prepare datasets
print("\nPreparing datasets...")
train_ds, val_ds, test_ds = prepare_datasets()

# %%
# Custom GPU Monitoring Callback
class GPUMonitorCallback(callbacks.Callback):
    """Monitor GPU memory usage during training."""
    def on_epoch_end(self, epoch, logs=None):
        if tf.config.list_physical_devices('GPU'):
            print(f"\n[Epoch {epoch + 1}] Training on GPU")

# %%
# Setup callbacks
callback_list = [
    # Save best model based on validation loss
    callbacks.ModelCheckpoint(
        filepath='../../models/best_unet_resnet50_large.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    # TensorBoard logging
    callbacks.TensorBoard(
        log_dir='../../logs/resnet50_large',
        histogram_freq=1
    ),
    # Early stopping to prevent overfitting
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate on plateau
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1,
        min_lr=1e-7
    ),
    # GPU monitoring
    GPUMonitorCallback()
]

# %%
# Train the model
print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)
print(f"Training on: {tf.test.gpu_device_name() if tf.config.list_physical_devices('GPU') else 'CPU'}")
print(f"Image size: {IMAGE_SIZE[1]}×{IMAGE_SIZE[0]} (W×H)")
print(f"Batch size: {BATCH_SIZE}")
print(f"Max epochs: {MAX_EPOCHS}")
print(f"Data augmentation: {AUGMENTATION_PROBABILITY * 100}% of training images")
print(f"Encoder: ResNet50 (ImageNet pre-trained)")
print("=" * 60)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=MAX_EPOCHS,
    callbacks=callback_list,
    verbose=1
)

# %%
# Evaluate on test set
print("\n" + "=" * 60)
print("EVALUATING ON TEST SET")
print("=" * 60)
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# %%
# Visualization functions

# Class colors for segmentation masks
CLASS_COLORS = np.array([
    [0, 0, 0],                      # Background - Black
    [0, 140, 0],                    # Fairway - Dark green
    [0, 255, 0],                    # Green - Bright green
    [255, 0, 0],                    # Tee - Red
    [217, 230, 122],                # Bunker - Sandy yellow
    [7, 15, 247]                    # Water - Blue
], dtype=np.float32) / 255.0  # Normalize to [0, 1]


def mask_to_rgb(mask):
    """Convert single-channel mask to RGB visualization."""
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.float32)

    for class_id in range(NUM_CLASSES):
        rgb_mask[mask == class_id] = CLASS_COLORS[class_id]

    return rgb_mask


def save_predictions(images, masks_true, masks_pred, output_dir='./predictions', batch_idx=0):
    """
    Save predictions as RGB images for visualization.

    Args:
        images: Input images (batch_size, H, W, 3)
        masks_true: Ground truth masks (batch_size, H, W)
        masks_pred: Predicted logits (batch_size, H, W, num_classes)
        output_dir: Directory to save images
        batch_idx: Batch index for filename
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert logits to class predictions (argmax)
    if len(masks_pred.shape) == 4:  # (B, H, W, C)
        masks_pred = np.argmax(masks_pred, axis=-1)  # (B, H, W)

    batch_size = images.shape[0]

    for i in range(batch_size):
        # Get single samples
        img = images[i].numpy() if tf.is_tensor(images) else images[i]
        mask_true = masks_true[i].numpy() if tf.is_tensor(masks_true) else masks_true[i]
        mask_pred = masks_pred[i] if isinstance(masks_pred, np.ndarray) else masks_pred[i].numpy()

        # Convert masks to RGB
        mask_true_rgb = mask_to_rgb(mask_true.astype(np.int32))
        mask_pred_rgb = mask_to_rgb(mask_pred.astype(np.int32))

        # Save images
        sample_idx = batch_idx * BATCH_SIZE + i + 1
        Image.fromarray((img * 255).astype(np.uint8)).save(
            os.path.join(output_dir, f'{sample_idx}_figure.jpg')
        )
        Image.fromarray((mask_true_rgb * 255).astype(np.uint8)).save(
            os.path.join(output_dir, f'{sample_idx}_groundtruth.jpg')
        )
        Image.fromarray((mask_pred_rgb * 255).astype(np.uint8)).save(
            os.path.join(output_dir, f'{sample_idx}_prediction.jpg')
        )


# %%
# Generate and save predictions on test set
print("\nGenerating predictions on test set...")
output_dir = '../../predictions/resnet50_large'
os.makedirs(output_dir, exist_ok=True)

for batch_idx, (images, masks) in enumerate(test_ds.take(7)):  # Save first 7 batches
    predictions = model.predict(images, verbose=0)
    save_predictions(images, masks, predictions, output_dir, batch_idx)
    print(f"Saved batch {batch_idx + 1}")

# %%
# Visualize predictions
print("\nVisualizing predictions...")
for idx in range(1, 8):  # Show first 7 batches
    try:
        orthophoto = Image.open(os.path.join(output_dir, f'{idx}_figure.jpg'))
        groundtruth = Image.open(os.path.join(output_dir, f'{idx}_groundtruth.jpg'))
        prediction = Image.open(os.path.join(output_dir, f'{idx}_prediction.jpg'))

        # Plot using matplotlib
        fig, axes = plt.subplots(1, 3)
        fig.set_size_inches(18.5, 10)

        axes[0].set_title('Orthophoto')
        axes[1].set_title('Groundtruth')
        axes[2].set_title('Prediction')

        axes[0].imshow(orthophoto)
        axes[1].imshow(groundtruth)
        axes[2].imshow(prediction)

        axes[0].axis('off')
        axes[1].axis('off')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()
    except FileNotFoundError:
        print(f"Images for sample {idx} not found")
        break

# %%
# Save final model
print("\nSaving final model...")
model.save('../../models/final_unet_resnet50_large.keras')
print("Training complete! Model saved as 'models/final_unet_resnet50_large.keras'")

# %%
# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss - ResNet50 U-Net')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy - ResNet50 U-Net')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)
print(f"Configuration:")
print(f"  Encoder: ResNet50 (ImageNet pre-trained)")
print(f"  Image Size: {IMAGE_SIZE[1]}×{IMAGE_SIZE[0]} (W×H)")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"\nResults:")
print(f"  Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"  Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
print(f"  Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"  Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"  Best Validation Loss: {min(history.history['val_loss']):.4f}")
print(f"\nTest Set Performance:")
print(f"  Test Loss: {test_loss:.4f}")
print(f"  Test Accuracy: {test_accuracy:.4f}")
print("=" * 60)

# %%
