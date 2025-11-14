# %%
# TensorFlow/Keras Implementation of U-Net for Golf Course Segmentation

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
# Download dataset
jacotaco_danish_golf_courses_orthophotos_path = kagglehub.dataset_download('jacotaco/danish-golf-courses-orthophotos')

print('Data source import complete.')
print(f"Dataset path: {jacotaco_danish_golf_courses_orthophotos_path}")

# %%
# Hyperparameters
BATCH_SIZE = 16  # Number of batches when training
IMAGE_SIZE = (256, 256)  # Images get resized to a smaller resolution
IN_CHANNELS = 3  # There are 3 channels for RGB
LEARNING_RATE = 1e-4
NUM_CLASSES = 6  # Background, Fairway, Green, Tee, Bunker, Water
MAX_EPOCHS = 50

base_path = jacotaco_danish_golf_courses_orthophotos_path
IMAGES_DIR = os.path.join(base_path, '1. orthophotos')
SEGMASKS_DIR = os.path.join(base_path, '2. segmentation masks')
LABELMASKS_DIR = os.path.join(base_path, '3. class masks')

# %%
# Loading the data
orthophoto_list = os.listdir(IMAGES_DIR)
print("There are ", len(orthophoto_list), " orthophotos in this dataset!")

# Load image with index of 5 (shows all classes)
idx = 5
golf_image = Image.open(os.path.join(IMAGES_DIR, orthophoto_list[idx]))
golf_segmask = Image.open(os.path.join(SEGMASKS_DIR, orthophoto_list[idx].replace(".jpg", ".png")))

# Plot using matplotlib
fig, axes = plt.subplots(1, 2)
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

    # Split the dataset
    train_ds = full_dataset.take(train_size)
    remaining = full_dataset.skip(train_size)
    val_ds = remaining.take(val_size)
    test_ds = remaining.skip(val_size)

    # Batch and prefetch for performance
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds

# %%
# U-Net Model Architecture

def double_conv_block(x, filters):
    """
    Double convolution block: Conv2D -> ReLU -> Conv2D -> ReLU
    """
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.ReLU()(x)
    return x


def build_unet(input_shape=(256, 256, 3), num_classes=6):
    """
    Build U-Net model for semantic segmentation.

    Architecture:
    - Encoder: 5 double-conv blocks with max pooling (3→64→128→256→512→1024 channels)
    - Bottleneck: 1024 channels
    - Decoder: 4 upsampling blocks with skip connections (1024→512→256→128→64)
    - Output: 1x1 convolution to num_classes
    """
    inputs = keras.Input(shape=input_shape)

    # ===== ENCODER =====
    # Block 1: 3 -> 64 channels
    conv1 = double_conv_block(inputs, 64)
    pool1 = layers.MaxPooling2D(pool_size=2, strides=2)(conv1)

    # Block 2: 64 -> 128 channels
    conv2 = double_conv_block(pool1, 128)
    pool2 = layers.MaxPooling2D(pool_size=2, strides=2)(conv2)

    # Block 3: 128 -> 256 channels
    conv3 = double_conv_block(pool2, 256)
    pool3 = layers.MaxPooling2D(pool_size=2, strides=2)(conv3)

    # Block 4: 256 -> 512 channels
    conv4 = double_conv_block(pool3, 512)
    pool4 = layers.MaxPooling2D(pool_size=2, strides=2)(conv4)

    # ===== BOTTLENECK =====
    # Block 5: 512 -> 1024 channels
    conv5 = double_conv_block(pool4, 1024)

    # ===== DECODER =====
    # Upsampling block 1: 1024 -> 512 channels
    up1 = layers.Conv2DTranspose(512, kernel_size=2, strides=2, padding='same')(conv5)
    # Center crop conv4 to match up1 dimensions (if needed)
    conv4_cropped = center_crop(conv4, up1)
    concat1 = layers.Concatenate()([up1, conv4_cropped])
    upconv1 = double_conv_block(concat1, 512)

    # Upsampling block 2: 512 -> 256 channels
    up2 = layers.Conv2DTranspose(256, kernel_size=2, strides=2, padding='same')(upconv1)
    conv3_cropped = center_crop(conv3, up2)
    concat2 = layers.Concatenate()([up2, conv3_cropped])
    upconv2 = double_conv_block(concat2, 256)

    # Upsampling block 3: 256 -> 128 channels
    up3 = layers.Conv2DTranspose(128, kernel_size=2, strides=2, padding='same')(upconv2)
    conv2_cropped = center_crop(conv2, up3)
    concat3 = layers.Concatenate()([up3, conv2_cropped])
    upconv3 = double_conv_block(concat3, 128)

    # Upsampling block 4: 128 -> 64 channels
    up4 = layers.Conv2DTranspose(64, kernel_size=2, strides=2, padding='same')(upconv3)
    conv1_cropped = center_crop(conv1, up4)
    concat4 = layers.Concatenate()([up4, conv1_cropped])
    upconv4 = double_conv_block(concat4, 64)

    # ===== OUTPUT =====
    # Final 1x1 convolution to produce class logits
    outputs = layers.Conv2D(num_classes, kernel_size=1, padding='same')(upconv4)

    model = keras.Model(inputs=inputs, outputs=outputs, name='UNet')
    return model


def center_crop(tensor, target_tensor):
    """
    Center crop tensor to match target_tensor dimensions.
    """
    tensor_shape = tf.shape(tensor)
    target_shape = tf.shape(target_tensor)

    # Calculate cropping amounts
    height_diff = tensor_shape[1] - target_shape[1]
    width_diff = tensor_shape[2] - target_shape[2]

    # Calculate crop offsets (center crop)
    offset_height = height_diff // 2
    offset_width = width_diff // 2

    # Crop the tensor
    cropped = tf.image.crop_to_bounding_box(
        tensor,
        offset_height=offset_height,
        offset_width=offset_width,
        target_height=target_shape[1],
        target_width=target_shape[2]
    )

    return cropped

# %%
# Build and compile the model
print("Building U-Net model...")
model = build_unet(input_shape=(256, 256, 3), num_classes=NUM_CLASSES)

# Compile the model
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Print model summary
model.summary()

# %%
# Prepare datasets
print("Preparing datasets...")
train_ds, val_ds, test_ds = prepare_datasets()

# %%
# Setup callbacks
callback_list = [
    # Save best model based on validation loss
    callbacks.ModelCheckpoint(
        filepath='best_unet_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    # TensorBoard logging
    callbacks.TensorBoard(
        log_dir='./logs',
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
    )
]

# %%
# Train the model
print("Starting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=MAX_EPOCHS,
    callbacks=callback_list,
    verbose=1
)

# %%
# Evaluate on test set
print("Evaluating on test set...")
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
print("Generating predictions on test set...")
output_dir = './predictions'
os.makedirs(output_dir, exist_ok=True)

for batch_idx, (images, masks) in enumerate(test_ds.take(7)):  # Save first 7 batches
    predictions = model.predict(images, verbose=0)
    save_predictions(images, masks, predictions, output_dir, batch_idx)
    print(f"Saved batch {batch_idx + 1}")

# %%
# Visualize predictions
print("Visualizing predictions...")
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
print("Saving final model...")
model.save('final_unet_model.keras')
print("Training complete! Model saved as 'final_unet_model.keras'")

# %%
# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\n=== Training Summary ===")
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Best Validation Loss: {min(history.history['val_loss']):.4f}")
