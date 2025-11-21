# %%
# Binary Golf Course Classifier - WITH DATA AUGMENTATION & STRATIFIED SPLIT
# Improvements:
# 1. Data augmentation for better generalization
# 2. Stratified split - harder negatives in validation
# 3. More Danish golf courses in training set
#
# NOTE: Run this script from the 'classifier/' directory
# All outputs (models, logs) will be saved in this directory

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from datasets import load_dataset

# %%
# GPU Configuration
print("=" * 60)
print("GPU SETUP")
print("=" * 60)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ GPU detected: {len(gpus)} GPU(s)")
else:
    print("⚠️  No GPU detected, using CPU")

# Enable mixed precision for efficiency
keras.mixed_precision.set_global_policy('mixed_float16')
print("✅ Mixed precision enabled")
print("=" * 60)

# %%
# Hyperparameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
MAX_EPOCHS = 15  # Increased from 10 due to augmentation

# %%
# Load UC Merced dataset
print("\n" + "=" * 60)
print("LOADING DATASETS")
print("=" * 60)

print("\n1. Loading UC Merced Land Use dataset...")
ucmerced = load_dataset("blanchon/UC_Merced", split="train")

# UC Merced has 21 classes (100 images each)
print(f"   Total UC Merced images: {len(ucmerced)}")
print(f"   Classes: 21 (100 images each)")

# %%
# Define challenging classes (look similar to golf courses)
ucmerced_classes = [
    "agricultural", "airplane", "baseballdiamond", "beach", "buildings",
    "chaparral", "denseresidential", "forest", "freeway", "golfcourse",
    "harbor", "intersection", "mediumresidential", "mobilehomepark", "overpass",
    "parkinglot", "river", "runway", "sparseresidential", "storagetanks", "tenniscourt"
]

# Challenging classes that might be confused with golf courses
CHALLENGING_CLASSES = {
    2: 'baseballdiamond',      # Sports fields with grass
    20: 'tenniscourt',         # Sports facilities
    0: 'agricultural',         # Green fields
    12: 'mediumresidential',   # Could have lawns/parks
    18: 'sparseresidential',   # Large properties with grass
    6: 'denseresidential',     # Might have parks
    16: 'river',               # Water features like golf courses
    5: 'chaparral',            # Natural vegetation
}

# Easy classes (clearly different from golf courses)
EASY_CLASSES = {
    1: 'airplane',
    3: 'beach',
    4: 'buildings',
    7: 'forest',
    8: 'freeway',
    10: 'harbor',
    11: 'intersection',
    13: 'mobilehomepark',
    14: 'overpass',
    15: 'parkinglot',
    17: 'runway',
    19: 'storagetanks',
}

print(f"\n   Class distribution:")
print(f"   Challenging classes (8): {list(CHALLENGING_CLASSES.values())}")
print(f"   Easy classes (12): {list(EASY_CLASSES.values())}")
print(f"   Golf course class (1): golfcourse (class 9)")

# %%
# Load Danish Golf Course dataset (positive examples)
print("\n2. Loading Danish Golf Course dataset...")

import kagglehub
golf_dataset_path = kagglehub.dataset_download('jacotaco/danish-golf-courses-orthophotos')
IMAGES_DIR = os.path.join(golf_dataset_path, '1. orthophotos')

danish_golf_files = [os.path.join(IMAGES_DIR, f) for f in os.listdir(IMAGES_DIR)]
print(f"   Danish golf courses: {len(danish_golf_files)} images")

# %%
# Extract UC Merced golf courses (class 9)
print("\n3. Extracting UC Merced golf courses...")

ucmerced_golf_samples = [item for item in ucmerced if item['label'] == 9]
print(f"   UC Merced golf courses: {len(ucmerced_golf_samples)} images")

print(f"\n   Total golf courses (positive examples): {len(danish_golf_files) + len(ucmerced_golf_samples)}")

# %%
# Data augmentation layer (applied only to training data)

def get_augmentation_layer():
    """
    Creates data augmentation pipeline for training.
    Applied to images AFTER normalization.

    Gentler augmentation settings to avoid distorting images too much.
    """
    return keras.Sequential([
        # Random horizontal flip (50% chance)
        layers.RandomFlip("horizontal"),

        # Random rotation (±18 degrees) - reduced from ±28°
        layers.RandomRotation(0.05),  # 0.05 * 2π ≈ 18 degrees

        # Random zoom (90-110%) - kept at moderate level
        layers.RandomZoom(0.1),

        # Random brightness adjustment (±10%) - reduced from ±20%
        layers.RandomBrightness(0.1),

        # Random contrast adjustment (90-110%) - reduced from 80-120%
        layers.RandomContrast(0.1),
    ], name='augmentation')


# %%
# Prepare datasets

def prepare_danish_golf_images(image_paths, target_size=(224, 224)):
    """Load and preprocess Danish golf course images from file paths."""
    images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        images.append(img_array)
    return np.array(images, dtype=np.float32)


def prepare_ucmerced_golf_images(golf_samples, target_size=(224, 224)):
    """Load and preprocess UC Merced golf course images."""
    images = []
    for sample in golf_samples:
        img = sample['image']
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        images.append(img_array)
    return np.array(images, dtype=np.float32)


def prepare_ucmerced_negatives_by_difficulty(dataset, target_size=(224, 224)):
    """
    Load UC Merced negative examples separated by difficulty.

    Returns:
        challenging_images: Array of challenging negative examples
        easy_images: Array of easy negative examples
    """
    challenging_images = []
    easy_images = []

    # Load challenging classes
    for class_id in CHALLENGING_CLASSES.keys():
        class_samples = [item for item in dataset if item['label'] == class_id]
        for sample in class_samples:
            img = sample['image'].resize(target_size)
            img_array = np.array(img) / 255.0
            challenging_images.append(img_array)

    # Load easy classes
    for class_id in EASY_CLASSES.keys():
        class_samples = [item for item in dataset if item['label'] == class_id]
        for sample in class_samples:
            img = sample['image'].resize(target_size)
            img_array = np.array(img) / 255.0
            easy_images.append(img_array)

    return np.array(challenging_images, dtype=np.float32), np.array(easy_images, dtype=np.float32)


print("\n4. Preparing datasets...")

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load positive examples (Danish Golf + UC Merced Golf)
print("   Loading Danish golf course images...")
danish_golf_images = prepare_danish_golf_images(danish_golf_files, IMAGE_SIZE)

print("   Loading UC Merced golf course images...")
ucmerced_golf_images = prepare_ucmerced_golf_images(ucmerced_golf_samples, IMAGE_SIZE)

# Load negative examples separated by difficulty
print("   Loading UC Merced negative examples by difficulty...")
challenging_negatives, easy_negatives = prepare_ucmerced_negatives_by_difficulty(ucmerced, IMAGE_SIZE)

print(f"\n   Dataset breakdown:")
print(f"   Positive examples:")
print(f"     - Danish golf: {len(danish_golf_images)}")
print(f"     - UC Merced golf: {len(ucmerced_golf_images)}")
print(f"     - Total positives: {len(danish_golf_images) + len(ucmerced_golf_images)}")
print(f"   Negative examples:")
print(f"     - Challenging (look like golf): {len(challenging_negatives)}")
print(f"     - Easy (clearly different): {len(easy_negatives)}")
print(f"     - Total negatives: {len(challenging_negatives) + len(easy_negatives)}")

# %%
# Stratified split with strategic distribution
print("\n5. Creating stratified train/validation/test split...")
print("   Strategy:")
print("   - More Danish golf in training (for primary task learning)")
print("   - More challenging negatives in validation (for realistic eval)")
print("   - Balanced test set (for unbiased final metrics)")

# Split Danish golf courses: 80% train, 10% val, 10% test
n_danish = len(danish_golf_images)
danish_indices = np.random.permutation(n_danish)

danish_train_size = int(0.80 * n_danish)  # 80% → training
danish_val_size = int(0.10 * n_danish)    # 10% → validation
# Remaining 10% → test

danish_train_idx = danish_indices[:danish_train_size]
danish_val_idx = danish_indices[danish_train_size:danish_train_size + danish_val_size]
danish_test_idx = danish_indices[danish_train_size + danish_val_size:]

# Split UC Merced golf: 60% train, 20% val, 20% test
n_ucm_golf = len(ucmerced_golf_images)
ucm_golf_indices = np.random.permutation(n_ucm_golf)

ucm_golf_train_size = int(0.60 * n_ucm_golf)
ucm_golf_val_size = int(0.20 * n_ucm_golf)

ucm_golf_train_idx = ucm_golf_indices[:ucm_golf_train_size]
ucm_golf_val_idx = ucm_golf_indices[ucm_golf_train_size:ucm_golf_train_size + ucm_golf_val_size]
ucm_golf_test_idx = ucm_golf_indices[ucm_golf_train_size + ucm_golf_val_size:]

# Split challenging negatives: 50% train, 30% val, 20% test
# (More in validation to make eval harder)
n_challenging = len(challenging_negatives)
challenging_indices = np.random.permutation(n_challenging)

challenging_train_size = int(0.50 * n_challenging)
challenging_val_size = int(0.30 * n_challenging)

challenging_train_idx = challenging_indices[:challenging_train_size]
challenging_val_idx = challenging_indices[challenging_train_size:challenging_train_size + challenging_val_size]
challenging_test_idx = challenging_indices[challenging_train_size + challenging_val_size:]

# Split easy negatives: 70% train, 20% val, 10% test
n_easy = len(easy_negatives)
easy_indices = np.random.permutation(n_easy)

easy_train_size = int(0.70 * n_easy)
easy_val_size = int(0.20 * n_easy)

easy_train_idx = easy_indices[:easy_train_size]
easy_val_idx = easy_indices[easy_train_size:easy_train_size + easy_val_size]
easy_test_idx = easy_indices[easy_train_size + easy_val_size:]

# Combine training set
train_images = np.concatenate([
    danish_golf_images[danish_train_idx],
    ucmerced_golf_images[ucm_golf_train_idx],
    challenging_negatives[challenging_train_idx],
    easy_negatives[easy_train_idx]
])

train_labels = np.concatenate([
    np.ones(len(danish_train_idx)),           # Danish golf
    np.ones(len(ucm_golf_train_idx)),         # UC Merced golf
    np.zeros(len(challenging_train_idx)),     # Challenging negatives
    np.zeros(len(easy_train_idx))             # Easy negatives
])

# Combine validation set
val_images = np.concatenate([
    danish_golf_images[danish_val_idx],
    ucmerced_golf_images[ucm_golf_val_idx],
    challenging_negatives[challenging_val_idx],
    easy_negatives[easy_val_idx]
])

val_labels = np.concatenate([
    np.ones(len(danish_val_idx)),
    np.ones(len(ucm_golf_val_idx)),
    np.zeros(len(challenging_val_idx)),
    np.zeros(len(easy_val_idx))
])

# Combine test set
test_images = np.concatenate([
    danish_golf_images[danish_test_idx],
    ucmerced_golf_images[ucm_golf_test_idx],
    challenging_negatives[challenging_test_idx],
    easy_negatives[easy_test_idx]
])

test_labels = np.concatenate([
    np.ones(len(danish_test_idx)),
    np.ones(len(ucm_golf_test_idx)),
    np.zeros(len(challenging_test_idx)),
    np.zeros(len(easy_test_idx))
])

# Shuffle each set
train_shuffle = np.random.permutation(len(train_images))
train_images = train_images[train_shuffle]
train_labels = train_labels[train_shuffle]

val_shuffle = np.random.permutation(len(val_images))
val_images = val_images[val_shuffle]
val_labels = val_labels[val_shuffle]

test_shuffle = np.random.permutation(len(test_images))
test_images = test_images[test_shuffle]
test_labels = test_labels[test_shuffle]

# Print split statistics
total_size = len(train_images) + len(val_images) + len(test_images)

print(f"\n   Split results:")
print(f"   Training set:   {len(train_images):4d} images ({len(train_images)/total_size*100:.1f}%)")
print(f"     - Golf courses:      {int(train_labels.sum())} ({int(train_labels.sum())/len(train_labels)*100:.1f}%)")
print(f"     - Danish golf:       {len(danish_train_idx)} (primary dataset)")
print(f"     - Challenging negs:  {len(challenging_train_idx)}")
print(f"     - Easy negatives:    {len(easy_train_idx)}")

print(f"   Validation set: {len(val_images):4d} images ({len(val_images)/total_size*100:.1f}%)")
print(f"     - Golf courses:      {int(val_labels.sum())} ({int(val_labels.sum())/len(val_labels)*100:.1f}%)")
print(f"     - Challenging negs:  {len(challenging_val_idx)} (harder evaluation!)")
print(f"     - Easy negatives:    {len(easy_val_idx)}")

print(f"   Test set:       {len(test_images):4d} images ({len(test_images)/total_size*100:.1f}%)")
print(f"     - Golf courses:      {int(test_labels.sum())} ({int(test_labels.sum())/len(test_labels)*100:.1f}%)")
print(f"     - Challenging negs:  {len(challenging_test_idx)}")
print(f"     - Easy negatives:    {len(easy_test_idx)}")

# %%
# Create TensorFlow datasets WITH augmentation for training

augmentation_layer = get_augmentation_layer()

# 50/50 mix: original + augmented images
def augment_with_passthrough(images, labels):
    """
    Apply augmentation with 50% probability.

    50% chance: Return original images (no augmentation)
    50% chance: Apply full augmentation pipeline

    This helps model learn from both clean originals and augmented variations.

    Note: Uses tf.cond with explicit dtype casting to handle mixed precision correctly.
    """
    def apply_augmentation():
        """Apply augmentation and cast to float32."""
        augmented = augmentation_layer(images, training=True)
        return tf.cast(augmented, tf.float32)

    def keep_original():
        """Keep original images and cast to float32."""
        return tf.cast(images, tf.float32)

    # 50/50 chance
    should_augment = tf.random.uniform([]) >= 0.5

    # Use tf.cond to ensure proper dtype handling in graph mode
    result = tf.cond(
        should_augment,
        apply_augmentation,
        keep_original
    )

    return result, labels

# Training dataset WITH 50/50 augmentation mix
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_ds = train_ds.shuffle(1000, reshuffle_each_iteration=True)  # Shuffle every epoch
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.map(
    augment_with_passthrough,
    num_parallel_calls=tf.data.AUTOTUNE
)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

# Validation dataset WITHOUT augmentation
val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Test dataset WITHOUT augmentation
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("\n✅ Data augmentation enabled: 50% original, 50% augmented")
print("=" * 60)

# %%
# Build classifier model

def build_golf_classifier(input_shape=(224, 224, 3)):
    """
    Build binary classifier using MobileNetV2 transfer learning.

    Returns:
        Binary classifier: 0 = NOT golf course, 1 = golf course
    """
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Freeze base model layers
    base_model.trainable = False

    # Build classifier on top
    inputs = keras.Input(shape=input_shape)

    # Preprocessing for MobileNetV2
    x = keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)

    # Base model
    x = base_model(x, training=False)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)

    # Output layer (binary classification)
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='GolfCourseClassifier_Augmented')

    return model


print("\n" + "=" * 60)
print("BUILDING MODEL")
print("=" * 60)

print("Building classifier with MobileNetV2 backbone...")
classifier_model = build_golf_classifier(input_shape=(*IMAGE_SIZE, 3))

# Compile model
classifier_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

print("\n✅ Model built successfully!")
classifier_model.summary()

print("=" * 60)

# %%
# Setup callbacks

callback_list = [
    # Save best model
    callbacks.ModelCheckpoint(
        filepath='best_golf_classifier_augmented.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    # Early stopping
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,  # Increased patience for augmentation
        restore_best_weights=True,
        verbose=1
    ),
    # Reduce learning rate on plateau
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,  # Increased patience
        verbose=1,
        min_lr=1e-7
    ),
    # TensorBoard logging
    callbacks.TensorBoard(
        log_dir='./logs/classifier_augmented',
        histogram_freq=1
    )
]

# %%
# Train the model

print("\n" + "=" * 60)
print("TRAINING CLASSIFIER WITH AUGMENTATION")
print("=" * 60)
print("Note: Training may take longer due to data augmentation")
print("Expected: More epochs needed but better generalization")

history = classifier_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=MAX_EPOCHS,
    callbacks=callback_list,
    verbose=1
)

print("\n✅ Training complete!")

# %%
# Evaluate model

print("\n" + "=" * 60)
print("EVALUATION")
print("=" * 60)

# Validation set evaluation
print("\n📊 Validation Set Performance (challenging negatives):")
val_loss, val_accuracy, val_precision, val_recall = classifier_model.evaluate(val_ds, verbose=0)
print(f"  Accuracy:  {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
print(f"  Precision: {val_precision:.4f}")
print(f"  Recall:    {val_recall:.4f}")
print(f"  Loss:      {val_loss:.4f}")

val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0
print(f"  F1 Score:  {val_f1:.4f}")

# Test set evaluation
print("\n🎯 Test Set Performance (balanced, unbiased):")
test_loss, test_accuracy, test_precision, test_recall = classifier_model.evaluate(test_ds, verbose=0)
print(f"  Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  Loss:      {test_loss:.4f}")

test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall) if (test_precision + test_recall) > 0 else 0
print(f"  F1 Score:  {test_f1:.4f}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Validation Accuracy: {val_accuracy*100:.2f}% (harder eval with challenging negatives)")
print(f"Test Accuracy:       {test_accuracy*100:.2f}% (balanced, final metric)")
print("\nImprovements over baseline:")
print("✅ Data augmentation → Better generalization")
print("✅ Stratified split → More realistic validation")
print("✅ More Danish golf in training → Better task learning")
print("=" * 60)

# %%
# Save final model

print("\nSaving final model...")
classifier_model.save('final_golf_classifier_augmented.keras')
print("✅ Model saved as 'final_golf_classifier_augmented.keras'")

# %%
# Plot training history

plt.figure(figsize=(14, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy (With Augmentation)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss (With Augmentation)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history_augmented.png', dpi=150, bbox_inches='tight')
print("✅ Training history plot saved as 'training_history_augmented.png'")
plt.show()

# %%
# Test predictions on sample images

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

# Test on validation images (includes challenging cases)
sample_indices = np.random.choice(len(val_images), size=6, replace=False)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, idx in enumerate(sample_indices):
    img = val_images[idx]
    true_label = val_labels[idx]

    # Predict
    pred_prob = classifier_model.predict(np.expand_dims(img, axis=0), verbose=0)[0][0]
    pred_label = 1 if pred_prob > 0.5 else 0

    # Plot
    axes[i].imshow(img)
    axes[i].set_title(
        f"True: {'Golf' if true_label == 1 else 'Not Golf'}\n"
        f"Pred: {'Golf' if pred_label == 1 else 'Not Golf'} ({pred_prob:.2%})",
        color='green' if pred_label == true_label else 'red'
    )
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('sample_predictions_augmented.png', dpi=150, bbox_inches='tight')
print("✅ Sample predictions saved as 'sample_predictions_augmented.png'")
plt.show()

print("\n✅ Classifier training complete with augmentation!")
print("=" * 60)

# %%
