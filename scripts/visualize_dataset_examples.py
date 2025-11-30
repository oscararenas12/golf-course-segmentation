# %%

"""
Dataset Example Visualizer for Paper Figures

This script generates example images from:
1. Danish Golf Courses Orthophoto Dataset (positive examples)
2. UC Merced Land Use Dataset (negative examples - challenging vs easy)

Run from the project root:
    python scripts/visualize_dataset_examples.py
"""
# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import kagglehub
from datasets import load_dataset

# %%
# Create output directory
OUTPUT_DIR = "paper_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATASETS
# ============================================================
print("=" * 60)
print("LOADING DATASETS")
print("=" * 60)

# Load UC Merced from HuggingFace
print("\n1. Loading UC Merced Land Use dataset...")
ucmerced = load_dataset("blanchon/UC_Merced", split="train")
print(f"   Loaded {len(ucmerced)} images")

# UC Merced class names
UCMERCED_CLASSES = [
    "agricultural", "airplane", "baseballdiamond", "beach", "buildings",
    "chaparral", "denseresidential", "forest", "freeway", "golfcourse",
    "harbor", "intersection", "mediumresidential", "mobilehomepark", "overpass",
    "parkinglot", "river", "runway", "sparseresidential", "storagetanks", "tenniscourt"
]

# Load Danish Golf Courses from Kaggle
print("\n2. Loading Danish Golf Courses dataset...")
golf_dataset_path = kagglehub.dataset_download('jacotaco/danish-golf-courses-orthophotos')
IMAGES_DIR = os.path.join(golf_dataset_path, '1. orthophotos')
danish_golf_files = sorted([os.path.join(IMAGES_DIR, f) for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))])
print(f"   Loaded {len(danish_golf_files)} Danish golf course images")

# ============================================================
# CATEGORY DEFINITIONS
# ============================================================

# Challenging negatives (visually similar to golf courses)
CHALLENGING_CLASSES = {
    2: 'baseballdiamond',      # Sports fields with grass
    20: 'tenniscourt',         # Sports facilities
    0: 'agricultural',         # Green fields
    12: 'mediumresidential',   # Could have lawns/parks
    18: 'sparseresidential',   # Large properties with grass
    16: 'river',               # Water features
}

# Easy negatives (clearly different from golf courses)
EASY_CLASSES = {
    1: 'airplane',
    8: 'freeway',
    10: 'harbor',
    7: 'forest',
    17: 'runway',
    15: 'parkinglot',
}

# ============================================================
# FIGURE 1: DANISH GOLF COURSES (Positive Examples)
# ============================================================
def plot_danish_golf_examples(num_examples=12, save_name="fig1_danish_golf_courses.png"):
    """Plot example images from Danish Golf Courses dataset."""
    print(f"\nGenerating: {save_name}")

    # Select evenly spaced examples
    indices = np.linspace(0, len(danish_golf_files)-1, num_examples, dtype=int)

    cols = 4
    rows = (num_examples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    fig.suptitle('Danish Golf Courses Orthophoto Dataset\n(Positive Examples - Golf)',
                 fontsize=14, fontweight='bold', y=1.02)

    axes = axes.flatten() if num_examples > cols else [axes] if num_examples == 1 else axes

    for i, idx in enumerate(indices):
        img = Image.open(danish_golf_files[idx])
        axes[i].imshow(img)
        axes[i].axis('off')
        # Get filename without path
        filename = os.path.basename(danish_golf_files[idx])
        axes[i].set_title(f'Golf Course {idx+1}', fontsize=10)

    # Hide unused axes
    for j in range(len(indices), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"   Saved to {OUTPUT_DIR}/{save_name}")


# ============================================================
# FIGURE 2: UC MERCED GOLF COURSES
# ============================================================
def plot_ucmerced_golf_examples(num_examples=8, save_name="fig2_ucmerced_golf_courses.png"):
    """Plot UC Merced golf course examples."""
    print(f"\nGenerating: {save_name}")

    golf_samples = [item for item in ucmerced if item['label'] == 9]
    indices = np.linspace(0, len(golf_samples)-1, min(num_examples, len(golf_samples)), dtype=int)

    cols = 4
    rows = (len(indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3*rows))
    fig.suptitle('UC Merced Land Use Dataset - Golf Course Class\n(Positive Examples - Golf)',
                 fontsize=14, fontweight='bold', y=1.02)

    axes = axes.flatten()

    for i, idx in enumerate(indices):
        axes[i].imshow(golf_samples[idx]['image'])
        axes[i].axis('off')
        axes[i].set_title(f'UC Merced Golf {i+1}', fontsize=10)

    for j in range(len(indices), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"   Saved to {OUTPUT_DIR}/{save_name}")


# ============================================================
# FIGURE 3: CHALLENGING NEGATIVES
# ============================================================
def plot_challenging_negatives(samples_per_class=3, save_name="fig3_challenging_negatives.png"):
    """Plot challenging negative examples (visually similar to golf)."""
    print(f"\nGenerating: {save_name}")

    class_ids = list(CHALLENGING_CLASSES.keys())
    class_names = list(CHALLENGING_CLASSES.values())

    fig, axes = plt.subplots(len(class_ids), samples_per_class,
                             figsize=(3*samples_per_class, 3*len(class_ids)))
    fig.suptitle('Challenging Negatives (Visually Similar to Golf)\n'
                 'Baseball Fields, Tennis Courts, Agricultural, Residential, Rivers',
                 fontsize=14, fontweight='bold', y=1.02)

    for row, (class_id, class_name) in enumerate(zip(class_ids, class_names)):
        samples = [item for item in ucmerced if item['label'] == class_id]
        indices = np.linspace(0, len(samples)-1, samples_per_class, dtype=int)

        for col, idx in enumerate(indices):
            axes[row, col].imshow(samples[idx]['image'])
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_ylabel(class_name.replace('residential', '\nresidential'),
                                          fontsize=11, fontweight='bold', rotation=0,
                                          ha='right', va='center', labelpad=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"   Saved to {OUTPUT_DIR}/{save_name}")


# ============================================================
# FIGURE 4: EASY NEGATIVES
# ============================================================
def plot_easy_negatives(samples_per_class=3, save_name="fig4_easy_negatives.png"):
    """Plot easy negative examples (clearly different from golf)."""
    print(f"\nGenerating: {save_name}")

    class_ids = list(EASY_CLASSES.keys())
    class_names = list(EASY_CLASSES.values())

    fig, axes = plt.subplots(len(class_ids), samples_per_class,
                             figsize=(3*samples_per_class, 3*len(class_ids)))
    fig.suptitle('Easy Negatives (Clearly Different from Golf)\n'
                 'Airports, Highways, Harbors, Forests, Runways, Parking Lots',
                 fontsize=14, fontweight='bold', y=1.02)

    for row, (class_id, class_name) in enumerate(zip(class_ids, class_names)):
        samples = [item for item in ucmerced if item['label'] == class_id]
        indices = np.linspace(0, len(samples)-1, samples_per_class, dtype=int)

        for col, idx in enumerate(indices):
            axes[row, col].imshow(samples[idx]['image'])
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_ylabel(class_name, fontsize=11, fontweight='bold',
                                          rotation=0, ha='right', va='center', labelpad=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"   Saved to {OUTPUT_DIR}/{save_name}")


# ============================================================
# FIGURE 5: SIDE-BY-SIDE COMPARISON (Golf vs Challenging)
# ============================================================
def plot_golf_vs_challenging(save_name="fig5_golf_vs_challenging_comparison.png"):
    """Side-by-side comparison of golf courses vs challenging negatives."""
    print(f"\nGenerating: {save_name}")

    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    fig.suptitle('Golf Courses vs Challenging Negatives Comparison\n'
                 'Top Row: Golf Courses | Bottom Row: Visually Similar Non-Golf',
                 fontsize=14, fontweight='bold', y=1.02)

    # Top row: Golf courses (mix of Danish and UC Merced)
    golf_samples = [item for item in ucmerced if item['label'] == 9]

    # 3 Danish, 3 UC Merced golf
    for i in range(3):
        img = Image.open(danish_golf_files[i*50])  # Spread out samples
        img = img.resize((256, 256))
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        axes[0, i].set_title('Danish Golf', fontsize=10, color='green', fontweight='bold')

    for i in range(3):
        axes[0, i+3].imshow(golf_samples[i*30]['image'])
        axes[0, i+3].axis('off')
        axes[0, i+3].set_title('UC Merced Golf', fontsize=10, color='green', fontweight='bold')

    # Bottom row: Challenging negatives
    challenging_examples = [
        (2, 'Baseball'),
        (20, 'Tennis'),
        (0, 'Agricultural'),
        (12, 'Med. Residential'),
        (18, 'Sparse Residential'),
        (16, 'River'),
    ]

    for i, (class_id, label) in enumerate(challenging_examples):
        samples = [item for item in ucmerced if item['label'] == class_id]
        axes[1, i].imshow(samples[25]['image'])  # Pick middle sample
        axes[1, i].axis('off')
        axes[1, i].set_title(label, fontsize=10, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"   Saved to {OUTPUT_DIR}/{save_name}")


# ============================================================
# FIGURE 6: ALL UC MERCED CLASSES OVERVIEW
# ============================================================
def plot_all_ucmerced_classes(save_name="fig6_ucmerced_all_classes.png"):
    """Show one example from each UC Merced class."""
    print(f"\nGenerating: {save_name}")

    fig, axes = plt.subplots(3, 7, figsize=(21, 9))
    fig.suptitle('UC Merced Land Use Dataset - All 21 Classes\n'
                 'Green border = Golf | Red border = Challenging Negative | Gray = Easy Negative',
                 fontsize=14, fontweight='bold', y=1.02)

    axes = axes.flatten()

    for class_id, class_name in enumerate(UCMERCED_CLASSES):
        samples = [item for item in ucmerced if item['label'] == class_id]
        axes[class_id].imshow(samples[50]['image'])  # Middle sample
        axes[class_id].axis('off')

        # Color-code based on category
        if class_id == 9:  # Golf
            color = 'green'
            linewidth = 4
        elif class_id in CHALLENGING_CLASSES:
            color = 'red'
            linewidth = 3
        else:
            color = 'gray'
            linewidth = 2

        # Add colored border
        for spine in axes[class_id].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(linewidth)
            spine.set_visible(True)

        axes[class_id].set_title(class_name, fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"   Saved to {OUTPUT_DIR}/{save_name}")


# ============================================================
# FIGURE 7: DATA AUGMENTATION EXAMPLES
# ============================================================
def plot_augmentation_examples(save_name="fig7_augmentation_examples.png"):
    """
    Show original images alongside their augmented versions.
    Demonstrates the augmentation strategy used during training.
    """
    print(f"\nGenerating: {save_name}")

    from PIL import ImageEnhance, ImageOps
    import random

    def apply_augmentations(img):
        """Apply the same augmentations used in training."""
        augmented = []

        # Original
        augmented.append(("Original", img.copy()))

        # Horizontal flip
        flipped = ImageOps.mirror(img)
        augmented.append(("Horizontal Flip", flipped))

        # Brightness +10%
        enhancer = ImageEnhance.Brightness(img)
        bright = enhancer.enhance(1.1)
        augmented.append(("Brightness +10%", bright))

        # Brightness -10%
        dark = enhancer.enhance(0.9)
        augmented.append(("Brightness -10%", dark))

        # Contrast +10%
        enhancer = ImageEnhance.Contrast(img)
        high_contrast = enhancer.enhance(1.1)
        augmented.append(("Contrast +10%", high_contrast))

        # Combined: Flip + Brightness + Contrast
        combined = ImageOps.mirror(img)
        combined = ImageEnhance.Brightness(combined).enhance(1.05)
        combined = ImageEnhance.Contrast(combined).enhance(1.05)
        augmented.append(("Combined Aug.", combined))

        return augmented

    # Select 2 example images
    example_indices = [50, 200]
    num_augs = 6  # Original + 5 augmentations

    fig, axes = plt.subplots(len(example_indices), num_augs, figsize=(18, 6))
    fig.suptitle('Data Augmentation Strategy\n'
                 'Augmentations applied to 25% of training images (synchronized for image+mask)',
                 fontsize=14, fontweight='bold', y=1.02)

    for row, idx in enumerate(example_indices):
        img = Image.open(danish_golf_files[idx])
        img = img.resize((256, 256))  # Resize for display

        augmented = apply_augmentations(img)

        for col, (aug_name, aug_img) in enumerate(augmented):
            axes[row, col].imshow(aug_img)
            axes[row, col].axis('off')
            if row == 0:
                axes[row, col].set_title(aug_name, fontsize=10, fontweight='bold')

    # Add row labels
    axes[0, 0].set_ylabel('Sample 1', fontsize=11, fontweight='bold', rotation=0,
                          ha='right', va='center', labelpad=10)
    axes[1, 0].set_ylabel('Sample 2', fontsize=11, fontweight='bold', rotation=0,
                          ha='right', va='center', labelpad=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"   Saved to {OUTPUT_DIR}/{save_name}")


# ============================================================
# FIGURE 8: SEGMENTATION AUGMENTATION (Image + Mask)
# ============================================================
def plot_segmentation_augmentation(save_name="fig8_segmentation_augmentation.png"):
    """
    Show synchronized augmentation of image and segmentation mask.
    Critical: geometric transforms must be applied identically to both.
    """
    print(f"\nGenerating: {save_name}")

    from PIL import ImageOps

    # Load image and corresponding mask
    MASKS_DIR = os.path.join(golf_dataset_path, '3. class masks')

    # Find an image with good class variety
    img_idx = 5
    img_filename = os.listdir(IMAGES_DIR)[img_idx]
    mask_filename = img_filename.replace('.jpg', '.png')

    img = Image.open(os.path.join(IMAGES_DIR, img_filename))
    mask = Image.open(os.path.join(MASKS_DIR, mask_filename))

    # Resize for display
    display_size = (416, 256)
    img = img.resize(display_size)
    mask = mask.resize(display_size, Image.NEAREST)  # NEAREST preserves label values

    # Create figure: 2 rows (original, flipped) x 2 cols (image, mask)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Synchronized Augmentation: Image and Segmentation Mask\n'
                 'Geometric transforms (flip, rotation) must be applied identically to preserve alignment',
                 fontsize=14, fontweight='bold', y=1.02)

    # Original
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(mask)
    axes[0, 1].set_title('Original Mask', fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')

    # Horizontal flip (synchronized)
    img_flipped = ImageOps.mirror(img)
    mask_flipped = ImageOps.mirror(mask)

    axes[0, 2].imshow(img_flipped)
    axes[0, 2].set_title('Flipped Image', fontsize=11, fontweight='bold')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(mask_flipped)
    axes[0, 3].set_title('Flipped Mask', fontsize=11, fontweight='bold')
    axes[0, 3].axis('off')

    # Brightness (only on image)
    from PIL import ImageEnhance
    img_bright = ImageEnhance.Brightness(img).enhance(1.15)

    axes[1, 0].imshow(img_bright)
    axes[1, 0].set_title('Brightness +15%', fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(mask)
    axes[1, 1].set_title('Mask (unchanged)', fontsize=11, fontweight='bold')
    axes[1, 1].axis('off')

    # Combined
    img_combined = ImageOps.mirror(img)
    img_combined = ImageEnhance.Brightness(img_combined).enhance(1.1)
    img_combined = ImageEnhance.Contrast(img_combined).enhance(1.1)

    axes[1, 2].imshow(img_combined)
    axes[1, 2].set_title('Combined (Flip+B+C)', fontsize=11, fontweight='bold')
    axes[1, 2].axis('off')

    axes[1, 3].imshow(mask_flipped)
    axes[1, 3].set_title('Mask (flip only)', fontsize=11, fontweight='bold')
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, save_name), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"   Saved to {OUTPUT_DIR}/{save_name}")


# ============================================================
# FIGURE 9: INDIVIDUAL HIGH-RES SAMPLES (for picking)
# ============================================================
def save_individual_samples():
    """Save individual high-res samples for manual selection."""
    print("\nSaving individual samples for manual selection...")

    individual_dir = os.path.join(OUTPUT_DIR, "individual_samples")
    os.makedirs(individual_dir, exist_ok=True)

    # Danish Golf - save 20 samples
    danish_dir = os.path.join(individual_dir, "danish_golf")
    os.makedirs(danish_dir, exist_ok=True)
    indices = np.linspace(0, len(danish_golf_files)-1, 20, dtype=int)
    for i, idx in enumerate(indices):
        img = Image.open(danish_golf_files[idx])
        img.save(os.path.join(danish_dir, f"danish_golf_{i+1:02d}.png"))
    print(f"   Saved 20 Danish golf samples to {danish_dir}/")

    # UC Merced Golf - save 10 samples
    ucmerced_golf_dir = os.path.join(individual_dir, "ucmerced_golf")
    os.makedirs(ucmerced_golf_dir, exist_ok=True)
    golf_samples = [item for item in ucmerced if item['label'] == 9]
    for i in range(min(10, len(golf_samples))):
        golf_samples[i*10]['image'].save(os.path.join(ucmerced_golf_dir, f"ucmerced_golf_{i+1:02d}.png"))
    print(f"   Saved 10 UC Merced golf samples to {ucmerced_golf_dir}/")

    # Challenging negatives - save 5 per class
    for class_id, class_name in CHALLENGING_CLASSES.items():
        class_dir = os.path.join(individual_dir, f"challenging_{class_name}")
        os.makedirs(class_dir, exist_ok=True)
        samples = [item for item in ucmerced if item['label'] == class_id]
        for i in range(min(5, len(samples))):
            samples[i*20]['image'].save(os.path.join(class_dir, f"{class_name}_{i+1:02d}.png"))
        print(f"   Saved 5 {class_name} samples to {class_dir}/")

    # Easy negatives - save 5 per class
    for class_id, class_name in EASY_CLASSES.items():
        class_dir = os.path.join(individual_dir, f"easy_{class_name}")
        os.makedirs(class_dir, exist_ok=True)
        samples = [item for item in ucmerced if item['label'] == class_id]
        for i in range(min(5, len(samples))):
            samples[i*20]['image'].save(os.path.join(class_dir, f"{class_name}_{i+1:02d}.png"))
        print(f"   Saved 5 {class_name} samples to {class_dir}/")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GENERATING PAPER FIGURES")
    print("=" * 60)

    # Generate all figures
    plot_danish_golf_examples(num_examples=12)
    plot_ucmerced_golf_examples(num_examples=8)
    plot_challenging_negatives(samples_per_class=3)
    plot_easy_negatives(samples_per_class=3)
    plot_golf_vs_challenging()
    plot_all_ucmerced_classes()
    plot_augmentation_examples()
    plot_segmentation_augmentation()
    save_individual_samples()

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nAll figures saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  - fig1_danish_golf_courses.png       (Danish golf examples)")
    print("  - fig2_ucmerced_golf_courses.png     (UC Merced golf examples)")
    print("  - fig3_challenging_negatives.png     (Challenging negative classes)")
    print("  - fig4_easy_negatives.png            (Easy negative classes)")
    print("  - fig5_golf_vs_challenging_comparison.png (Side-by-side)")
    print("  - fig6_ucmerced_all_classes.png      (All 21 UC Merced classes)")
    print("  - fig7_augmentation_examples.png     (Data augmentation demo)")
    print("  - fig8_segmentation_augmentation.png (Synchronized image+mask aug)")
    print("  - individual_samples/                (High-res samples to pick from)")

# %%
