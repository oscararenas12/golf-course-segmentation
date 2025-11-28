# Replication Plan: Aalborg Paper Approach

## Goal
Replicate the semantic segmentation and course rating system from:
> Mortensen et al. "Semantic Segmentation of Golf Courses for Course Rating Assistance" (IEEE ICMEW 2023)

## Current Status vs. Paper

| Aspect | Our Current | Paper's Approach | Status |
|--------|-------------|------------------|--------|
| **Dataset** | Danish Golf Courses (Kaggle) | Same dataset | ✅ Match |
| **Classes** | 6 (Background, Green, Fairway, Tee, Bunker, Water) | Same 6 classes | ✅ Match |
| **Architecture** | U-Net | U-Net | ✅ Match |
| **Framework** | TensorFlow/Keras | PyTorch | ⚠️ Different (OK) |
| **Image Size** | 224×224 | 832×512 | ❌ Need to change |
| **Encoder** | MobileNetV2 | timm-gernet-l (best) | ❌ Can try others |
| **Batch Size** | 32 | 16 | ⚠️ Different |
| **Learning Rate** | 1e-4 | 1e-4 | ✅ Match |
| **Split** | 70/20/10 | 70/30 + separate test | ⚠️ Different |
| **Results** | Not evaluated yet | 69.6% mIoU, 78% Sensitivity | ⏳ Pending |
| **Course Rating System** | Not implemented | Implemented | ❌ Not started |

---

## Phase 1: Improve Segmentation (Match Paper Performance)

### Step 1.1: Increase Image Resolution
**Current**: 256×256
**Target**: 832×512 (paper's size)

**Why**: Small features (tees, bunkers) need higher resolution

**Changes**:
```python
# In unet_golf_segmentation_tf.py
IMAGE_SIZE = (512, 832)  # Height, Width (H×W)
```

**Expected Impact**: Better tee/bunker detection

---

### Step 1.2: Try Different Encoders

Paper tested 6 encoders. Best was **timm-gernet-l** (model 3).

**TensorFlow equivalents to try**:

```python
# Option 1: ResNet50 (similar to paper's model 6)
base = keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(512, 832, 3)
)

# Option 2: EfficientNetB4 (similar to paper's model 1)
base = keras.applications.EfficientNetB4(
    include_top=False,
    weights='imagenet',
    input_shape=(512, 832, 3)
)

# Option 3: ResNet101 (deeper)
base = keras.applications.ResNet101(...)
```

**Test each and compare**:
- IoU per class
- Training time
- GPU memory usage

---

### Step 1.3: Use Paper's Train/Val Split

**Current**: 70/20/10 (train/val/test)
**Paper**: 70/30 (train/val) + separate 108 test images

**Why**: More validation data = better model selection

**Changes**:
```python
# Keep all 1,123 images for train/val
train_size = int(0.70 * 1123)  # 786 images
val_size = 1123 - train_size    # 337 images

# Use separate testing script for 108 test images
```

---

### Step 1.4: Adjust Batch Size

**Current**: 32
**Paper**: 16

**Why**: Larger images (832×512) need smaller batch size

```python
BATCH_SIZE = 16  # or 8 if GPU memory issues
```

---

### Step 1.5: Evaluate Using Paper's Metrics

**Add these metrics**:

```python
# IoU (Intersection over Union)
def iou_metric(y_true, y_pred, class_id):
    intersection = ((y_true == class_id) & (y_pred == class_id)).sum()
    union = ((y_true == class_id) | (y_pred == class_id)).sum()
    return intersection / union if union > 0 else 0

# Sensitivity (Recall)
# Already have: keras.metrics.Recall()

# PPV (Precision)
# Already have: keras.metrics.Precision()

# Report per-class:
classes = ['Background', 'Green', 'Fairway', 'Tee', 'Bunker', 'Water']
for i, class_name in enumerate(classes):
    iou = calculate_iou(y_true, y_pred, class_id=i)
    print(f"{class_name}: IoU={iou:.1%}")
```

**Target Results** (from paper):
```
Fairway: 76.6% IoU
Green:   80.2% IoU
Tee:     48.0% IoU (difficult)
Bunker:  80.0% IoU
Water:   63.4% IoU
Mean:    69.6% IoU
```

---

## Phase 2: Implement Course Rating System

### Step 2.1: Measure Green Size

```python
def measure_green(green_mask):
    """
    Calculate green length and width.

    Length: Longest distance between any two points on green
    Width: Perpendicular distance to length line
    """
    from scipy.spatial.distance import pdist, squareform
    from skimage.measure import find_contours

    # Get green contour
    contours = find_contours(green_mask, level=0.5)
    points = contours[0]  # Assume largest contour

    # Find two points with maximum distance (length)
    distances = squareform(pdist(points))
    max_idx = np.unravel_index(distances.argmax(), distances.shape)
    p1, p2 = points[max_idx[0]], points[max_idx[1]]
    length = distances[max_idx]

    # Find perpendicular width
    length_vector = p2 - p1
    perpendicular = np.array([-length_vector[1], length_vector[0]])
    perpendicular = perpendicular / np.linalg.norm(perpendicular)

    # Project all points onto perpendicular
    projections = points @ perpendicular
    width = projections.max() - projections.min()

    return length, width
```

**Expected accuracy**: 2.7% error (length), 17.7% error (width)

---

### Step 2.2: Calculate Hole Length

```python
# USGA stroke distances (meters)
STROKE_DISTANCES = {
    'scratch_male': 228,
    'scratch_female': 192,
    'bogey_male': 183,
    'bogey_female': 137
}

def calculate_hole_length(tee_pos, green_mask, fairway_mask, player_type):
    """
    Calculate total hole length from tee to green.

    Simulates player strokes along fairway until reaching green.
    """
    stroke_length = STROKE_DISTANCES[player_type]

    total_distance = 0
    current_pos = tee_pos
    landing_points = [tee_pos]

    # Find landing points along fairway
    max_strokes = 10  # Safety limit
    for _ in range(max_strokes):
        # Draw circle with stroke radius
        circle_mask = create_circle_mask(current_pos, stroke_length, fairway_mask.shape)

        # Find intersection with fairway
        intersection = circle_mask & fairway_mask

        if not intersection.any():
            # Can't find fairway landing - go to green
            break

        # Next landing point = center of fairway intersection
        y_coords, x_coords = np.where(intersection)
        next_pos = np.array([y_coords.mean(), x_coords.mean()])

        # Check if we're past the green
        green_y, green_x = np.where(green_mask)
        green_center = np.array([green_y.mean(), green_x.mean()])

        if np.linalg.norm(next_pos - green_center) < stroke_length / 2:
            # Close enough to green
            break

        landing_points.append(next_pos)
        total_distance += stroke_length
        current_pos = next_pos

    # Add final distance to green center
    final_distance = np.linalg.norm(current_pos - green_center)
    total_distance += final_distance

    # Convert pixels to meters (depends on image scale)
    # Scale 1:1000 means 1 pixel = X meters (calculate from metadata)
    total_distance_meters = pixel_to_meters(total_distance)

    return total_distance_meters, landing_points
```

**Expected accuracy**: 3.3% error (male), 4.2% error (female)

---

### Step 2.3: Calculate Fairway Width

```python
def calculate_fairway_width(tee_pos, landing_point, fairway_mask, stroke_length):
    """
    Measure fairway width perpendicular to line of play.

    Width is measured at the landing zone between carry and total stroke distance.
    """
    # Draw circle at tee with stroke radius
    circle_mask = create_circle_mask(tee_pos, stroke_length, fairway_mask.shape)

    # Find intersections with fairway
    intersection = circle_mask & fairway_mask

    # Get boundary points
    from skimage.measure import find_contours
    contours = find_contours(intersection, level=0.5)

    if len(contours) == 0:
        return 0

    # Find outermost points on fairway
    points = contours[0]

    # Direction from tee to landing point
    direction = landing_point - tee_pos
    direction = direction / np.linalg.norm(direction)

    # Perpendicular direction
    perpendicular = np.array([-direction[1], direction[0]])

    # Project intersection points onto perpendicular
    projections = (points - landing_point) @ perpendicular

    # Width = distance between min and max projections
    width = projections.max() - projections.min()

    return pixel_to_meters(width)
```

---

### Step 2.4: Distance to Obstacles

```python
def distance_to_nearest_obstacle(reference_point, bunker_mask, water_mask, max_radius=47):
    """
    Find distance from reference point (tee or landing zone) to nearest obstacle.

    Only considers obstacles within max_radius meters.
    """
    # Combine obstacle masks
    obstacle_mask = bunker_mask | water_mask

    # Get obstacle pixels
    obstacle_y, obstacle_x = np.where(obstacle_mask)
    obstacle_points = np.column_stack([obstacle_y, obstacle_x])

    if len(obstacle_points) == 0:
        return None, None

    # Calculate distances
    distances = np.linalg.norm(obstacle_points - reference_point, axis=1)

    # Convert to meters
    distances_meters = pixel_to_meters(distances)

    # Filter by max radius
    within_radius = distances_meters < max_radius

    if not within_radius.any():
        return None, None

    # Find nearest
    nearest_idx = distances[within_radius].argmin()
    nearest_distance = distances_meters[within_radius][nearest_idx]
    nearest_point = obstacle_points[within_radius][nearest_idx]

    # Determine obstacle type
    if bunker_mask[nearest_point[0], nearest_point[1]]:
        obstacle_type = 'bunker'
    else:
        obstacle_type = 'water'

    return nearest_distance, obstacle_type
```

---

### Step 2.5: Build GUI (Like Paper's Figure 4)

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def visualize_course_rating(image, segmentation, measurements, player_type):
    """
    Create visualization showing all measurements for a given player type.

    Similar to paper's Figure 4.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: Original image with overlay
    axes[0].imshow(image)
    axes[0].imshow(segmentation, alpha=0.5)
    axes[0].set_title(f'Segmentation - {player_type}')

    # Right: Measurements
    axes[1].imshow(image)

    # Draw tee position
    tee = measurements['tee_position']
    axes[1].plot(tee[1], tee[0], 'ro', markersize=10, label='Tee')

    # Draw landing points
    for i, landing in enumerate(measurements['landing_points']):
        axes[1].plot(landing[1], landing[0], 'yo', markersize=8)
        if i > 0:
            prev = measurements['landing_points'][i-1]
            axes[1].plot([prev[1], landing[1]], [prev[0], landing[0]], 'y-', linewidth=2)

    # Draw stroke circle
    stroke_radius_pixels = meters_to_pixels(measurements['stroke_distance'])
    circle = Circle(tee[::-1], stroke_radius_pixels, fill=False, color='cyan', linewidth=2)
    axes[1].add_patch(circle)

    # Draw fairway width line
    fw_start = measurements['fairway_width_start']
    fw_end = measurements['fairway_width_end']
    axes[1].plot([fw_start[1], fw_end[1]], [fw_start[0], fw_end[0]], 'g-', linewidth=3, label='Fairway Width')

    # Draw green size
    green_length_line = measurements['green_length_line']
    axes[1].plot([green_length_line[0][1], green_length_line[1][1]],
                 [green_length_line[0][0], green_length_line[1][0]],
                 'b-', linewidth=3, label='Green Length')

    # Add text annotations
    axes[1].text(10, 30, f"Hole Length: {measurements['hole_length']:.1f}m",
                 color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.7))
    axes[1].text(10, 60, f"Fairway Width: {measurements['fairway_width']:.1f}m",
                 color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.7))
    axes[1].text(10, 90, f"Green: {measurements['green_length']:.1f}m × {measurements['green_width']:.1f}m",
                 color='white', fontsize=12, bbox=dict(facecolor='black', alpha=0.7))

    axes[1].legend(loc='upper right')
    axes[1].set_title(f'Measurements - {player_type}')

    plt.tight_layout()
    return fig
```

---

## Phase 3: Validation

### Compare with Manual Measurements

The paper validated against 108 golf holes with known measurements.

```python
def validate_measurements(predictions, ground_truth):
    """
    Compare automated measurements with manual ground truth.

    Metrics:
    - Mean Absolute Error (MAE)
    - Mean Percentage Error (MPE)
    """
    errors = {
        'green_length': [],
        'green_width': [],
        'hole_length_male': [],
        'hole_length_female': []
    }

    for pred, gt in zip(predictions, ground_truth):
        # Green measurements
        errors['green_length'].append(abs(pred['green_length'] - gt['green_length']))
        errors['green_width'].append(abs(pred['green_width'] - gt['green_width']))

        # Hole lengths
        errors['hole_length_male'].append(abs(pred['hole_length_male'] - gt['hole_length_male']))
        errors['hole_length_female'].append(abs(pred['hole_length_female'] - gt['hole_length_female']))

    # Calculate metrics
    results = {}
    for metric, error_list in errors.items():
        mae = np.mean(error_list)
        mpe = np.mean(np.array(error_list) / np.array([gt[metric] for gt in ground_truth])) * 100
        results[metric] = {'MAE': mae, 'MPE': mpe}

    return results
```

**Paper's Results**:
```
Green Length:  MAE = 0.8m  (2.7%)
Green Width:   MAE = 4.1m  (17.7%)
Hole Length (Male):   MAE = 11.2m (3.3%)
Hole Length (Female): MAE = 12.6m (4.2%)
```

---

## Implementation Order

1. ✅ **Segmentation improvements** (Phase 1) - Do this first
   - Larger images (832×512)
   - Try ResNet50/EfficientNet encoders
   - Evaluate with IoU/Sensitivity/PPV per class

2. ⏳ **Course rating system** (Phase 2) - After segmentation works well
   - Implement measurement functions
   - Build visualization GUI
   - Validate against known golf holes

3. 🎯 **Target Performance**:
   - Segmentation: 70% mIoU (match paper)
   - Measurements: <5% error (match paper)

---

## Files to Create/Modify

### New Files:
- `notebooks/unet_large_resolution.py` - U-Net with 832×512 images
- `notebooks/evaluate_segmentation.py` - Calculate IoU/Sensitivity/PPV per class
- `course_rating/measure_green.py` - Green size calculation
- `course_rating/measure_hole_length.py` - Hole length calculation
- `course_rating/measure_fairway.py` - Fairway width calculation
- `course_rating/visualize_measurements.py` - GUI like paper's Figure 4
- `course_rating/validate_measurements.py` - Compare with ground truth

### Modified Files:
- `notebooks/unet_golf_segmentation_tf.py` - Change IMAGE_SIZE
- `.gitignore` - Add course_rating outputs

---

## Next Steps

1. Start with Phase 1, Step 1.1 (increase image resolution)
2. Train and evaluate
3. If performance improves, continue with other encoders
4. Once segmentation matches paper (~70% mIoU), move to Phase 2

