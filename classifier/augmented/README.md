# Golf Course Classifier - Augmented Version

This is an improved version of the binary golf course classifier with data augmentation and stratified data splitting.

## Key Improvements Over Baseline

### 1. **Data Augmentation (50/50 Mix)**
Training uses a balanced mix of original and augmented images:
- **50% of batches**: Original images (no transformation)
- **50% of batches**: Augmented images with:
  - Horizontal flips (50% chance)
  - Random rotation (±18 degrees) - gentle rotation
  - Random zoom (90-110%)
  - Random brightness (±10%) - subtle adjustment
  - Random contrast (90-110%) - subtle adjustment

**Why**: Model learns from clean originals AND variations, preventing over-distortion
**Benefit**: Training accuracy closer to validation (healthy learning curve)

### 2. **Stratified Data Split**
Instead of random splitting, we strategically distribute images:

**Challenging Negatives** (look similar to golf courses):
- Classes: baseballdiamond, tenniscourt, agricultural, residential areas, river, chaparral
- Split: 50% train, **30% validation**, 20% test
- More in validation for realistic evaluation

**Easy Negatives** (clearly different):
- Classes: airplane, buildings, freeway, harbor, parking lots, etc.
- Split: 70% train, 20% validation, 10% test

**Positive Examples**:
- Danish golf: **80% train**, 10% val, 10% test (more for primary learning)
- UC Merced golf: 60% train, 20% val, 20% test

### 3. **Benefits**

✅ **Better Generalization**: Augmentation exposes model to more variations
✅ **Realistic Validation**: Harder negatives give honest performance estimate
✅ **More Danish Data in Training**: Primary dataset gets priority

## Training

Run from the `classifier/augmented/` directory:

```bash
cd classifier/augmented
python train_classifier.py
```

## Outputs

- `best_golf_classifier.keras` - Best model during training
- `final_golf_classifier.keras` - Final trained model
- `training_history_augmented.png` - Training curves
- `sample_predictions_augmented.png` - Example predictions
- `logs/` - TensorBoard logs

## Expected Results

**Compared to baseline**:
- Similar or slightly lower validation accuracy (validation set is harder now!)
- Better test accuracy (better generalization from augmentation)
- More robust to real-world variations

**Typical metrics**:
- Validation accuracy: 90-95% (with challenging negatives)
- Test accuracy: 93-97% (balanced evaluation)
- Training takes ~15 epochs (more than baseline due to augmentation)

## Viewing Training Progress

```bash
tensorboard --logdir=./logs
```

Open http://localhost:6006 to see:
- Training/validation curves
- Learning rate changes
- Model architecture
