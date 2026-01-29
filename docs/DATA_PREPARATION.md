# Data Preparation Guide

## 📁 Directory Structure

```
data/
├── train/
│   ├── images/           # Training images
│   └── labels/           # YOLO format labels
├── val/
│   ├── images/           # Validation images
│   └── labels/           # YOLO format labels
├── test/
│   ├── images/           # Test images
│   └── labels/           # Test labels
└── classification/       # For classifier training
    ├── scratch/
    ├── crack/
    ├── dent/
    ├── missing_component/
    └── contamination/
```

## 📝 Label Format (YOLO)

Each image requires a corresponding `.txt` file with the same name:

```
<class_id> <x_center> <y_center> <width> <height>
```

Example (`image001.txt`):
```
0 0.5 0.5 0.1 0.2
2 0.3 0.7 0.15 0.1
```

## 🏷️ Class IDs

| ID | Class Name |
|----|------------|
| 0 | scratch |
| 1 | crack |
| 2 | dent |
| 3 | missing_component |
| 4 | contamination |

## 🔄 Synthetic Data Generation

```python
from src.data.synthetic_data import SyntheticDefectGenerator

generator = SyntheticDefectGenerator(output_dir='data/synthetic')

# Generate training data
generator.generate_dataset(
    num_images=1000,
    defects_per_image=(1, 5)
)
```

## 📊 Data Augmentation

The system includes built-in augmentation:

- **Light**: Basic flips, rotations
- **Medium**: + Brightness, contrast, noise
- **Heavy**: + Cutout, mosaic, mixup

Configure in `configs/default_config.yaml`:

```yaml
data:
  augmentation_strength: medium
```

## ✅ Data Validation

Before training, validate your dataset:

```python
from src.data.dataset import DefectDetectionDataset

dataset = DefectDetectionDataset(
    images_dir='data/train/images',
    labels_dir='data/train/labels'
)

# Check loading
sample = dataset[0]
print(f"Image shape: {sample['image'].shape}")
print(f"Bboxes: {sample['bboxes']}")
```
