# Datasets

Place datasets here using the `torchvision.datasets.ImageFolder` layout:

```text
datasets/
├── lungs_ultrasound/
│   ├── train/
│   │   ├── class_1/
│   │   └── class_2/
│   └── test/
│       ├── class_1/
│       └── class_2/
├── brain_tumor/
│   ├── train/
│   │   ├── class_1/
│   │   └── class_2/
│   └── test/
│       ├── class_1/
│       └── class_2/
└── fundus_diabetic_retinopathy/
    ├── train/
    │   ├── class_1/
    │   └── class_2/
    └── test/
        ├── class_1/
        └── class_2/
```

Each class folder should contain the image files for that class.
