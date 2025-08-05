import os
import shutil
import random
from pathlib import Path

# Paths
SOURCE_DIR = 'Bangladeshi Hospitals Dataset'           # Folder containing the 10 class folders
TARGET_DIR = 'processed_data'     # New structured dataset

# Ratios
TRAIN_SPLIT = 0.7
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1

# Set seed for reproducibility
random.seed(42)

# Make destination directories
for split in ['train', 'test', 'val']:
    for category in os.listdir(SOURCE_DIR):
        Path(os.path.join(TARGET_DIR, split, category)).mkdir(parents=True, exist_ok=True)

# Distribute files
for category in os.listdir(SOURCE_DIR):
    category_path = os.path.join(SOURCE_DIR, category)
    images = os.listdir(category_path)
    random.shuffle(images)
    n = len(images)
    
    train_cutoff = int(TRAIN_SPLIT * n)
    test_cutoff = int((TRAIN_SPLIT + TEST_SPLIT) * n)

    for i, img in enumerate(images):
        src_path = os.path.join(category_path, img)
        if i < train_cutoff:
            dst_path = os.path.join(TARGET_DIR, 'train', category, img)
        elif i < test_cutoff:
            dst_path = os.path.join(TARGET_DIR, 'test', category, img)
        else:
            dst_path = os.path.join(TARGET_DIR, 'val', category, img)
        shutil.copyfile(src_path, dst_path)
