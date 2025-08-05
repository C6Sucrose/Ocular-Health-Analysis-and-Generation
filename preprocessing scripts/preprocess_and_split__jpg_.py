import os
from pathlib import Path
from PIL import Image
from torchvision import transforms

# === Configuration ===
SOURCE_DIR = "processed_data"  # where your 'train', 'val', 'test' folders are
TARGET_DIR = "preprocessed_images"  # destination to save preprocessed versions
IMAGE_SIZE = (224, 224)

# === Transformations ===
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === Save pipeline (reverse normalize for saving as JPG) ===
def reverse_normalize(tensor):
    return tensor * 0.5 + 0.5  # Reverse normalization to [0,1] range

def process_and_save_image(img_path, save_path):
    try:
        image = Image.open(img_path).convert("RGB")
        image = transform(image)
        image = reverse_normalize(image).clamp(0, 1)  # Ensure valid range
        image = transforms.ToPILImage()(image)
        image.save(save_path)
    except Exception as e:
        print(f"❌ Failed to process {img_path}: {e}")

# === Main loop ===
for split in ['train', 'val', 'test']:
    split_dir = os.path.join(SOURCE_DIR, split)
    for class_name in os.listdir(split_dir):
        class_dir = os.path.join(split_dir, class_name)
        target_class_dir = os.path.join(TARGET_DIR, split, class_name)
        Path(target_class_dir).mkdir(parents=True, exist_ok=True)

        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                continue
            src_path = os.path.join(class_dir, img_name)
            dst_path = os.path.join(target_class_dir, img_name)
            process_and_save_image(src_path, dst_path)
