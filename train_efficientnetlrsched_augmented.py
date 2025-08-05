import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# === CONFIGURATION ===
BATCH_SIZE = 32
NUM_CLASSES = 10
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_ROOT = "preprocessed_images"

# === TRANSFORMATIONS ===
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === LOAD DATA ===
train_dataset = datasets.ImageFolder(os.path.join(DATASET_ROOT, 'train'), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(DATASET_ROOT, 'val'), transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print(f"🔍 Found {len(train_dataset)} training samples")
print(f"🔍 Found {len(val_dataset)} validation samples")

# === MODEL SETUP ===
model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.2,       # More aggressive reduction
    patience=1,       # React quickly to no improvement
    min_lr=5e-6,      # Prevent going too low
    verbose=True      # Print when LR is reduced
)
# === TRAINING FUNCTIONS ===
def train_epoch(model, loader):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), 100 * correct / len(loader.dataset)

def validate_epoch(model, loader):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader), 100 * correct / len(loader.dataset)

# === TRAINING LOOP ===
best_val_acc = 0
best_val_loss = 2
best_epoch = -1
model_path = "best_efficientnet_b0.pth"
patience = 3
no_improvement = 0

print("🚀 Starting training...\n")
for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader)
    val_loss, val_acc = validate_epoch(model, val_loader)
    scheduler.step(val_loss)

    print(f"📘 Epoch {epoch+1}/{EPOCHS} | "
          f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
          f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_acc > best_val_acc and val_loss < best_val_loss:
        best_val_acc = val_acc
        best_val_loss = val_loss
        best_epoch = epoch + 1
        no_improvement = 0
        torch.save(model.state_dict(), model_path)
        print(f"💾 Best model saved at Epoch {best_epoch} "
              f"(Val Acc: {best_val_acc:.2f}%, Val Loss: {best_val_loss:.4f})")
    else:
        no_improvement += 1
        print(f"⚠️ No improvement for {no_improvement} epoch(s)")

    if no_improvement >= patience:
        print("🛑 Early stopping triggered.")
        break

print(f"\n✅ Training complete. Best model saved from epoch {best_epoch} at: {model_path}")
