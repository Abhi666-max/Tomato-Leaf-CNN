import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n" + "="*50)
print(f"🔥 Using Device: {device}")
if torch.cuda.is_available():
    print(f"🎮 GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ WARNING: GPU not detected. Make sure PyTorch CUDA is installed.")
print("="*50 + "\n")

# Dataset Path
DATA_DIR = 'TomatoLeaf-Dataset'
BATCH_SIZE = 32
IMG_SIZE = 224

# Data Augmentation (matches PPT)
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
full_dataset = datasets.ImageFolder(DATA_DIR)

# 80-20 Split
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
# Use a fixed generator for reproducible splits
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

# We need to apply different transforms to train and val, so we use a wrapper class
class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            # The ImageFolder returns (PIL Image, label). We need to get the raw image back if transform is applied at wrapper level.
            # Actually, subset[index] already applies the underlying dataset's transform.
            pass
        return x, y
        
    def __len__(self):
        return len(self.subset)

# Better way: replace the transform directly (hacky but works for ImageFolder)
train_dataset.dataset = copy.copy(full_dataset)
train_dataset.dataset.transform = train_transforms
val_dataset.dataset = copy.copy(full_dataset)
val_dataset.dataset.transform = val_transforms

# Using num_workers=0 to avoid Windows multiprocessing issues for now
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

class_names = full_dataset.classes
NUM_CLASSES = len(class_names)
print(f"✅ Found {len(full_dataset)} images across {NUM_CLASSES} classes.")

# Build Model EXACTLY matching PPT Slide 8: 
# MobileNetV2 -> GlobalAvgPool -> Dropout(0.3) -> Dense(128, ReLU) -> Dense(10, Softmax)
print("🧠 Building MobileNetV2 Architecture (from your presentation)...")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# Freeze Backbone (Phase 1)
for param in model.features.parameters():
    param.requires_grad = False

# Replace Classifier Head
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=False),
    nn.Linear(model.last_channel, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, NUM_CLASSES)
)

model = model.to(device)

def train_model(model, criterion, optimizer, scheduler, num_epochs=10, phase_name="Phase"):
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs} [{phase_name}]')
        print('-' * 20)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader
                
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'val':
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(epoch_loss)
                else:
                    scheduler.step()
                    
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), 'best_tomato_model.pth')
                    print(f"🌟 Saved new best model (Accuracy: {best_acc:.4f})")
                    
    print(f'\n🏆 {phase_name} Best Val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

# Setup Loss
criterion = nn.CrossEntropyLoss()

# =======================
# PHASE 1: Train Head Only
# =======================
print("\n🚀 STARTING PHASE 1: Training Custom Head Only (Backbone Frozen)")
optimizer_1 = optim.Adam(model.classifier.parameters(), lr=1e-3)
scheduler_1 = ReduceLROnPlateau(optimizer_1, mode='min', factor=0.5, patience=2)

model = train_model(model, criterion, optimizer_1, scheduler_1, num_epochs=10, phase_name="Phase 1")

# =======================
# PHASE 2: Fine-Tuning
# =======================
print("\n🚀 STARTING PHASE 2: Fine-Tuning Entire Model")
# Unfreeze backbone
for param in model.parameters():
    param.requires_grad = True

optimizer_2 = optim.Adam(model.parameters(), lr=1e-5)
scheduler_2 = ReduceLROnPlateau(optimizer_2, mode='min', factor=0.5, patience=2)

model = train_model(model, criterion, optimizer_2, scheduler_2, num_epochs=10, phase_name="Phase 2")

print("\n🎉 Training Complete! Model saved as 'best_tomato_model.pth'")
