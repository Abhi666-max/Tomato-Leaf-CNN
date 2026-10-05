import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = 'TomatoLeaf-Dataset'
MODEL_PATH = 'best_tomato_model.pth'

CLASS_DISPLAY_NAMES = [
    "Bacterial Spot",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites",
    "Target Spot",
    "Yellow Leaf Curl",
    "Mosaic Virus",
    "Healthy",
]

# Load Dataset (val split only)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
full_dataset = datasets.ImageFolder(DATA_DIR, transform=val_transforms)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
_, val_dataset = random_split(full_dataset, [train_size, val_size],
                               generator=torch.Generator().manual_seed(42))
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

# Load Model
model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.last_channel, 128),
    nn.ReLU(inplace=True),
    nn.Linear(128, 10)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# Evaluate
correct = [0] * 10
total   = [0] * 10
print("Running evaluation...")
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        _, preds = torch.max(model(images), 1)
        for label, pred in zip(labels, preds):
            total[label.item()]   += 1
            correct[label.item()] += (pred == label).item()

accuracies = [correct[i] / total[i] * 100 if total[i] > 0 else 0 for i in range(10)]
avg_acc = sum(accuracies) / len(accuracies)

for name, acc in zip(CLASS_DISPLAY_NAMES, accuracies):
    print(f"  {name:<25}: {acc:.2f}%")
print(f"\n  Average Accuracy: {avg_acc:.2f}%")

# =============================================
# Single clean horizontal bar chart
# =============================================
BG_COLOR = '#0f172a'
BG_CARD  = '#1e293b'

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_CARD)

# Sort by accuracy descending
sorted_pairs = sorted(zip(accuracies, CLASS_DISPLAY_NAMES), reverse=True)
sorted_acc, sorted_names = zip(*sorted_pairs)

y = np.arange(len(sorted_names))

# Color: green if >= avg, blue otherwise
bar_colors = ['#10b981' if a >= avg_acc else '#3b82f6' for a in sorted_acc]

bars = ax.barh(y, sorted_acc, color=bar_colors, height=0.6, zorder=3)

# Add value labels at end of bar
for bar, acc in zip(bars, sorted_acc):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f'{acc:.2f}%', va='center', ha='left',
            color='white', fontsize=11, fontweight='bold')

# Average vertical line
ax.axvline(x=avg_acc, color='#f59e0b', linestyle='--', linewidth=2, zorder=4)

ax.set_xlim([88, 102.5])
ax.set_yticks(y)
ax.set_yticklabels(sorted_names, fontsize=12, color='#e2e8f0')
ax.set_xlabel('Accuracy (%)', color='#94a3b8', fontsize=12)
ax.set_title('LeafLens - Per-Class Validation Accuracy\nMobileNetV2 · 18,160 Images · 10 Classes',
             color='white', fontsize=15, fontweight='bold', pad=18)

ax.grid(True, axis='x', color='#334155', linewidth=0.8, alpha=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#334155')
ax.spines['left'].set_color('#334155')
ax.tick_params(colors='#94a3b8')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, p: f'{val:.0f}%'))

# Legend
green_patch = mpatches.Patch(color='#10b981', label='Above Average')
blue_patch  = mpatches.Patch(color='#3b82f6', label='Below Average')
avg_line    = plt.Line2D([0], [0], color='#f59e0b', linestyle='--', linewidth=2,
                          label=f'Average Accuracy: {avg_acc:.2f}%')
ax.legend(handles=[green_patch, blue_patch, avg_line],
          facecolor='#0f172a', edgecolor='#334155', fontsize=10, loc='lower right')

# Overall accuracy text
ax.text(0.99, 0.02,
        f'Overall Val Accuracy: {sum(correct)/sum(total)*100:.2f}%',
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=11, color='#f59e0b', fontweight='bold')

plt.tight_layout()
plt.savefig('per_class_accuracy.png', dpi=200, bbox_inches='tight',
            facecolor=BG_COLOR, edgecolor='none')
print("\nSaved: per_class_accuracy.png")
plt.show()
