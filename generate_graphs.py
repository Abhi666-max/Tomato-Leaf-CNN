import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# =============================================
# Training Data from actual training run
# =============================================
phase1_train_acc = [0.7965, 0.8653, 0.8729, 0.8853, 0.8937, 0.8964, 0.9001, 0.9032, 0.9057, 0.9103]
phase1_val_acc   = [0.8802, 0.9042, 0.9111, 0.9188, 0.9240, 0.9273, 0.9243, 0.9284, 0.9221, 0.9394]

phase2_train_acc = [0.9391, 0.9567, 0.9692, 0.9742, 0.9749, 0.9807, 0.9846, 0.9856, 0.9886, 0.9889]
phase2_val_acc   = [0.9637, 0.9730, 0.9818, 0.9860, 0.9832, 0.9882, 0.9876, 0.9904, 0.9904, 0.9906]

phase1_train_loss = [0.6405, 0.3916, 0.3679, 0.3265, 0.3164, 0.2995, 0.2935, 0.2778, 0.2748, 0.2565]
phase1_val_loss   = [0.3683, 0.2807, 0.2637, 0.2401, 0.2198, 0.2155, 0.2297, 0.2052, 0.2278, 0.1829]

phase2_train_loss = [0.1784, 0.1223, 0.0917, 0.0767, 0.0670, 0.0552, 0.0451, 0.0440, 0.0350, 0.0332]
phase2_val_loss   = [0.1087, 0.0832, 0.0570, 0.0480, 0.0502, 0.0400, 0.0377, 0.0327, 0.0269, 0.0261]

# Combine Phase 1 and Phase 2 for full view
all_train_acc = phase1_train_acc + phase2_train_acc
all_val_acc   = phase1_val_acc   + phase2_val_acc
all_train_loss = phase1_train_loss + phase2_train_loss
all_val_loss   = phase1_val_loss   + phase2_val_loss
epochs = list(range(1, 21))

# =============================================
# Style Setup - Dark Professional Theme
# =============================================
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('dark_background')

COLOR_TRAIN = '#10b981'   # Green
COLOR_VAL   = '#3b82f6'   # Blue
COLOR_GRID  = '#1f2937'
BG_COLOR    = '#111827'
BG_CARD     = '#1f2937'

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor(BG_COLOR)
fig.suptitle('LeafLens - MobileNetV2 Training Results\nTomato Leaf Disease Classification', 
             fontsize=16, color='white', fontweight='bold', y=1.02)

def style_ax(ax, title, ylabel):
    ax.set_facecolor(BG_CARD)
    ax.set_title(title, color='white', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Epoch', color='#9ca3af', fontsize=11)
    ax.set_ylabel(ylabel, color='#9ca3af', fontsize=11)
    ax.tick_params(colors='#9ca3af')
    ax.grid(True, color=COLOR_GRID, linewidth=0.8, alpha=0.5)
    ax.spines['bottom'].set_color('#374151')
    ax.spines['left'].set_color('#374151')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Phase divider
    ax.axvline(x=10.5, color='#f59e0b', linestyle='--', alpha=0.6, linewidth=1.5, label='Phase 1→2')
    ax.text(10.5, ax.get_ylim()[0] + 0.01, ' Fine-tune starts', color='#f59e0b', fontsize=8, va='bottom')

# ---- Plot 1: Accuracy ----
ax1 = axes[0]
ax1.plot(epochs, [a*100 for a in all_train_acc], color=COLOR_TRAIN, linewidth=2.5, marker='o', markersize=5, label='Training Accuracy')
ax1.plot(epochs, [a*100 for a in all_val_acc],   color=COLOR_VAL,   linewidth=2.5, marker='s', markersize=5, label='Validation Accuracy', linestyle='--')

# Highlight final accuracy
ax1.annotate(f'99.06%', xy=(20, 99.06), xytext=(17, 96),
             arrowprops=dict(arrowstyle='->', color='#f59e0b'), 
             color='#f59e0b', fontweight='bold', fontsize=11)

ax1.set_ylim([75, 101])
ax1.set_xlim([0.5, 20.5])
ax1.set_xticks(epochs)
style_ax(ax1, 'Model Accuracy Over 20 Epochs', 'Accuracy (%)')
ax1.legend(loc='lower right', facecolor='#111827', edgecolor='#374151', fontsize=10)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))

# ---- Plot 2: Loss ----
ax2 = axes[1]
ax2.plot(epochs, all_train_loss, color=COLOR_TRAIN, linewidth=2.5, marker='o', markersize=5, label='Training Loss')
ax2.plot(epochs, all_val_loss,   color=COLOR_VAL,   linewidth=2.5, marker='s', markersize=5, label='Validation Loss', linestyle='--')

ax2.annotate(f'Val Loss: 0.0261', xy=(20, 0.0261), xytext=(16, 0.12),
             arrowprops=dict(arrowstyle='->', color='#f59e0b'), 
             color='#f59e0b', fontweight='bold', fontsize=10)

ax2.set_ylim([-0.02, 0.70])
ax2.set_xlim([0.5, 20.5])
ax2.set_xticks(epochs)
style_ax(ax2, 'Model Loss Over 20 Epochs', 'Loss Value')
ax2.legend(loc='upper right', facecolor='#111827', edgecolor='#374151', fontsize=10)

plt.tight_layout(pad=3.0)
plt.savefig('training_results_graph.png', dpi=200, bbox_inches='tight', 
            facecolor=BG_COLOR, edgecolor='none')
print("✅ Graph saved as 'training_results_graph.png'")
plt.show()
print("\n📊 Final Results Summary:")
print(f"   Training Accuracy:   {all_train_acc[-1]*100:.2f}%")
print(f"   Validation Accuracy: {all_val_acc[-1]*100:.2f}%")
print(f"   Training Loss:       {all_train_loss[-1]:.4f}")
print(f"   Validation Loss:     {all_val_loss[-1]:.4f}")
print(f"   Overfitting Gap:     {abs(all_train_acc[-1]-all_val_acc[-1])*100:.2f}% (Excellent!)")
