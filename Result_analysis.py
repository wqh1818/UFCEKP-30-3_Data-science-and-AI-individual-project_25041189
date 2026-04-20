# ── Load all 8 history files ──────────────────────────────────────────────────
base_path = '/content/drive/MyDrive/individual projects/'

files = {
    'ResNet50 Original':       'resnet50_original_history.csv',
    'ResNet50 Segmented 100%': 'resnet50_segmented100_history.csv',
    'ResNet50 Segmented 75%':  'resnet50_segmented75_history.csv',
    'ResNet50 Segmented 50%':  'resnet50_segmented50_history.csv',
    'DenseNet121 Original':       'densenet121_original_history.csv',
    'DenseNet121 Segmented 100%': 'densenet121_segmented100_history.csv',
    'DenseNet121 Segmented 75%':  'densenet121_segmented75_history.csv',
    'DenseNet121 Segmented 50%':  'densenet121_segmented50_history.csv',
}

histories = {}
for label, filename in files.items():
    histories[label] = pd.read_csv(base_path + filename)

# ── Colour & style settings ───────────────────────────────────────────────────
styles = {
    'ResNet50 Original':          ('blue',   '-'),
    'ResNet50 Segmented 100%':    ('royalblue',  '--'),
    'ResNet50 Segmented 75%':     ('cornflowerblue', '-.'),
    'ResNet50 Segmented 50%':     ('lightskyblue',   ':'),
    'DenseNet121 Original':          ('darkorange',  '-'),
    'DenseNet121 Segmented 100%':    ('orange',      '--'),
    'DenseNet121 Segmented 75%':     ('sandybrown',  '-.'),
    'DenseNet121 Segmented 50%':     ('peachpuff',   ':'),
}

# ── Graph 1: Validation Accuracy ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

for label, df in histories.items():
    color, linestyle = styles[label]
    ax.plot(df['epoch'], df['val_accuracy'], label=label, color=color, linestyle=linestyle, linewidth=2)

ax.set_title('Validation Accuracy over Epochs — All Models', fontsize=14, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Accuracy')
ax.legend(loc='lower right', fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(base_path + 'val_accuracy_comparison.png', dpi=150)
plt.show()

# ── Graph 2: Validation Loss ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

for label, df in histories.items():
    color, linestyle = styles[label]
    ax.plot(df['epoch'], df['val_loss'], label=label, color=color, linestyle=linestyle, linewidth=2)

ax.set_title('Validation Loss over Epochs — All Models', fontsize=14, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Loss')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(base_path + 'val_loss_comparison.png', dpi=150)
plt.show()
