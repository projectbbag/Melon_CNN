# ============================================================
# train.py - CNN Training Script
# Deteksi Kematangan Buah Melon (mendukung 3 ATAU 4 kelas)
# ============================================================
#
# Jalankan: python train.py
# Output  : model_melon.h5, class_names.json, model_config.json,
#           accuracy_plot.png, loss_plot.png, confusion_matrix.png
#
# Struktur dataset (jumlah kelas otomatis terdeteksi dari folder):
#   dataset/
#   ├── mentah/             ← melon mentah
#   ├── setengah_matang/    ← melon setengah matang
#   ├── matang/             ← melon matang
#   └── bukan_melon/        ← (opsional) bukan buah melon
#
# CLASS WEIGHT
# ------------
# Jika jumlah gambar antar kelas timpang (rasio > 1.5x), script ini
# otomatis menghitung class_weight agar kelas dengan data lebih sedikit
# tidak "tenggelam" saat training -- loss dari kelas minoritas diberi
# bobot lebih besar agar kontribusinya seimbang dengan kelas mayoritas.
# ============================================================

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Config ────────────────────────────────────────────────────
DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 30
SEED        = 42

print("=" * 60)
print("  CNN Training - Deteksi Kematangan Buah Melon")
print("=" * 60)
print(f"  TensorFlow  : {tf.__version__}")
print(f"  Dataset Dir : {DATASET_DIR}")
print(f"  Image Size  : {IMG_SIZE}x{IMG_SIZE}")
print(f"  Batch Size  : {BATCH_SIZE}")
print(f"  Epochs      : {EPOCHS}")
print("=" * 60)

# ── Validasi folder dataset (deteksi otomatis, tidak hardcode jumlah kelas) ──
found_folders = [f for f in os.listdir(DATASET_DIR)
                  if os.path.isdir(os.path.join(DATASET_DIR, f))] if os.path.isdir(DATASET_DIR) else []
print(f"\n  Folder kelas ditemukan: {found_folders}")
if not found_folders:
    raise SystemExit("  [ERROR] Tidak ada folder kelas di dataset/. Buat folder per kelas terlebih dahulu.")

# ── Data Augmentation (Train) ─────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

# ── Generators ────────────────────────────────────────────────
train_gen = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    seed=SEED,
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=SEED,
    shuffle=False
)

# ── Save Class Names ──────────────────────────────────────────
class_indices = train_gen.class_indices
class_names   = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]

print(f"\n  Kelas terdeteksi  : {class_names}")
print(f"  Jumlah kelas      : {len(class_names)}")
print(f"  Training samples  : {train_gen.samples}")
print(f"  Validation samples: {val_gen.samples}")

with open('class_names.json', 'w') as f:
    json.dump(class_names, f, indent=2)
print(f"\n  class_names.json disimpan: {class_names}")

# Catat jenis preprocessing yang dipakai model ini, agar predict.py
# tahu cara menormalisasi citra dengan benar saat inferensi.
with open('model_config.json', 'w') as f:
    json.dump({
        "architecture": "custom_cnn",
        "preprocessing": "rescale",
        "img_size": IMG_SIZE
    }, f, indent=2)
print("  model_config.json disimpan (architecture: custom_cnn, preprocessing: rescale)")

# ── Hitung Jumlah Gambar per Kelas & Class Weight ──────────────
print("\n  Distribusi data training per kelas:")
unique, counts = np.unique(train_gen.classes, return_counts=True)
class_counts = {class_names[u]: c for u, c in zip(unique, counts)}
for cls, c in class_counts.items():
    print(f"    {cls:18s}: {c:4d} gambar")

imbalance_ratio = max(class_counts.values()) / min(class_counts.values())
print(f"\n  Rasio imbalance (terbanyak/tersedikit): {imbalance_ratio:.2f}x")

if imbalance_ratio > 1.5:
    print("  → Data timpang (>1.5x), menghitung class_weight untuk menyeimbangkan loss...")
    class_weight_arr = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weight_arr)}
    print("  Class weight yang dipakai:")
    for i, w in class_weight_dict.items():
        print(f"    {class_names[i]:18s}: {w:.4f}")
else:
    class_weight_dict = None
    print("  → Data cukup seimbang, class_weight tidak diperlukan.")

# ── CNN Architecture ──────────────────────────────────────────
num_classes = len(class_names)

model = models.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                  input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2, 2),

    # Block 2
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),

    # Block 3
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),

    # Head
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
], name='MelonCNN')

model.summary()

# ── Compile ───────────────────────────────────────────────────
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ── Callbacks ─────────────────────────────────────────────────
cb_list = [
    callbacks.ModelCheckpoint(
        'model_melon.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]

# ── Training ──────────────────────────────────────────────────
print("\n  Memulai training...\n")
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=cb_list,
    class_weight=class_weight_dict,
    verbose=1
)

# ── Load Best Model ───────────────────────────────────────────
model.load_weights('model_melon.h5')

# ── Accuracy Plot ─────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'],     label='Train Accuracy', color='#2ecc71', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Val Accuracy',   color='#27ae60', linewidth=2, linestyle='--')
plt.title('Akurasi Model CNN - Deteksi Kematangan Melon', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('accuracy_plot.png', dpi=150)
plt.close()
print("\n  Grafik akurasi disimpan: accuracy_plot.png")

# ── Loss Plot ─────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'],     label='Train Loss', color='#e74c3c', linewidth=2)
plt.plot(history.history['val_loss'], label='Val Loss',   color='#c0392b', linewidth=2, linestyle='--')
plt.title('Loss Model CNN - Deteksi Kematangan Melon', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('loss_plot.png', dpi=150)
plt.close()
print("  Grafik loss disimpan: loss_plot.png")

# ── Confusion Matrix ──────────────────────────────────────────
val_gen.reset()
y_pred_probs = model.predict(val_gen, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_gen.classes[:len(y_pred)]

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Greens',
    xticklabels=class_names, yticklabels=class_names,
    linewidths=0.5, linecolor='white'
)
plt.title('Confusion Matrix - Deteksi Kematangan Melon', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()
print("  Confusion matrix disimpan: confusion_matrix.png")

# ── Final Report ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("  CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(y_true, y_pred, target_names=class_names))

val_loss, val_acc = model.evaluate(val_gen, verbose=0)
print(f"\n  Kelas              : {class_names}")
print(f"  Final Val Accuracy : {val_acc * 100:.2f}%")
print(f"  Final Val Loss     : {val_loss:.4f}")
print("\n  Model disimpan: model_melon.h5")
print("  Training selesai!")
print("=" * 60)
