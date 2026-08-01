# ============================================================
# train_transfer.py - MODEL 2: Klasifikasi Kematangan (3 Kelas)
# Cascade Classification Stage 2 — MobileNetV2 Transfer Learning
# ============================================================
#
# TUJUAN
# ------
# Melatih classifier 3 kelas kematangan buah melon (Mentah, Setengah
# Matang, Matang) menggunakan MobileNetV2 transfer learning dengan
# fine-tuning. Model ini HANYA menerima citra yang telah lolos Model 1
# (validator objek, train_validator.py) -- dataset dan arsitektur
# TIDAK bercampur dengan Model 1.
#
# ARSITEKTUR
# ----------
# MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
#   → GlobalAveragePooling2D
#   → Dropout(0.3)
#   → Dense(128, relu)
#   → Dense(3, softmax)
#
# TRAINING 2 TAHAP
# -----------------
# Tahap 1 (Feature Extraction): seluruh layer MobileNetV2 dibekukan.
# Tahap 2 (Fine-Tuning): 30 layer terakhir MobileNetV2 dibuka kembali,
#          compile ulang Adam learning_rate=1e-5, lanjutkan training.
#
# DATASET
# -------
#   dataset/
#   ├── mentah/
#   ├── setengah_matang/
#   └── matang/
#
# OUTPUT
# ------
# model_melon.keras, model_melon.h5, class_names.json, model_config.json,
# accuracy_plot.png, loss_plot.png, confusion_matrix.png,
# classification_report.txt
#
# CARA PAKAI
# ----------
#   python train_transfer.py
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
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Config ────────────────────────────────────────────────────
DATASET_DIR    = os.path.join(os.path.dirname(__file__), 'dataset')
IMG_SIZE       = 224
BATCH_SIZE     = 32
SEED           = 42

PHASE1_EPOCHS    = 15   # feature extraction (base dibekukan)
PHASE2_EPOCHS    = 15   # fine-tuning (30 layer terakhir dibuka)
FINE_TUNE_LAYERS = 30   # jumlah layer terakhir MobileNetV2 yang dibuka

print("=" * 60)
print("  MODEL 2: Klasifikasi Kematangan Buah Melon (3 Kelas)")
print("  Cascade Classification - Stage 2")
print("  Base Model: MobileNetV2 (pretrained ImageNet)")
print("=" * 60)
print(f"  TensorFlow  : {tf.__version__}")
print(f"  Dataset Dir : {DATASET_DIR}")
print(f"  Image Size  : {IMG_SIZE}x{IMG_SIZE}")
print(f"  Batch Size  : {BATCH_SIZE}")
print(f"  Total Epoch : {PHASE1_EPOCHS + PHASE2_EPOCHS} "
      f"({PHASE1_EPOCHS} freeze + {PHASE2_EPOCHS} fine-tune)")
print("=" * 60)

# ── Validasi folder dataset ───────────────────────────────────
required_folders = ['mentah', 'setengah_matang', 'matang']
for folder in required_folders:
    path = os.path.join(DATASET_DIR, folder)
    if not os.path.exists(path):
        print(f"  [WARNING] Folder tidak ditemukan: {path}")
        os.makedirs(path, exist_ok=True)

# ── Data Augmentation ──────────────────────────────────────────
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
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

class_indices = train_gen.class_indices
class_names   = [k for k, v in sorted(class_indices.items(), key=lambda x: x[1])]
num_classes   = len(class_names)

print(f"\n  Kelas terdeteksi   : {class_names}")
print(f"  Training samples   : {train_gen.samples}")
print(f"  Validation samples : {val_gen.samples}")

with open('class_names.json', 'w') as f:
    json.dump(class_names, f, indent=2)
print(f"  class_names.json disimpan: {class_names}")

# ── Class Weight (jika data antar kelas kematangan timpang) ────
unique, counts = np.unique(train_gen.classes, return_counts=True)
class_counts = {class_names[u]: c for u, c in zip(unique, counts)}
print("\n  Distribusi data training per kelas:")
for cls, c in class_counts.items():
    print(f"    {cls:18s}: {c:4d} gambar")

imbalance_ratio = max(class_counts.values()) / min(class_counts.values())
print(f"  Rasio imbalance: {imbalance_ratio:.2f}x")

if imbalance_ratio > 1.5:
    class_weight_arr = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weight_arr)}
    print("  Class weight diterapkan:")
    for i, w in class_weight_dict.items():
        print(f"    {class_names[i]:18s}: {w:.4f}")
else:
    class_weight_dict = None
    print("  Data cukup seimbang, class_weight tidak diperlukan.")

# ── Bangun Model: MobileNetV2 (base) + Head ────────────────────
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False   # Tahap 1: bekukan seluruh base

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
], name='KlasifikasiKematangan_MobileNetV2')

model.summary()
print(f"\n  Total layer base MobileNetV2: {len(base_model.layers)}")

# ══════════════════════════════════════════════════════════════
# TAHAP 1: Feature Extraction
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"  TAHAP 1: Feature Extraction ({PHASE1_EPOCHS} epoch, base dibekukan)")
print("=" * 60)

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cb_phase1 = [
    callbacks.ModelCheckpoint(
        'model_melon.h5', monitor='val_accuracy',
        save_best_only=True, verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )
]

history1 = model.fit(
    train_gen,
    epochs=PHASE1_EPOCHS,
    validation_data=val_gen,
    callbacks=cb_phase1,
    class_weight=class_weight_dict,
    verbose=1
)

# ══════════════════════════════════════════════════════════════
# TAHAP 2: Fine-Tuning (buka 30 layer terakhir)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"  TAHAP 2: Fine-Tuning ({PHASE2_EPOCHS} epoch, "
      f"{FINE_TUNE_LAYERS} layer terakhir dibuka)")
print("=" * 60)

base_model.trainable = True
freeze_until = len(base_model.layers) - FINE_TUNE_LAYERS
for layer in base_model.layers[:freeze_until]:
    layer.trainable = False

trainable_count = sum(1 for l in base_model.layers if l.trainable)
print(f"  Layer dibuka untuk fine-tuning: {trainable_count} / {len(base_model.layers)}")

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cb_phase2 = [
    callbacks.ModelCheckpoint(
        'model_melon.h5', monitor='val_accuracy',
        save_best_only=True, verbose=1
    ),
    callbacks.EarlyStopping(
        monitor='val_loss', patience=7,
        restore_best_weights=True, verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1
    )
]

history2 = model.fit(
    train_gen,
    epochs=PHASE1_EPOCHS + PHASE2_EPOCHS,
    initial_epoch=history1.epoch[-1] + 1,
    validation_data=val_gen,
    callbacks=cb_phase2,
    class_weight=class_weight_dict,
    verbose=1
)

# ── Load Best Weights & Simpan Model (dua format) ──────────────
model.load_weights('model_melon.h5')
model.save('model_melon.keras')
print("\n  model_melon.h5 dan model_melon.keras disimpan.")

# ── Simpan Konfigurasi Model ────────────────────────────────────
with open('model_config.json', 'w') as f:
    json.dump({
        "architecture": "MobileNetV2",
        "type": "classifier",
        "classes": num_classes,
        "preprocessing": "mobilenet_v2",
        "img_size": IMG_SIZE
    }, f, indent=2)
print("  model_config.json disimpan.")

# ── Gabungkan History Kedua Tahap ───────────────────────────────
def combine_history(h1, h2):
    combined = {}
    for key in h1.history:
        combined[key] = h1.history[key] + h2.history.get(key, [])
    return combined

full_history = combine_history(history1, history2)
switch_epoch = len(history1.history['accuracy'])

# ── Accuracy Plot ─────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(full_history['accuracy'],     label='Train Accuracy', color='#2ecc71', linewidth=2)
plt.plot(full_history['val_accuracy'], label='Val Accuracy',   color='#27ae60', linewidth=2, linestyle='--')
plt.axvline(x=switch_epoch - 0.5, color='gray', linestyle=':', linewidth=1.5, label='Mulai Fine-Tuning')
plt.title('Akurasi Model - Klasifikasi Kematangan Melon', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('accuracy_plot.png', dpi=150)
plt.close()
print("\n  accuracy_plot.png disimpan.")

# ── Loss Plot ─────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(full_history['loss'],     label='Train Loss', color='#e74c3c', linewidth=2)
plt.plot(full_history['val_loss'], label='Val Loss',   color='#c0392b', linewidth=2, linestyle='--')
plt.axvline(x=switch_epoch - 0.5, color='gray', linestyle=':', linewidth=1.5, label='Mulai Fine-Tuning')
plt.title('Loss Model - Klasifikasi Kematangan Melon', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('loss_plot.png', dpi=150)
plt.close()
print("  loss_plot.png disimpan.")

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
plt.title('Confusion Matrix - Klasifikasi Kematangan Melon', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()
print("  confusion_matrix.png disimpan.")

# ── Classification Report (cetak + simpan ke file) ─────────────
report_str = classification_report(y_true, y_pred, target_names=class_names)
print("\n" + "=" * 60)
print("  CLASSIFICATION REPORT - KLASIFIKASI KEMATANGAN")
print("=" * 60)
print(report_str)

with open('classification_report.txt', 'w') as f:
    f.write("CLASSIFICATION REPORT - KLASIFIKASI KEMATANGAN (MODEL 2)\n")
    f.write("=" * 60 + "\n")
    f.write(f"Kelas: {class_names}\n\n")
    f.write(report_str)
print("  classification_report.txt disimpan.")

val_loss, val_acc = model.evaluate(val_gen, verbose=0)
print(f"\n  Final Val Accuracy : {val_acc * 100:.2f}%")
print(f"  Final Val Loss     : {val_loss:.4f}")
print("\n  Training Model 2 (Klasifikasi Kematangan) selesai!")
print("=" * 60)

# ── Backup otomatis dengan prefix 'model2_' ──────────────────────
# CATATAN: nama file plot sama dengan output train_validator.py (Model 1).
# Salinan berprefix disimpan otomatis agar hasil Model 1 tidak hilang
# tertimpa, dan sebaliknya hasil Model 2 ini juga aman jika suatu saat
# train_validator.py dijalankan ulang setelah script ini.
import shutil
for fname in ['accuracy_plot.png', 'loss_plot.png', 'confusion_matrix.png', 'classification_report.txt']:
    if os.path.exists(fname):
        shutil.copy2(fname, f'model2_{fname}')
print("\n  [BACKUP] Salinan berprefix 'model2_' juga disimpan:")
print("  model2_accuracy_plot.png, model2_loss_plot.png,")
print("  model2_confusion_matrix.png, model2_classification_report.txt")
print("  → File ini AMAN dari tertimpa saat menjalankan train_validator.py.")
print("=" * 60)
