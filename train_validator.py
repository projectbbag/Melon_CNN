# ============================================================
# train_validator.py - MODEL 1: Validator Binary (Melon vs Bukan Melon)
# Cascade Classification Stage 1
# ============================================================
#
# TUJUAN
# ------
# Melatih classifier BINARY yang menentukan apakah citra yang diunggah
# merupakan buah melon atau bukan, menggunakan MobileNetV2 transfer
# learning dengan fine-tuning. Model ini SEPENUHNYA TERPISAH dari
# Model 2 (klasifikasi kematangan 3 kelas di train_transfer.py) --
# tidak ada arsitektur, dataset, maupun bobot yang saling tercampur.
#
# PRASYARAT
# ---------
# Jalankan prepare_validator_dataset.py terlebih dahulu untuk membentuk:
#   dataset_validator/
#   ├── melon/          (gabungan mentah + setengah_matang + matang)
#   └── bukan_melon/
#
# ARSITEKTUR
# ----------
# MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
#   → GlobalAveragePooling2D
#   → Dropout(0.3)
#   → Dense(128, relu)
#   → Dense(2, softmax)
#
# TRAINING 2 TAHAP
# -----------------
# Tahap 1 (Feature Extraction): seluruh layer MobileNetV2 dibekukan,
#          hanya head baru yang belajar.
# Tahap 2 (Fine-Tuning): 30 layer terakhir MobileNetV2 dibuka kembali,
#          compile ulang dengan Adam learning_rate=1e-5, lanjutkan training.
#
# OUTPUT
# ------
# validator_model.keras, validator_model.h5, validator_class_names.json,
# validator_model_config.json, accuracy_plot.png, loss_plot.png,
# confusion_matrix.png, classification_report.txt
#
# CARA PAKAI
# ----------
#   python prepare_validator_dataset.py   (sekali, atau tiap dataset berubah)
#   python train_validator.py
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
BASE_DIR       = os.path.dirname(__file__)
DATASET_DIR    = os.path.join(BASE_DIR, 'dataset_validator')
IMG_SIZE       = 224
BATCH_SIZE     = 32
SEED           = 42

PHASE1_EPOCHS  = 15   # feature extraction (base dibekukan)
PHASE2_EPOCHS  = 15   # fine-tuning (30 layer terakhir dibuka)
FINE_TUNE_LAYERS = 30 # jumlah layer terakhir MobileNetV2 yang dibuka

print("=" * 60)
print("  MODEL 1: Validator Binary (Melon vs Bukan Melon)")
print("  Cascade Classification - Stage 1")
print("=" * 60)
print(f"  TensorFlow  : {tf.__version__}")
print(f"  Dataset Dir : {DATASET_DIR}")
print(f"  Image Size  : {IMG_SIZE}x{IMG_SIZE}")
print(f"  Batch Size  : {BATCH_SIZE}")
print(f"  Total Epoch : {PHASE1_EPOCHS + PHASE2_EPOCHS} "
      f"({PHASE1_EPOCHS} freeze + {PHASE2_EPOCHS} fine-tune)")
print("=" * 60)

if not os.path.isdir(DATASET_DIR):
    raise SystemExit(
        f"\n  [ERROR] Folder {DATASET_DIR} tidak ditemukan.\n"
        f"  Jalankan terlebih dahulu: python prepare_validator_dataset.py\n"
    )

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

with open('validator_class_names.json', 'w') as f:
    json.dump(class_names, f, indent=2)
print(f"  validator_class_names.json disimpan: {class_names}")

# ── Class Weight (menangani proporsi melon vs bukan_melon) ─────
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
], name='ValidatorObjek_MobileNetV2')

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
        'validator_model.h5', monitor='val_accuracy',
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
        'validator_model.h5', monitor='val_accuracy',
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
model.load_weights('validator_model.h5')
model.save('validator_model.keras')
print("\n  validator_model.h5 dan validator_model.keras disimpan.")

# ── Simpan Konfigurasi Model ────────────────────────────────────
with open('validator_model_config.json', 'w') as f:
    json.dump({
        "architecture": "MobileNetV2",
        "type": "validator",
        "classes": num_classes,
        "preprocessing": "mobilenet_v2",
        "img_size": IMG_SIZE
    }, f, indent=2)
print("  validator_model_config.json disimpan.")

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
plt.plot(full_history['accuracy'],     label='Train Accuracy', color='#8b5cf6', linewidth=2)
plt.plot(full_history['val_accuracy'], label='Val Accuracy',   color='#6d28d9', linewidth=2, linestyle='--')
plt.axvline(x=switch_epoch - 0.5, color='gray', linestyle=':', linewidth=1.5, label='Mulai Fine-Tuning')
plt.title('Akurasi Validator Objek (Melon vs Bukan Melon)', fontsize=14, fontweight='bold')
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
plt.title('Loss Validator Objek (Melon vs Bukan Melon)', fontsize=14, fontweight='bold')
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
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Purples',
    xticklabels=class_names, yticklabels=class_names,
    linewidths=0.5, linecolor='white'
)
plt.title('Confusion Matrix - Validator Objek', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()
print("  confusion_matrix.png disimpan.")

# ── Classification Report (cetak + simpan ke file) ─────────────
report_str = classification_report(y_true, y_pred, target_names=class_names)
print("\n" + "=" * 60)
print("  CLASSIFICATION REPORT - VALIDATOR OBJEK")
print("=" * 60)
print(report_str)

with open('classification_report.txt', 'w') as f:
    f.write("CLASSIFICATION REPORT - VALIDATOR OBJEK (MODEL 1)\n")
    f.write("=" * 60 + "\n")
    f.write(f"Kelas: {class_names}\n\n")
    f.write(report_str)
print("  classification_report.txt disimpan.")

val_loss, val_acc = model.evaluate(val_gen, verbose=0)
print(f"\n  Final Val Accuracy : {val_acc * 100:.2f}%")
print(f"  Final Val Loss     : {val_loss:.4f}")
print("\n  Training Model 1 (Validator Objek) selesai!")
print("  Model CNN kematangan (Model 2) TIDAK terpengaruh oleh script ini.")
print("=" * 60)

# ── Backup otomatis dengan prefix 'validator_' ──────────────────
# CATATAN: accuracy_plot.png, loss_plot.png, confusion_matrix.png, dan
# classification_report.txt memakai nama yang SAMA dengan output
# train_transfer.py (Model 2). Jika kedua script dijalankan berurutan,
# file dari Model 2 akan MENIMPA file Model 1. Untuk mencegah kehilangan
# hasil (dibutuhkan keduanya untuk BAB IV), salinan berprefix disimpan
# otomatis di bawah ini -- gunakan file BERPREFIX sebagai referensi utama.
import shutil
for fname in ['accuracy_plot.png', 'loss_plot.png', 'confusion_matrix.png', 'classification_report.txt']:
    if os.path.exists(fname):
        shutil.copy2(fname, f'validator_{fname}')
print("\n  [BACKUP] Salinan berprefix 'validator_' juga disimpan:")
print("  validator_accuracy_plot.png, validator_loss_plot.png,")
print("  validator_confusion_matrix.png, validator_classification_report.txt")
print("  → File ini AMAN dari tertimpa saat menjalankan train_transfer.py.")
print("=" * 60)
