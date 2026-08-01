# ============================================================
# predict.py - Modul Inferensi Cascade Classification
# Model 1 (Validator Objek) + Model 2 (Klasifikasi Kematangan)
# ============================================================
#
# ALUR
# ----
# 1. Load validator_model.h5 + validator_class_names.json +
#    validator_model_config.json
# 2. Prediksi gambar dengan Model 1 (validator)
# 3. Jika hasil = bukan_melon → langsung return status "bukan_melon",
#    Model 2 TIDAK dijalankan sama sekali.
# 4. Jika hasil = melon → load model_melon.h5 + class_names.json +
#    model_config.json, lanjutkan prediksi dengan Model 2.
# 5. Return status "success" beserta kelas, confidence, dan probabilitas.
#
# Kedua model dimuat secara lazy (hanya sekali, saat pertama dibutuhkan)
# dan disimpan di variabel modul-level agar tidak perlu load ulang
# setiap request.
# ============================================================

import os

# Hemat RAM TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import json
import numpy as np
import cv2
from PIL import Image

BASE_DIR = os.path.dirname(__file__)

# ── State Model 1: Validator Objek (lazy-loaded) ────────────────
_validator_model    = None
_validator_classes  = None
_validator_config   = None

# ── State Model 2: Klasifikasi Kematangan (lazy-loaded) ─────────
_classifier_model    = None
_classifier_classes  = None
_classifier_config   = None


# ════════════════════════════════════════════════════════════════
# LOADING MODEL 1: VALIDATOR OBJEK
# ════════════════════════════════════════════════════════════════
def _load_validator():
    global _validator_model, _validator_classes, _validator_config
    if _validator_model is not None:
        return

    import tensorflow as tf
    
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    model_path  = os.path.join(BASE_DIR, 'validator_model.h5')
    classes_path = os.path.join(BASE_DIR, 'validator_class_names.json')
    config_path = os.path.join(BASE_DIR, 'validator_model_config.json')

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "validator_model.h5 tidak ditemukan. Jalankan "
            "prepare_validator_dataset.py lalu train_validator.py terlebih dahulu."
        )
    if not os.path.exists(classes_path):
        raise FileNotFoundError("validator_class_names.json tidak ditemukan.")
    if not os.path.exists(config_path):
        raise FileNotFoundError("validator_model_config.json tidak ditemukan.")

    _validator_model = tf.keras.models.load_model(model_path)

    with open(classes_path, 'r') as f:
        _validator_classes = json.load(f)

    with open(config_path, 'r') as f:
        _validator_config = json.load(f)

    print(f"[predict.py] Model 1 (Validator) dimuat. Kelas: {_validator_classes}")


# ════════════════════════════════════════════════════════════════
# LOADING MODEL 2: KLASIFIKASI KEMATANGAN
# ════════════════════════════════════════════════════════════════
def _load_classifier():
    global _classifier_model, _classifier_classes, _classifier_config
    if _classifier_model is not None:
        return

    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)


    model_path   = os.path.join(BASE_DIR, 'model_melon.h5')
    classes_path = os.path.join(BASE_DIR, 'class_names.json')
    config_path  = os.path.join(BASE_DIR, 'model_config.json')

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "model_melon.h5 tidak ditemukan. Jalankan train_transfer.py "
            "(atau train.py) terlebih dahulu."
        )
    if not os.path.exists(classes_path):
        raise FileNotFoundError("class_names.json tidak ditemukan.")
    if not os.path.exists(config_path):
        raise FileNotFoundError("model_config.json tidak ditemukan.")

    _classifier_model = tf.keras.models.load_model(model_path)

    with open(classes_path, 'r') as f:
        _classifier_classes = json.load(f)

    with open(config_path, 'r') as f:
        _classifier_config = json.load(f)

    print(f"[predict.py] Model 2 (Klasifikasi Kematangan) dimuat. "
          f"Kelas: {_classifier_classes}")


# ════════════════════════════════════════════════════════════════
# PREPROCESSING CITRA
# ════════════════════════════════════════════════════════════════
def _preprocess_image(image_path: str, img_size: int, preprocessing: str) -> np.ndarray:
    """
    Baca, resize, dan normalisasi citra sesuai jenis preprocessing
    yang dibutuhkan model ('mobilenet_v2' atau 'rescale').
    """
    img = cv2.imread(image_path)
    if img is None:
        img = np.array(Image.open(image_path).convert('RGB'))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float32')

    if preprocessing == 'mobilenet_v2':
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        img = preprocess_input(img)
    else:
        img = img / 255.0

    return np.expand_dims(img, axis=0)


def get_display_name(raw_label: str) -> str:
    """
    Konversi nama kelas dari folder dataset (huruf kecil, underscore)
    ke nama tampilan berkapital rapi.
    Contoh: 'setengah_matang' → 'Setengah Matang'
    """
    return raw_label.replace('_', ' ').title()


# ════════════════════════════════════════════════════════════════
# FUNGSI UTAMA: PREDIKSI CASCADE (Model 1 → Model 2)
# ════════════════════════════════════════════════════════════════
def predict_image(image_path: str) -> dict:
    """
    Menjalankan pipeline cascade classification lengkap.

    Returns
    -------
    dict dengan struktur:

    Jika citra BUKAN melon (Model 1 menolak):
        {
            "status": "bukan_melon",
            "validator_confidence": float   # confidence Model 1
        }

    Jika citra ADALAH melon (Model 1 lolos, Model 2 berjalan):
        {
            "status": "success",
            "kelas": str,                   # label tampilan, mis. "Matang"
            "confidence": float,            # confidence Model 2 (0-1)
            "probabilitas": dict            # {nama_kelas: probabilitas}
        }
    """
    # ── TAHAP 1: Validasi Objek ──
    _load_validator()

    x1 = _preprocess_image(
        image_path,
        img_size=_validator_config.get('img_size', 224),
        preprocessing=_validator_config.get('preprocessing', 'mobilenet_v2')
    )
    val_probs = _validator_model.predict(x1, verbose=0)[0]
    val_idx   = int(np.argmax(val_probs))
    val_label = _validator_classes[val_idx]
    val_conf  = float(val_probs[val_idx])

    if val_label != 'melon':
        return {
            "status": "bukan_melon",
            "validator_confidence": round(val_conf, 4)
        }

    # ── TAHAP 2: Klasifikasi Kematangan ──
    _load_classifier()

    x2 = _preprocess_image(
        image_path,
        img_size=_classifier_config.get('img_size', 224),
        preprocessing=_classifier_config.get('preprocessing', 'mobilenet_v2')
    )
    cls_probs = _classifier_model.predict(x2, verbose=0)[0]
    cls_idx   = int(np.argmax(cls_probs))
    raw_label = _classifier_classes[cls_idx]
    confidence = float(cls_probs[cls_idx])

    display_label = get_display_name(raw_label)
    probabilitas = {
        get_display_name(_classifier_classes[i]): float(cls_probs[i])
        for i in range(len(_classifier_classes))
    }

    return {
        "status": "success",
        "kelas": display_label,
        "confidence": round(confidence, 4),
        "probabilitas": {k: round(v, 4) for k, v in probabilitas.items()}
    }
