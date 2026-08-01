# 🍈 MelonVision CNN
**Implementasi CNN untuk Mendeteksi Kematangan Buah Melon Berdasarkan Citra RGB Berbasis Web (Studi Kasus: PT. IVES)**

---

## 🏗️ Arsitektur: Cascade Classification (2 Model Terpisah)

```
Upload Gambar
    ↓
┌───────────────────────────────────────────┐
│ MODEL 1: Validator Objek (Binary)           │
│ validator_model.h5 — MobileNetV2 + fine-tune│
│ Kelas: melon / bukan_melon                  │
└───────────────────────────────────────────┘
    ↓
    ├── bukan_melon → STOP
    │   "Gambar yang diunggah bukan merupakan buah melon.
    │    Silakan unggah gambar buah melon untuk dilakukan
    │    deteksi kematangan."
    │   (Model 2 TIDAK dijalankan, tidak disimpan ke database)
    │
    └── melon →
        ┌───────────────────────────────────────────┐
        │ MODEL 2: Klasifikasi Kematangan (3 Kelas)   │
        │ model_melon.h5 — MobileNetV2 + fine-tune    │
        │ Kelas: Mentah / Setengah Matang / Matang    │
        └───────────────────────────────────────────┘
            ↓
        Simpan ke database + tampilkan hasil & confidence
```

Kedua model **sepenuhnya independen** — Model 2 (klasifikasi kematangan)
tidak pernah menerima data atau kelas "bukan_melon" dalam proses
trainingnya, sehingga fokus penelitian tetap murni pada 3 tingkat
kematangan sesuai judul skripsi.

---

## 📋 Dataset

### Model 2 — Klasifikasi Kematangan (dataset/)
| Kelas | Folder |
|---|---|
| Mentah | `dataset/mentah/` |
| Setengah Matang | `dataset/setengah_matang/` |
| Matang | `dataset/matang/` |

### Model 1 — Validator Objek (dataset_validator/, dibuat otomatis)
| Kelas | Sumber |
|---|---|
| melon | Gabungan dataset/mentah + setengah_matang + matang |
| bukan_melon | `dataset/bukan_melon/` |

`dataset_validator/` **tidak diisi manual** — dibuat otomatis oleh
`prepare_validator_dataset.py` dengan menyalin (bukan memindah) gambar
dari `dataset/`, sehingga data asli tidak pernah berubah.

---

## 📁 Struktur Project

```
melon_cnn/
│
├── app.py                        # Flask app (tidak berubah strukturnya)
├── predict.py                    # Modul inferensi cascade (Model 1 → Model 2)
├── prepare_validator_dataset.py  # Bangun folder dataset_validator/
├── train_validator.py            # Training Model 1 (Validator Binary)
├── train_transfer.py             # Training Model 2 (Klasifikasi Kematangan)
├── train.py                      # (opsional/legacy) CNN custom 3 kelas
├── dataset_audit.py              # Diagnosis kualitas dataset
│
├── validator_model.h5            # Model 1 (dibuat oleh train_validator.py)
├── validator_model.keras
├── validator_class_names.json    # ["bukan_melon", "melon"]
├── validator_model_config.json
│
├── model_melon.h5                # Model 2 (dibuat oleh train_transfer.py)
├── model_melon.keras
├── class_names.json              # ["matang", "mentah", "setengah_matang"]
├── model_config.json
│
├── static/{uploads,css,js}/      # TIDAK DIUBAH (UI/UX tetap)
├── templates/*.html              # TIDAK DIUBAH (UI/UX tetap)
├── database/melon.sql
│
└── dataset/
    ├── mentah/
    ├── setengah_matang/
    ├── matang/
    └── bukan_melon/
```

---

## ⚙️ Instalasi

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
mysql -u root -p < database/melon.sql
cp .env.example .env
```

---

## 🧠 Training — Urutan Wajib

### Langkah 1: Siapkan struktur dataset validator
```bash
python prepare_validator_dataset.py
```
Membentuk `dataset_validator/melon/` dan `dataset_validator/bukan_melon/`
dari data yang sudah ada di `dataset/`. Jalankan ulang kapan pun data
sumber diperbarui.

### Langkah 2: Latih Model 1 (Validator Objek)
```bash
python train_validator.py
```
MobileNetV2 pretrained → freeze seluruh base → training 15 epoch →
buka 30 layer terakhir → fine-tune 15 epoch (Adam, lr=1e-5).

**Output**: `validator_model.h5`, `validator_model.keras`,
`validator_class_names.json`, `validator_model_config.json`,
`accuracy_plot.png`, `loss_plot.png`, `confusion_matrix.png`,
`classification_report.txt` (+ salinan berprefix `validator_*` sebagai
backup, lihat catatan di bawah).

### Langkah 3: Latih Model 2 (Klasifikasi Kematangan)
```bash
python train_transfer.py
```
Arsitektur identik (MobileNetV2 freeze → fine-tune 30 layer terakhir),
tapi dengan 3 kelas kematangan dan dataset yang sepenuhnya terpisah.

**Output**: `model_melon.h5`, `model_melon.keras`, `class_names.json`,
`model_config.json`, + plot & report yang sama (backup prefix `model2_*`).

> ⚠️ **Penting**: kedua script menghasilkan nama file plot yang SAMA
> (`accuracy_plot.png`, dst.) — kalau dijalankan berurutan, yang terakhir
> jalan akan menimpa punya yang pertama. Karena itu masing-masing script
> otomatis membuat salinan berprefix (`validator_*` dan `model2_*`) agar
> kedua hasil tetap tersedia untuk BAB IV.

### Langkah 4: Jalankan Website
```bash
python app.py
# http://localhost:5000  |  Login: admin / admin123
```
Kedua model (`validator_model.h5` DAN `model_melon.h5`) harus sudah ada
sebelum menjalankan `app.py`.

---

## 🔎 Cara Kerja predict.py

```python
from predict import predict_image
result = predict_image('path/ke/gambar.jpg')
```

**Jika bukan melon:**
```python
{"status": "bukan_melon", "validator_confidence": 0.92}
```

**Jika melon:**
```python
{
    "status": "success",
    "kelas": "Matang",
    "confidence": 0.95,
    "probabilitas": {"Matang": 0.95, "Mentah": 0.03, "Setengah Matang": 0.02}
}
```

Model 2 hanya dimuat ke memori jika Model 1 mengembalikan `melon` —
menghemat waktu inferensi untuk kasus yang langsung ditolak di tahap 1.

---

## 📸 Fitur Kamera Langsung
Tidak berubah — halaman Deteksi Melon tetap mendukung upload file dan
ambil foto langsung via kamera perangkat.

---

## 🔧 Troubleshooting

**"validator_model.h5 tidak ditemukan"**
→ Jalankan `python prepare_validator_dataset.py` lalu `python train_validator.py`.

**"model_melon.h5 tidak ditemukan"**
→ Jalankan `python train_transfer.py`.

**Validator terlalu sering menolak foto melon asli / terlalu longgar menerima:**
→ Tambah variasi data di `dataset/bukan_melon/` atau kelas melon, lalu
  jalankan ulang `prepare_validator_dataset.py` dan `train_validator.py`.

**Plot Model 1 tertimpa oleh Model 2 (atau sebaliknya):**
→ Gunakan file berprefix `validator_*` dan `model2_*` sebagai referensi
  permanen untuk BAB IV.

**Kamera tidak muncul:**
→ Akses via `http://localhost:5000` atau HTTPS di production.
