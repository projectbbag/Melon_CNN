# ============================================================
# dataset_audit.py - Alat Diagnosis Dataset Sebelum Training Ulang
# ============================================================
#
# TUJUAN
# ------
# Mendiagnosis PENYEBAB kebingungan model antara kelas Matang dan
# Mentah SEBELUM memutuskan untuk training ulang. Retraining tanpa
# memperbaiki akar masalah biasanya hanya mengulang masalah yang sama.
#
# Tool ini memeriksa 3 penyebab paling umum:
#   1. KETIDAKSEIMBANGAN KELAS   (jumlah gambar per kelas timpang)
#   2. KUALITAS GAMBAR TIDAK KONSISTEN (blur, gelap/terlalu terang)
#   3. KEMUNGKINAN SALAH LABEL   (gambar yang warnanya menyerupai
#      kelas lain lebih dari kelasnya sendiri -- kandidat untuk
#      diperiksa ulang secara manual)
#
# CARA PAKAI
# ----------
#   python dataset_audit.py
#
# Output: laporan ringkas di terminal + file audit_report.csv
#         berisi detail per gambar untuk ditelusuri lebih lanjut.
# ============================================================

import os
import csv
import cv2
import numpy as np
from collections import defaultdict

DATASET_DIR = os.path.join(os.path.dirname(__file__), 'dataset')
CLASSES     = ['mentah', 'setengah_matang', 'matang']
VALID_EXT   = {'.jpg', '.jpeg', '.png'}

BLUR_THRESHOLD       = 100.0   # Laplacian variance < ini = kemungkinan blur
DARK_THRESHOLD        = 60     # rata-rata V (brightness) < ini = terlalu gelap
BRIGHT_THRESHOLD      = 220    # rata-rata V > ini = terlalu terang/overexposed
MISLABEL_MARGIN       = 1.15   # jarak ke kelas lain harus < (jarak ke kelas sendiri / margin) utk dicurigai


def list_images(folder):
    if not os.path.isdir(folder):
        return []
    return [f for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in VALID_EXT]


def analyze_image(path):
    """Ekstrak metrik kualitas & warna dari satu gambar."""
    img = cv2.imread(path)
    if img is None:
        return None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(np.mean(hsv[:, :, 2]))

    # Rata-rata Hue & Saturation (representasi warna dominan sederhana)
    mean_hue = float(np.mean(hsv[:, :, 0]))
    mean_sat = float(np.mean(hsv[:, :, 1]))

    return {
        'width': w, 'height': h,
        'blur_score': round(blur_score, 1),
        'brightness': round(brightness, 1),
        'mean_hue': round(mean_hue, 1),
        'mean_sat': round(mean_sat, 1),
    }


def main():
    print("=" * 64)
    print("  AUDIT DATASET - Diagnosis Kebingungan Matang vs Mentah")
    print("=" * 64)

    # ── Kumpulkan data semua gambar ──────────────────────────
    all_records = []   # list of dict: {class, filename, path, **metrics}
    per_class_count = {}

    for cls in CLASSES:
        folder = os.path.join(DATASET_DIR, cls)
        files = list_images(folder)
        per_class_count[cls] = len(files)

        for fname in files:
            path = os.path.join(folder, fname)
            metrics = analyze_image(path)
            if metrics is None:
                print(f"  [SKIP] Gagal membaca: {cls}/{fname}")
                continue
            record = {'class': cls, 'filename': fname, 'path': path, **metrics}
            all_records.append(record)

    if not all_records:
        print("\n  Tidak ada gambar ditemukan di folder dataset/.")
        print("  Pastikan struktur folder: dataset/mentah, dataset/setengah_matang, dataset/matang")
        return

    # ══════════════════════════════════════════════════════════
    # 1. KETIDAKSEIMBANGAN KELAS
    # ══════════════════════════════════════════════════════════
    print("\n" + "-" * 64)
    print("  1. KESEIMBANGAN JUMLAH DATA PER KELAS")
    print("-" * 64)
    total = sum(per_class_count.values())
    max_count = max(per_class_count.values())
    min_count = min(per_class_count.values())

    for cls in CLASSES:
        n = per_class_count[cls]
        pct = (n / total * 100) if total else 0
        bar = '█' * int(pct / 2)
        print(f"  {cls:18s} : {n:4d} gambar  ({pct:5.1f}%)  {bar}")

    if min_count > 0 and (max_count / min_count) > 1.5:
        print(f"\n  ⚠ PERINGATAN: Kelas timpang! Kelas terbanyak {max_count} vs "
              f"tersedikit {min_count} gambar (rasio {max_count/min_count:.1f}x).")
        print("    Dampak: model cenderung bias memprediksi kelas mayoritas.")
        print("    Saran: tambah data pada kelas minoritas, atau gunakan class_weight saat training.")
    else:
        print("\n  ✓ Jumlah data antar kelas relatif seimbang.")

    # ══════════════════════════════════════════════════════════
    # 2. KUALITAS GAMBAR (blur & pencahayaan)
    # ══════════════════════════════════════════════════════════
    print("\n" + "-" * 64)
    print("  2. KUALITAS GAMBAR (BLUR & PENCAHAYAAN)")
    print("-" * 64)

    blur_issues  = [r for r in all_records if r['blur_score'] < BLUR_THRESHOLD]
    dark_issues  = [r for r in all_records if r['brightness'] < DARK_THRESHOLD]
    bright_issues = [r for r in all_records if r['brightness'] > BRIGHT_THRESHOLD]

    print(f"  Gambar kemungkinan blur       : {len(blur_issues)} / {len(all_records)}")
    print(f"  Gambar terlalu gelap          : {len(dark_issues)} / {len(all_records)}")
    print(f"  Gambar terlalu terang/silau   : {len(bright_issues)} / {len(all_records)}")

    # Bandingkan rata-rata brightness ANTAR kelas -- ini kunci untuk
    # kasus Matang vs Mentah, karena jika pencahayaan saat pemotretan
    # tiap kelas berbeda jauh, model bisa "belajar" brightness alih-alih
    # warna kematangan yang sebenarnya.
    print("\n  Rata-rata brightness (V) per kelas:")
    class_brightness = {}
    for cls in CLASSES:
        vals = [r['brightness'] for r in all_records if r['class'] == cls]
        avg = np.mean(vals) if vals else 0
        class_brightness[cls] = avg
        print(f"    {cls:18s} : {avg:6.1f}")

    bvals = list(class_brightness.values())
    if bvals and (max(bvals) - min(bvals)) > 30:
        print(f"\n  ⚠ PERINGATAN: Selisih brightness antar kelas cukup besar "
              f"({max(bvals)-min(bvals):.1f}).")
        print("    Ini BERISIKO: model bisa keliru mengasosiasikan kecerahan foto")
        print("    dengan tingkat kematangan, alih-alih fitur kematangan yang asli.")
        print("    Saran: pastikan kondisi pencahayaan konsisten saat pengambilan foto")
        print("    untuk SEMUA kelas (gunakan lightbox/pencahayaan seragam).")
    else:
        print("\n  ✓ Pencahayaan antar kelas relatif konsisten.")

    # ══════════════════════════════════════════════════════════
    # 3. KEMUNGKINAN SALAH LABEL (berbasis kemiripan warna Hue)
    # ══════════════════════════════════════════════════════════
    print("\n" + "-" * 64)
    print("  3. KANDIDAT SALAH LABEL (berdasarkan warna dominan)")
    print("-" * 64)

    # Hitung centroid (rata-rata hue, saturation) tiap kelas
    centroids = {}
    for cls in CLASSES:
        vals = [(r['mean_hue'], r['mean_sat']) for r in all_records if r['class'] == cls]
        if vals:
            centroids[cls] = (np.mean([v[0] for v in vals]), np.mean([v[1] for v in vals]))

    print("  Centroid warna (Hue, Saturation) tiap kelas:")
    for cls, (h, s) in centroids.items():
        print(f"    {cls:18s} : Hue={h:5.1f}  Sat={s:5.1f}")

    suspects = []
    for r in all_records:
        own_cls = r['class']
        if own_cls not in centroids:
            continue
        own_h, own_s = centroids[own_cls]
        own_dist = np.hypot(r['mean_hue'] - own_h, r['mean_sat'] - own_s)

        for other_cls, (oh, os_) in centroids.items():
            if other_cls == own_cls:
                continue
            other_dist = np.hypot(r['mean_hue'] - oh, r['mean_sat'] - os_)
            if other_dist * MISLABEL_MARGIN < own_dist:
                suspects.append({
                    **r,
                    'closer_to': other_cls,
                    'own_dist': round(own_dist, 1),
                    'other_dist': round(other_dist, 1),
                })
                break

    if suspects:
        print(f"\n  ⚠ Ditemukan {len(suspects)} gambar yang warnanya lebih mirip "
              f"kelas LAIN dibanding kelas labelnya sendiri:")
        for s in sorted(suspects, key=lambda x: x['other_dist'])[:15]:
            print(f"    - {s['class']}/{s['filename']:30s} "
                  f"→ lebih mirip '{s['closer_to']}' "
                  f"(jarak sendiri={s['own_dist']}, ke-{s['closer_to']}={s['other_dist']})")
        if len(suspects) > 15:
            print(f"    ... dan {len(suspects)-15} gambar lainnya (lihat audit_report.csv)")
        print("\n  → PERIKSA gambar-gambar ini secara manual. Bisa jadi:")
        print("    a) Salah label saat pengumpulan data, ATAU")
        print("    b) Gambar transisi/ambigu (mis. melon yang benar2 di ambang mentah↔matang)")
    else:
        print("\n  ✓ Tidak ditemukan kandidat salah label yang mencolok.")

    # ══════════════════════════════════════════════════════════
    # EXPORT CSV
    # ══════════════════════════════════════════════════════════
    csv_path = os.path.join(os.path.dirname(__file__), 'audit_report.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'class', 'filename', 'width', 'height', 'blur_score',
            'brightness', 'mean_hue', 'mean_sat'
        ])
        writer.writeheader()
        for r in all_records:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    print("\n" + "-" * 64)
    print(f"  Detail lengkap semua gambar disimpan ke: audit_report.csv")
    print("-" * 64)

    # ══════════════════════════════════════════════════════════
    # KESIMPULAN & REKOMENDASI
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 64)
    print("  KESIMPULAN")
    print("=" * 64)
    issues_found = []
    if min_count > 0 and (max_count / min_count) > 1.5:
        issues_found.append("ketidakseimbangan jumlah data antar kelas")
    if bvals and (max(bvals) - min(bvals)) > 30:
        issues_found.append("inkonsistensi pencahayaan antar kelas")
    if len(suspects) > max(3, len(all_records) * 0.03):
        issues_found.append("indikasi salah label pada beberapa gambar")
    if len(blur_issues) > len(all_records) * 0.1:
        issues_found.append("proporsi gambar blur cukup tinggi (>10%)")

    if issues_found:
        print("  Ditemukan potensi masalah:")
        for i, issue in enumerate(issues_found, 1):
            print(f"    {i}. {issue}")
        print("\n  REKOMENDASI: Perbaiki dulu masalah di atas SEBELUM training ulang.")
        print("  Training ulang tanpa memperbaiki data cenderung mengulang hasil yang sama.")
    else:
        print("  Tidak ditemukan masalah data yang signifikan.")
        print("  Kebingungan Matang/Mentah kemungkinan berasal dari FAKTOR MODEL,")
        print("  bukan data. Lihat saran arsitektur & training di bawah.")

    print("=" * 64)


if __name__ == '__main__':
    main()
