# ============================================================
# prepare_validator_dataset.py - Persiapan Struktur Dataset Validator
# ============================================================
#
# TUJUAN
# ------
# Membentuk struktur folder dataset_validator/ yang dibutuhkan oleh
# train_validator.py, dengan MENYALIN (bukan memindahkan) gambar dari
# dataset/ yang sudah ada:
#
#   dataset/mentah/           ─┐
#   dataset/setengah_matang/   ├──► disalin ke ──► dataset_validator/melon/
#   dataset/matang/           ─┘
#
#   dataset/bukan_melon/       ──► disalin ke ──► dataset_validator/bukan_melon/
#
# Dataset ASLI di dataset/ TIDAK PERNAH diubah atau dihapus -- script
# ini murni menyalin file agar Model 2 (klasifikasi kematangan, dataset/)
# dan Model 1 (validator objek, dataset_validator/) memiliki struktur
# folder masing-masing yang sesuai kebutuhan flow_from_directory,
# tanpa data yang tumpang tindih secara destruktif.
#
# CARA PAKAI
# ----------
#   python prepare_validator_dataset.py
#
# Jalankan ulang kapan saja jika dataset/ diperbarui -- script akan
# membersihkan dataset_validator/ lalu menyalin ulang agar selalu
# sinkron dengan data sumber terbaru.
# ============================================================

import os
import shutil

BASE_DIR      = os.path.dirname(__file__)
SOURCE_DIR    = os.path.join(BASE_DIR, 'dataset')
TARGET_DIR    = os.path.join(BASE_DIR, 'dataset_validator')
VALID_EXT     = {'.jpg', '.jpeg', '.png'}

MELON_FOLDERS     = ['mentah', 'setengah_matang', 'matang']
NON_MELON_FOLDER  = 'bukan_melon'


def clean_and_make(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)


def copy_images(src_folders, dst_folder):
    count = 0
    for src_name in src_folders:
        src_path = os.path.join(SOURCE_DIR, src_name)
        if not os.path.isdir(src_path):
            print(f"  [WARNING] Folder sumber tidak ditemukan: {src_path}")
            continue
        for fname in os.listdir(src_path):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in VALID_EXT:
                continue
            # Prefix nama folder asal agar tidak ada nama file bentrok
            # antar sub-kelas kematangan yang digabung ke folder 'melon'
            new_name = f"{src_name}_{fname}"
            shutil.copy2(
                os.path.join(src_path, fname),
                os.path.join(dst_folder, new_name)
            )
            count += 1
    return count


def main():
    print("=" * 60)
    print("  Persiapan Dataset Validator Objek")
    print("=" * 60)

    melon_dst     = os.path.join(TARGET_DIR, 'melon')
    bukan_melon_dst = os.path.join(TARGET_DIR, 'bukan_melon')

    print(f"\n  Membersihkan & membuat ulang: {TARGET_DIR}")
    clean_and_make(melon_dst)
    clean_and_make(bukan_melon_dst)

    print("\n  Menyalin gambar kelas 'melon' (mentah + setengah_matang + matang)...")
    n_melon = copy_images(MELON_FOLDERS, melon_dst)
    print(f"    → {n_melon} gambar disalin ke dataset_validator/melon/")

    print("\n  Menyalin gambar kelas 'bukan_melon'...")
    n_bukan = copy_images([NON_MELON_FOLDER], bukan_melon_dst)
    print(f"    → {n_bukan} gambar disalin ke dataset_validator/bukan_melon/")

    print("\n" + "-" * 60)
    print(f"  Total dataset_validator/melon        : {n_melon} gambar")
    print(f"  Total dataset_validator/bukan_melon  : {n_bukan} gambar")
    print("-" * 60)

    if n_melon == 0 or n_bukan == 0:
        print("\n  [ERROR] Salah satu kelas kosong. Pastikan folder dataset/mentah,")
        print("  dataset/setengah_matang, dataset/matang, dan dataset/bukan_melon")
        print("  sudah berisi gambar sebelum menjalankan script ini.")
        return

    print("\n  Selesai. Struktur dataset_validator/ siap dipakai oleh train_validator.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
