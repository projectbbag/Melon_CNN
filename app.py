# ============================================================
# app.py - Main Flask Application
# Deteksi Kematangan Buah Melon CNN
# ============================================================

import os
import json
import uuid
from datetime import datetime
from functools import wraps

from flask import (Flask, render_template, request, redirect,
                   url_for, session, flash, jsonify)
from flask_mysqldb import MySQL
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from predict import predict_image
# Catatan: melon_validator.py (heuristik warna/bentuk) tidak lagi dipakai
# sebagai gerbang utama sejak model CNN dilatih ulang dengan kelas
# "Bukan Melon" yang sesungguhnya (data asli, bukan heuristik). File
# melon_validator.py tetap disimpan sebagai referensi/opsional.

# ── App Configuration ────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'melon_cnn_secret_2024_xK9#mP')

# ── Upload Config ─────────────────────────────────────────────
UPLOAD_FOLDER  = os.path.join('static', 'uploads')
ALLOWED_EXT    = {'jpg', 'jpeg', 'png'}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024   # 5 MB
CONFIDENCE_THRESHOLD = 0.70            # < 70% → tampilkan peringatan "gambar tidak dikenali"

app.config['UPLOAD_FOLDER']       = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH']  = MAX_CONTENT_LENGTH
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── MySQL Config ──────────────────────────────────────────────
app.config['MYSQL_HOST']     = os.environ.get('MYSQL_HOST',   'cnnmelon-cnnmelon.e.aivencloud.com')
app.config['MYSQL_USER']     = os.environ.get('MYSQL_USER',   'avnadmin')
app.config['MYSQL_PASSWORD'] = os.environ.get('MYSQL_PASSWORD', 'AVNS_YW17ZNKdPGZWiaFQ4XH')
app.config['MYSQL_DB']       = os.environ.get('MYSQL_DB',     'defaultdb')
app.config['MYSQL_PORT'] = int(os.getenv('DB_PORT', 14000))
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)


# ── Helpers ───────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Silakan login terlebih dahulu.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Silakan login terlebih dahulu.', 'warning')
            return redirect(url_for('login'))
        if session.get('role') != 'admin':
            flash('Akses ditolak. Hanya admin yang diperbolehkan.', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated


# ── Auth Routes ───────────────────────────────────────────────
@app.route('/', methods=['GET', 'POST'])
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM tbl_user WHERE username = %s", (username,))
        user = cur.fetchone()
        cur.close()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id_user']
            session['username'] = user['username']
            session['nama']     = user['nama']
            session['role']     = user['role']
            flash(f'Selamat datang, {user["nama"]}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Username atau password salah.', 'danger')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('Anda telah keluar.', 'info')
    return redirect(url_for('login'))


# ── Dashboard ─────────────────────────────────────────────────
@app.route('/dashboard')
@login_required
def dashboard():
    cur = mysql.connection.cursor()

    cur.execute("SELECT COUNT(*) AS total FROM tbl_data_melon")
    total_data = cur.fetchone()['total']

    cur.execute("SELECT COUNT(*) AS total FROM tbl_hasil_klasifikasi")
    total_prediksi = cur.fetchone()['total']

    cur.execute("""
        SELECT hasil_prediksi, COUNT(*) AS jumlah
        FROM tbl_hasil_klasifikasi
        GROUP BY hasil_prediksi
    """)
    chart_raw = cur.fetchall()

    cur.execute("""
        SELECT h.hasil_prediksi, h.confidence, h.tanggal,
               m.nama_file, m.gambar
        FROM tbl_hasil_klasifikasi h
        JOIN tbl_data_melon m ON h.id_melon = m.id_melon
        ORDER BY h.tanggal DESC
        LIMIT 5
    """)
    recent = cur.fetchall()
    cur.close()

    chart_labels = [r['hasil_prediksi'] for r in chart_raw]
    chart_values = [r['jumlah'] for r in chart_raw]

    return render_template('dashboard.html',
                           total_data=total_data,
                           total_prediksi=total_prediksi,
                           chart_labels=json.dumps(chart_labels),
                           chart_values=json.dumps(chart_values),
                           recent=recent)


# ── Prediksi ──────────────────────────────────────────────────
@app.route('/prediksi', methods=['GET', 'POST'])
@login_required
def prediksi():
    if request.method == 'POST':
        if 'gambar' not in request.files:
            flash('Tidak ada file yang dipilih.', 'danger')
            return redirect(request.url)

        file = request.files['gambar']
        if file.filename == '':
            flash('Tidak ada file yang dipilih.', 'danger')
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash('Format file tidak didukung. Gunakan JPG, JPEG, atau PNG.', 'danger')
            return redirect(request.url)

        # Save file with unique name
        ext       = file.filename.rsplit('.', 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex}.{ext}"
        save_path   = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(save_path)

        # ── Cascade Classification: Model 1 (Validator) → Model 2 (Kematangan) ──
        # predict_image() menjalankan kedua tahap secara internal dan
        # mengembalikan dict dengan field 'status':
        #   - status == 'bukan_melon' → Model 2 TIDAK dijalankan
        #   - status == 'success'     → hasil klasifikasi kematangan tersedia
        try:
            result = predict_image(save_path)
        except Exception as e:
            flash(f'Error saat prediksi: {str(e)}', 'danger')
            os.remove(save_path)
            return redirect(request.url)

        if result['status'] == 'bukan_melon':
            # Proses berhenti di sini -- tidak masuk ke Model 2, tidak
            # disimpan ke database. Hapus file yang sudah terlanjur disimpan.
            os.remove(save_path)
            flash(
                'Gambar yang diunggah bukan merupakan buah melon. '
                'Silakan unggah gambar buah melon untuk dilakukan deteksi kematangan.',
                'warning'
            )
            return redirect(url_for('prediksi'))

        # status == 'success' -- Model 2 sudah menghasilkan kelas kematangan
        label      = result['kelas']
        confidence = result['confidence']

        # Insert ke DB (hanya untuk citra yang lolos validasi sebagai melon)
        cur = mysql.connection.cursor()
        cur.execute(
            "INSERT INTO tbl_data_melon (nama_file, gambar) VALUES (%s, %s)",
            (secure_filename(file.filename), unique_name)
        )
        id_melon = cur.lastrowid

        cur.execute(
            "INSERT INTO tbl_hasil_klasifikasi (id_melon, hasil_prediksi, confidence) VALUES (%s, %s, %s)",
            (id_melon, label, round(confidence * 100, 2))
        )
        mysql.connection.commit()
        id_hasil = cur.lastrowid
        cur.close()

        # Jaring pengaman kedua: tandai jika confidence Model 2 rendah
        # (mis. melon asli tapi sudut/kualitas foto ambigu antar kelas).
        low_conf = confidence < CONFIDENCE_THRESHOLD

        return redirect(url_for('hasil', id=id_hasil, low_conf=int(low_conf)))

    return render_template('prediksi.html')


# ── Hasil ─────────────────────────────────────────────────────
@app.route('/hasil/<int:id>')
@login_required
def hasil(id):
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT h.id_hasil, h.hasil_prediksi, h.confidence, h.tanggal,
               m.nama_file, m.gambar
        FROM tbl_hasil_klasifikasi h
        JOIN tbl_data_melon m ON h.id_melon = m.id_melon
        WHERE h.id_hasil = %s
    """, (id,))
    data = cur.fetchone()
    cur.close()

    if not data:
        flash('Data tidak ditemukan.', 'danger')
        return redirect(url_for('history'))

    # Ambil flag low_conf dari query string (dikirim saat redirect dari prediksi)
    low_conf = request.args.get('low_conf', '0') == '1'

    return render_template('hasil.html', data=data, low_conf=low_conf)


# ── History ───────────────────────────────────────────────────
@app.route('/history')
@login_required
def history():
    search  = request.args.get('search', '').strip()
    page    = max(1, int(request.args.get('page', 1)))
    per_page = 10
    offset  = (page - 1) * per_page

    cur = mysql.connection.cursor()

    base_where = "WHERE h.hasil_prediksi LIKE %s OR m.nama_file LIKE %s"
    like_val   = f'%{search}%'

    cur.execute(f"""
        SELECT COUNT(*) AS total
        FROM tbl_hasil_klasifikasi h
        JOIN tbl_data_melon m ON h.id_melon = m.id_melon
        {base_where}
    """, (like_val, like_val))
    total = cur.fetchone()['total']

    cur.execute(f"""
        SELECT h.id_hasil, h.hasil_prediksi, h.confidence, h.tanggal,
               m.nama_file, m.gambar
        FROM tbl_hasil_klasifikasi h
        JOIN tbl_data_melon m ON h.id_melon = m.id_melon
        {base_where}
        ORDER BY h.tanggal DESC
        LIMIT %s OFFSET %s
    """, (like_val, like_val, per_page, offset))
    rows = cur.fetchall()
    cur.close()

    total_pages = max(1, (total + per_page - 1) // per_page)

    return render_template('history.html',
                           rows=rows,
                           search=search,
                           page=page,
                           total_pages=total_pages,
                           total=total)


@app.route('/history/delete/<int:id>', methods=['POST'])
@login_required
def delete_history(id):
    cur = mysql.connection.cursor()
    # Get gambar filename before delete
    cur.execute("""
        SELECT m.gambar FROM tbl_hasil_klasifikasi h
        JOIN tbl_data_melon m ON h.id_melon = m.id_melon
        WHERE h.id_hasil = %s
    """, (id,))
    row = cur.fetchone()

    if row:
        # Delete file
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], row['gambar'])
        if os.path.exists(img_path):
            os.remove(img_path)

    cur.execute("DELETE FROM tbl_hasil_klasifikasi WHERE id_hasil = %s", (id,))
    mysql.connection.commit()
    cur.close()
    flash('Data berhasil dihapus.', 'success')
    return redirect(url_for('history'))


# ── Manajemen User ────────────────────────────────────────────
@app.route('/users')
@admin_required
def users():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM tbl_user ORDER BY id_user")
    all_users = cur.fetchall()
    cur.close()
    return render_template('users.html', users=all_users)


@app.route('/users/tambah', methods=['GET', 'POST'])
@admin_required
def tambah_user():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        nama     = request.form.get('nama', '').strip()
        role     = request.form.get('role', 'user')

        hashed = generate_password_hash(password)
        try:
            cur = mysql.connection.cursor()
            cur.execute(
                "INSERT INTO tbl_user (username, password, nama, role) VALUES (%s,%s,%s,%s)",
                (username, hashed, nama, role)
            )
            mysql.connection.commit()
            cur.close()
            flash('User berhasil ditambahkan.', 'success')
            return redirect(url_for('users'))
        except Exception:
            flash('Username sudah digunakan.', 'danger')

    return render_template('users.html', mode='tambah')


@app.route('/users/edit/<int:id>', methods=['POST'])
@admin_required
def edit_user(id):
    nama     = request.form.get('nama', '').strip()
    role     = request.form.get('role', 'user')
    password = request.form.get('password', '')

    cur = mysql.connection.cursor()
    if password:
        hashed = generate_password_hash(password)
        cur.execute(
            "UPDATE tbl_user SET nama=%s, role=%s, password=%s WHERE id_user=%s",
            (nama, role, hashed, id)
        )
    else:
        cur.execute(
            "UPDATE tbl_user SET nama=%s, role=%s WHERE id_user=%s",
            (nama, role, id)
        )
    mysql.connection.commit()
    cur.close()
    flash('User berhasil diperbarui.', 'success')
    return redirect(url_for('users'))


@app.route('/users/hapus/<int:id>', methods=['POST'])
@admin_required
def hapus_user(id):
    if id == session['user_id']:
        flash('Tidak dapat menghapus akun yang sedang aktif.', 'danger')
        return redirect(url_for('users'))
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM tbl_user WHERE id_user = %s", (id,))
    mysql.connection.commit()
    cur.close()
    flash('User berhasil dihapus.', 'success')
    return redirect(url_for('users'))


# ── Error Handlers ────────────────────────────────────────────
@app.errorhandler(413)
def too_large(e):
    flash('File terlalu besar. Maksimal 5 MB.', 'danger')
    return redirect(url_for('prediksi'))


@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
