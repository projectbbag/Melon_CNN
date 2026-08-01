"""
setup_db.py - Database Setup & Seed Script
Jalankan SETELAH membuat database melon_cnn di MySQL
"""
import os
from werkzeug.security import generate_password_hash
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

conn = mysql.connector.connect(
    host=os.getenv('DB_HOST','localhost'),
    port=int(os.getenv('DB_PORT',3306)),
    user=os.getenv('DB_USER','root'),
    password=os.getenv('DB_PASSWORD',''),
    database=os.getenv('DB_NAME','melon_cnn'),
    charset='utf8mb4'
)
cur = conn.cursor()

# Create tables
cur.execute("""CREATE TABLE IF NOT EXISTS tbl_user (
    id_user INT(11) NOT NULL AUTO_INCREMENT,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    nama VARCHAR(100) NOT NULL,
    role ENUM('admin','user') NOT NULL DEFAULT 'user',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id_user)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")

cur.execute("""CREATE TABLE IF NOT EXISTS tbl_data_melon (
    id_melon INT(11) NOT NULL AUTO_INCREMENT,
    nama_file VARCHAR(255) NOT NULL,
    gambar VARCHAR(255) NOT NULL,
    tanggal_upload DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id_melon)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")

cur.execute("""CREATE TABLE IF NOT EXISTS tbl_hasil_klasifikasi (
    id_hasil INT(11) NOT NULL AUTO_INCREMENT,
    id_melon INT(11) NOT NULL,
    hasil_prediksi ENUM('Mentah','Setengah Matang','Matang') NOT NULL,
    confidence DECIMAL(5,2) NOT NULL,
    tanggal DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id_hasil),
    FOREIGN KEY (id_melon) REFERENCES tbl_data_melon(id_melon) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")

# Seed users
users = [
    ('admin', generate_password_hash('admin123'), 'Administrator', 'admin'),
    ('user1', generate_password_hash('admin123'), 'User Demo', 'user'),
]
for u in users:
    try:
        cur.execute("INSERT INTO tbl_user (username, password, nama, role) VALUES (%s,%s,%s,%s)", u)
        print(f"  Created user: {u[0]}")
    except Exception as e:
        print(f"  User {u[0]} already exists, skip.")

conn.commit()
cur.close()
conn.close()
print("\nDatabase setup selesai!")
print("Login: admin / admin123")
