-- ============================================================
-- DATABASE: melon_cnn
-- Project: Deteksi Kematangan Buah Melon dengan CNN
-- ============================================================

CREATE DATABASE IF NOT EXISTS melon_cnn
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE melon_cnn;

-- ------------------------------------------------------------
-- Tabel: tbl_user
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tbl_user (
  id_user    INT          NOT NULL AUTO_INCREMENT,
  username   VARCHAR(50)  NOT NULL UNIQUE,
  password   VARCHAR(255) NOT NULL,
  nama       VARCHAR(100) NOT NULL,
  role       ENUM('admin','user') NOT NULL DEFAULT 'user',
  created_at DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id_user)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ------------------------------------------------------------
-- Tabel: tbl_data_melon
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tbl_data_melon (
  id_melon       INT          NOT NULL AUTO_INCREMENT,
  nama_file      VARCHAR(255) NOT NULL,
  gambar         VARCHAR(255) NOT NULL,
  tanggal_upload DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id_melon)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ------------------------------------------------------------
-- Tabel: tbl_hasil_klasifikasi
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tbl_hasil_klasifikasi (
  id_hasil      INT           NOT NULL AUTO_INCREMENT,
  id_melon      INT           NOT NULL,
  hasil_prediksi VARCHAR(50)  NOT NULL,
  confidence    DECIMAL(5,2)  NOT NULL,
  tanggal       DATETIME      NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id_hasil),
  CONSTRAINT fk_melon FOREIGN KEY (id_melon)
    REFERENCES tbl_data_melon (id_melon)
    ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ------------------------------------------------------------
-- Seed: Default admin user
-- password: admin123 (bcrypt hashed)
-- ------------------------------------------------------------
INSERT INTO tbl_user (username, password, nama, role) VALUES
(
  'admin',
  '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6hsxq7.GWS',
  'Administrator',
  'admin'
);

-- Note: password hash di atas adalah untuk 'admin123'
-- Generate ulang dengan: python -c "from werkzeug.security import generate_password_hash; print(generate_password_hash('admin123'))"
