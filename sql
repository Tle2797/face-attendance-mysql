CREATE DATABASE IF NOT EXISTS face_attendance
  CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
USE face_attendance;

-- นักศึกษา
CREATE TABLE IF NOT EXISTS students (
  student_code VARCHAR(32) PRIMARY KEY,              -- กันซ้ำด้วย PK
  title ENUM('นาย','นางสาว') NOT NULL,
  full_name VARCHAR(120) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uniq_full_name (full_name)              -- ⛔ ชื่อซ้ำไม่ได้
) ENGINE=InnoDB;

-- รูปภาพ
CREATE TABLE IF NOT EXISTS student_images (
  id INT AUTO_INCREMENT PRIMARY KEY,
  student_code VARCHAR(32) NOT NULL,
  filename VARCHAR(255) NOT NULL,
  mime VARCHAR(64) NOT NULL,
  image LONGBLOB NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_imgs_student
    FOREIGN KEY (student_code) REFERENCES students(student_code)
    ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB;

-- ประวัติเช็คชื่อ
CREATE TABLE IF NOT EXISTS attendance (
  id INT AUTO_INCREMENT PRIMARY KEY,
  student_code VARCHAR(32) NOT NULL,
  checked_at DATETIME NOT NULL,
  CONSTRAINT fk_att_student
    FOREIGN KEY (student_code) REFERENCES students(student_code)
    ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB;
