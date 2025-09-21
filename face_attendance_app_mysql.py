# face_attendance_app_mysql.py
# Windows + Python 3.10.x + XAMPP (MySQL/MariaDB via PyMySQL)
# ✔ ลงทะเบียนหลายรูป/หลายคน -> เก็บรูปลง MySQL (BLOB) + เก็บสำเนาที่ dataset/ เพื่อฝึก LBPH
# ✔ ฝึกโมเดลรวมทุกคน + บันทึก mapping label<->student_code ที่ meta/*.json
# ✔ สแกนด้วยกล้อง (MSMF index=1) -> "ติ๊ง" + พูดไทย (edge-tts) ว่า "เช็คชื่อเรียบร้อยแล้ว คำนำหน้า ชื่อ"
# ✔ บันทึกประวัติการเช็คชื่อลง DB (attendance)
# ✔ กันซ้ำ: คนเดิมต้องเว้น 10 วินาที (ทั้งในโปรแกรมและ DB)
# ✔ ลงทะเบียน: ห้ามให้ "รหัสนักศึกษา" หรือ "ชื่อ-นามสกุล" ซ้ำ
# ✔ ลด false positive: Unknown gate + ต้องได้ผลเดิมติดกันหลายเฟรม + Preprocess (CLAHE+blur)

import os, json, glob, time, threading, tempfile, asyncio
from datetime import datetime
from collections import deque
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

import pymysql  # DB
import edge_tts # TTS (online)
import pygame    # เล่นไฟล์เสียง
import winsound  # beep (ติ๊ง)

# -------------------- ตั้งค่า MySQL (XAMPP) --------------------
DB_HOST = "localhost"
DB_PORT = 3306
DB_USER = "root"          # ค่าเริ่มต้น XAMPP
DB_PASS = ""              # ถ้าตั้งรหัสผ่านเอง ให้ใส่ตรงนี้
DB_NAME = "face_attendance"

def get_db():
    return pymysql.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS,
        database=DB_NAME, charset="utf8mb4", autocommit=True
    )

# -------------------- Path & ค่าคงที่ --------------------
DATASET_DIR = "dataset"
META_DIR = "meta"
STUDENTS_JSON = os.path.join(META_DIR, "students.json")   # cache สำหรับสแกน
LABEL_MAP_JSON = os.path.join(META_DIR, "label_map.json") # cache สำหรับสแกน
MODEL_PATH = "lbph_model.yml"

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(HAAR_PATH)
FACE_SIZE = (200, 200)

PREFERRED_BACKEND = cv2.CAP_MSMF  # จาก probe ของคุณ
PREFERRED_INDEX  = 1

# --------- Recognition tuning ----------
UNKNOWN_THRESHOLD = 65.0      # ยอมรับว่า "รู้จัก" เฉพาะ conf < 60 (LBPH: ยิ่งต่ำยิ่งดี)
CONSEC_REQUIRED   = 5         # ต้องได้ชื่อเดียวกันติดกันกี่เฟรม ก่อนจะยืนยัน/บันทึก
REPEAT_COOLDOWN_SECONDS = 10  # หน่วงเวลาคนเดิม 10 วินาที

# -------------------- เสียง (edge-tts + pygame) --------------------
class Speaker:
    """
    ใช้ Microsoft Edge TTS (online) -> สังเคราะห์เสียงไทยคุณภาพสูง แล้วเล่นผ่าน pygame
    เลือกเสียงได้: 'th-TH-AcharaNeural' (หญิง), 'th-TH-NiwatNeural' (ชาย)
    """
    def __init__(self, voice: str = "th-TH-AcharaNeural"):
        self.voice = voice
        try:
            pygame.mixer.init()
        except Exception as e:
            print("pygame init failed:", e)

    def ding(self):
        try:
            winsound.Beep(1000, 180)
        except Exception:
            pass

    async def _synthesize_to_file(self, text, outpath):
        tts = edge_tts.Communicate(text, voice=self.voice)
        await tts.save(outpath)

    def say_async(self, text: str):
        def _worker():
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    mp3_path = tmp.name
                asyncio.run(self._synthesize_to_file(text, mp3_path))
                pygame.mixer.music.load(mp3_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                try:
                    os.remove(mp3_path)
                except Exception:
                    pass
            except Exception as e:
                print("TTS error:", e)
        threading.Thread(target=_worker, daemon=True).start()

speaker = Speaker(voice="th-TH-AcharaNeural")  # เปลี่ยนเป็น 'th-TH-NiwatNeural' ได้

# -------------------- DB Helpers --------------------
def student_code_exists(code: str) -> bool:
    sql = "SELECT 1 FROM students WHERE student_code=%s LIMIT 1"
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (code,))
            return cur.fetchone() is not None

def student_name_exists(full_name: str) -> bool:
    sql = "SELECT 1 FROM students WHERE full_name=%s LIMIT 1"
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (full_name,))
            return cur.fetchone() is not None

def insert_student_strict(code: str, title: str, name: str):
    # ห้ามซ้ำทั้ง "รหัส" และ "ชื่อ-นามสกุล"
    if student_code_exists(code):
        raise RuntimeError(f"รหัสนักศึกษา {code} มีอยู่แล้ว")
    if student_name_exists(name):
        raise RuntimeError(f"ชื่อ-นามสกุล '{name}' มีอยู่แล้ว")
    sql = "INSERT INTO students (student_code, title, full_name) VALUES (%s,%s,%s)"
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (code, title, name))

def insert_student_image(code: str, filename: str, mime: str, blob: bytes):
    sql = """
        INSERT INTO student_images (student_code, filename, mime, image)
        VALUES (%s,%s,%s,%s)
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (code, filename, mime, blob))

def write_attendance_db(code: str):
    """
    บันทึก attendance โดยเช็คว่า 10 วินาทีล่าสุดมีบันทึกของคนนี้ไปแล้วหรือยัง
    ถ้ามีภายในช่วง cooldown จะ 'ไม่' แทรกซ้ำ
    """
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT checked_at FROM attendance WHERE student_code=%s ORDER BY checked_at DESC LIMIT 1",
                (code,),
            )
            row = cur.fetchone()
            now = datetime.now()
            if row:
                last = row[0] if isinstance(row[0], datetime) else datetime.strptime(str(row[0]), "%Y-%m-%d %H:%M:%S")
                if (now - last).total_seconds() < REPEAT_COOLDOWN_SECONDS:
                    return  # ภายใน 10 วินาที ไม่บันทึกซ้ำ
            cur.execute(
                "INSERT INTO attendance (student_code, checked_at) VALUES (%s, %s)",
                (code, now.strftime("%Y-%m-%d %H:%M:%S")),
            )

def load_students_from_db():
    sql = "SELECT student_code, title, full_name FROM students"
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    return {r[0]: {"title": r[1], "name": r[2]} for r in rows}

# -------------------- Face Helpers --------------------
def detect_faces(gray: np.ndarray):
    return FACE_CASCADE.detectMultiScale(gray, 1.2, 5, minSize=(60,60))

def face_from_path(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    faces = detect_faces(arr)
    if len(faces)==0:
        return cv2.resize(arr, FACE_SIZE)
    x,y,w,h = faces[0]
    return cv2.resize(arr[y:y+h, x:x+w], FACE_SIZE)

def preprocess_face(gray_face_200x200: np.ndarray) -> np.ndarray:
    """
    ปรับภาพก่อนส่งเข้า LBPH:
    - CLAHE เพิ่มคอนทราสต์
    - GaussianBlur เบาๆ กันนอยซ์
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(gray_face_200x200)
    blur = cv2.GaussianBlur(eq, (3,3), 0)
    return blur

def augment(img: np.ndarray):
    out = [img, cv2.flip(img,1), cv2.GaussianBlur(img,(3,3),0)]
    out.append(cv2.convertScaleAbs(img, alpha=1.2, beta=30))   # bright
    out.append(cv2.convertScaleAbs(img, alpha=0.8, beta=-30))  # dark
    return out

def build_training_data():
    faces, labels = [], []
    label_map, code_to_label = {}, {}
    next_label = 1
    for code_dir in sorted(os.listdir(DATASET_DIR)):
        d = os.path.join(DATASET_DIR, code_dir)
        if not os.path.isdir(d): continue
        files = sorted(glob.glob(os.path.join(d, "*.*")))
        if not files: continue
        code_to_label[code_dir] = next_label
        label_map[next_label] = code_dir
        lab = next_label; next_label += 1
        for p in files:
            try:
                base = face_from_path(p)
                for a in augment(base):
                    faces.append(a); labels.append(lab)
            except Exception:
                pass
    return faces, np.array(labels, dtype=np.int32), label_map

def train_and_save_model():
    faces, labels, label_map = build_training_data()
    if len(faces)==0:
        raise RuntimeError("ไม่มีรูปใน dataset/ ให้ฝึกโมเดล")
    recog = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recog.train(faces, labels)
    recog.write(MODEL_PATH)
    # cache mapping และ students (จาก DB) สำหรับตอนสแกน
    with open(LABEL_MAP_JSON, "w", encoding="utf-8") as f:
        json.dump({int(k): v for k, v in label_map.items()}, f, ensure_ascii=False, indent=2)
    students = load_students_from_db()
    with open(STUDENTS_JSON, "w", encoding="utf-8") as f:
        json.dump(students, f, ensure_ascii=False, indent=2)

# -------------------- GUI --------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Attendance (MySQL + Thai TTS)")
        self.geometry("680x640")
        self.resizable(False, False)

        self.student_code = tk.StringVar()
        self.title_choice = tk.StringVar(value="นาย")
        self.fullname = tk.StringVar()
        self.selected_images = []

        container = ttk.Frame(self); container.pack(fill="both", expand=True)
        self.frames = {}
        for F in (RegisterPage, ScanPage):
            frame = F(parent=container, controller=self)
            self.frames[F.__name__] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame("RegisterPage")

    def show_frame(self, name): self.frames[name].tkraise()

class RegisterPage(ttk.Frame):
    def __init__(self, parent, controller: App):
        super().__init__(parent); self.controller=controller
        pad = {"padx":10, "pady":6}
        ttk.Label(self, text="ลงทะเบียน (เก็บรูปเข้า MySQL)", font=("Segoe UI",16,"bold")).grid(row=0,column=0,columnspan=3,pady=(20,10))

        ttk.Label(self, text="รหัสนักศึกษา *").grid(row=1,column=0,sticky="e",**pad)
        ttk.Entry(self, textvariable=controller.student_code, width=34).grid(row=1,column=1,columnspan=2,sticky="w",**pad)

        ttk.Label(self, text="คำนำหน้า *").grid(row=2,column=0,sticky="e",**pad)
        ttk.Combobox(self, textvariable=controller.title_choice, values=["นาย","นางสาว"], width=12, state="readonly").grid(row=2,column=1,sticky="w",**pad)

        ttk.Label(self, text="ชื่อ-นามสกุล *").grid(row=3,column=0,sticky="e",**pad)
        ttk.Entry(self, textvariable=controller.fullname, width=34).grid(row=3,column=1,columnspan=2,sticky="w",**pad)

        ttk.Label(self, text="เลือกรูป (หลายไฟล์) *").grid(row=4,column=0,sticky="e",**pad)
        self.files_var = tk.StringVar(value="(ยังไม่เลือก)")
        ttk.Label(self, textvariable=self.files_var, foreground="#555", wraplength=420).grid(row=4,column=1,sticky="w",**pad)
        ttk.Button(self, text="เลือกไฟล์…", command=self.choose_files).grid(row=4,column=2,sticky="w",**pad)

        ttk.Separator(self).grid(row=5,column=0,columnspan=3,sticky="ew",pady=(10,10))
        ttk.Button(self, text="บันทึกเข้า DB + เทรนใหม่ทั้งหมด", command=self.save_to_db_and_train).grid(row=6,column=0,columnspan=3,pady=8,ipadx=12,ipady=4)
        ttk.Button(self, text="ไปหน้าเปิดกล้อง (สแกน)", command=self.goto_scan).grid(row=7,column=0,columnspan=3,pady=8,ipadx=12,ipady=4)

        note=("หมายเหตุ:\n- รูปทั้งหมดจะถูกบันทึกลงตาราง student_images (BLOB) และเก็บสำเนาที่ dataset/<code>/ เพื่อใช้เทรน\n"
              "- ทุกครั้งที่กดบันทึก ระบบจะตรวจสอบรหัส/ชื่อซ้ำก่อน และเทรนโมเดลรวมทุกคนใหม่")
        ttk.Label(self,text=note,foreground="#444").grid(row=8,column=0,columnspan=3, padx=10, sticky="w")

    def choose_files(self):
        paths = filedialog.askopenfilenames(title="เลือกรูป", filetypes=[("Images","*.jpg;*.jpeg;*.png")])
        if paths:
            self.controller.selected_images = list(paths)
            self.files_var.set(f"{len(paths)} ไฟล์ที่เลือก")

    def save_to_db_and_train(self):
        code = self.controller.student_code.get().strip()
        title= self.controller.title_choice.get().strip()
        name = self.controller.fullname.get().strip()
        files= self.controller.selected_images
        if not code or not title or not name or not files:
            messagebox.showerror("ข้อมูลไม่ครบ","กรุณากรอกข้อมูลและเลือกรูปอย่างน้อย 1 รูป"); return
        try:
            # 0) ตรวจสอบซ้ำก่อน insert (ห้ามซ้ำ code / name)
            if student_code_exists(code):
                messagebox.showerror("ซ้ำ", f"รหัสนักศึกษา {code} มีอยู่แล้ว"); return
            if student_name_exists(name):
                messagebox.showerror("ซ้ำ", f"ชื่อ-นามสกุล '{name}' มีอยู่แล้ว"); return

            # 1) insert นักศึกษา (strict)
            insert_student_strict(code, title, name)

            # 2) บันทึกรูปเข้า DB + เซฟสำเนาไป dataset/<code>/
            dst_dir = os.path.join(DATASET_DIR, code)
            os.makedirs(dst_dir, exist_ok=True)
            exist = len(glob.glob(os.path.join(dst_dir,"*.jpg")))
            for i, p in enumerate(files, start=1):
                # เซฟสำเนาไว้เทรน
                try:
                    img = Image.open(p).convert("RGB")
                    dst_path = os.path.join(dst_dir, f"{exist+i:03d}.jpg")
                    img.save(dst_path)
                except Exception as e:
                    print("save copy failed:", e)
                # อ่าน blob + mime
                with open(p, "rb") as f: data = f.read()
                ext = os.path.splitext(p)[1].lower()
                mime = "image/jpeg" if ext in [".jpg",".jpeg"] else "image/png"
                insert_student_image(code, os.path.basename(p), mime, data)

            # 3) เทรนใหม่ทั้งหมด
            train_and_save_model()
            messagebox.showinfo("สำเร็จ","บันทึกเข้า DB และเทรนโมเดลเรียบร้อย")
        except Exception as e:
            messagebox.showerror("ผิดพลาด", str(e))

    def goto_scan(self):
        self.controller.show_frame("ScanPage")
        self.controller.frames["ScanPage"].start_camera()

class ScanPage(ttk.Frame):
    def __init__(self, parent, controller: App):
        super().__init__(parent); self.controller=controller
        ttk.Label(self, text="สแกนใบหน้าเพื่อเช็คชื่อ", font=("Segoe UI",16,"bold")).pack(pady=(20,8))
        self.info_var = tk.StringVar(value="กำลังเปิดกล้อง…")
        ttk.Label(self, textvariable=self.info_var).pack(pady=(0,6))
        self.overlay_var = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.overlay_var, foreground="#D4A000").pack()

        self.canvas = tk.Canvas(self, width=520, height=360, bg="black", highlightthickness=1, highlightbackground="#bbb")
        self.canvas.pack(pady=6)

        row = ttk.Frame(self); row.pack(pady=8)
        ttk.Button(row, text="กลับไปลงทะเบียน", command=self.back).pack(side="left", padx=6)
        ttk.Button(row, text="ปิดกล้อง/ออก", command=self.quit_app).pack(side="left", padx=6)

        self.cap=None; self._loop=False; self.tk_img=None
        self.recog=None; self.label_map={}; self.students={}

        # ป้องกันซ้ำ
        self.last_seen_ts: dict[int, float] = {}     # label -> epoch seconds
        self.pred_buffer: deque = deque(maxlen=12)   # เก็บ (label, conf) ล่าสุด
        self.confirm_label: int | None = None        # label ที่เพิ่งยืนยันสำเร็จ

    def load_model_and_meta(self):
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError("ยังไม่มีโมเดล (กรุณาบันทึก+เทรนจากหน้าแรกก่อน)")
        r = cv2.face.LBPHFaceRecognizer_create(); r.read(MODEL_PATH); self.recog=r
        with open(LABEL_MAP_JSON,"r",encoding="utf-8") as f:
            self.label_map = {int(k):v for k,v in json.load(f).items()}
        with open(STUDENTS_JSON,"r",encoding="utf-8") as f:
            self.students = json.load(f)

    def _open_camera(self):
        cap = cv2.VideoCapture(PREFERRED_INDEX, PREFERRED_BACKEND)
        if not cap.isOpened(): return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        for _ in range(10):
            ok,_ = cap.read()
            if ok: break
        return cap

    def start_camera(self):
        try:
            self.load_model_and_meta()
        except Exception as e:
            messagebox.showerror("ผิดพลาด", str(e)); return
        self.cap = self._open_camera()
        if self.cap is None:
            messagebox.showerror("กล้องไม่พร้อม","เปิดกล้องไม่ได้"); return
        self._loop=True
        self.info_var.set("พร้อมสแกนแล้ว — เข้ามาใกล้กล้อง 40–80 ซม.")
        self.after(10, self.update_frame)

    def stop_camera(self):
        self._loop=False
        if self.cap is not None:
            try: self.cap.release()
            except Exception: pass
            self.cap=None

    def back(self):
        self.stop_camera(); self.controller.show_frame("RegisterPage")

    def quit_app(self):
        self.stop_camera(); self.controller.destroy()

    def _confirm_identity(self, label: int, conf: float) -> bool:
        """
        เก็บผลทำนายในบัฟเฟอร์ แล้วตรวจว่า label นี้ที่ conf < UNKNOWN_THRESHOLD
        ปรากฏ 'ติดกัน' อย่างน้อย CONSEC_REQUIRED เฟรมหรือยัง
        """
        self.pred_buffer.append((label, conf))
        count = 0
        for lb, cf in reversed(self.pred_buffer):
            if lb == label and cf < UNKNOWN_THRESHOLD:
                count += 1
                if count >= CONSEC_REQUIRED:
                    return True
            else:
                break
        return False

    def update_frame(self):
        if not self._loop or self.cap is None: return
        ok, frame = self.cap.read()
        if not ok or frame is None or frame.size==0:
            self.info_var.set("อ่านภาพจากกล้องไม่ได้"); self.after(30, self.update_frame); return

        frame = cv2.flip(frame,1)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray,1.2,5,minSize=(60,60))

        text=""
        if len(faces)==1 and self.recog is not None:
            x,y,w,h = faces[0]
            roi = cv2.resize(gray[y:y+h, x:x+w], FACE_SIZE)
            roi = preprocess_face(roi)

            label, conf = self.recog.predict(roi)
            color = (0,180,255)

            # 1) conf ไม่ผ่าน -> ไม่รู้จัก
            if conf >= UNKNOWN_THRESHOLD or label not in self.label_map:
                self.confirm_label = None
                self.pred_buffer.clear()
                text = "ไม่รู้จักบุคคลนี้ — กรุณาลงทะเบียน"
                color = (80,80,220)
            else:
                # 2) ต้องได้ซ้ำติดกันพอ ก่อนยืนยัน
                if self._confirm_identity(label, conf):
                    code = self.label_map[label]
                    meta = self.students.get(code, {"title":"", "name":""})
                    title, name = meta.get("title",""), meta.get("name","")
                    now = time.time()
                    last = self.last_seen_ts.get(label, 0)
                    remain = REPEAT_COOLDOWN_SECONDS - (now - last)

                    if remain > 0:
                        text = f"โปรดรอ {int(remain)} วินาทีก่อนสแกนซ้ำของ {title} {name}"
                        color = (40,140,220)
                    else:
                        if self.confirm_label != label:
                            speaker.ding()
                            speaker.say_async(f"เช็คชื่อเรียบร้อยแล้ว {title} {name}")
                            write_attendance_db(code)
                            self.last_seen_ts[label] = now
                            self.confirm_label = label
                        text = f"เช็คชื่อเรียบร้อยแล้ว {title} {name}"
                        color = (60,200,80)
                else:
                    text = "กำลังตรวจสอบใบหน้า…"
                    color = (200,170,60)

            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame, f"conf:{conf:.1f}", (x,y+h+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230,230,230), 1, cv2.LINE_AA)
        else:
            text = "หันหน้าเข้าหากล้อง / เหลือ 1 คน"

        self.overlay_var.set(text)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb,(520,360))
        img = Image.fromarray(rgb)
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0,0,anchor="nw",image=self.tk_img)
        self.after(10, self.update_frame)

# -------------------- main --------------------
if __name__ == "__main__":
    app = App()
    app.mainloop()
