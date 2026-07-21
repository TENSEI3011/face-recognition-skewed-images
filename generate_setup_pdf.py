"""
generate_setup_pdf.py
Generates a professional Setup Guide PDF for the Face Recognition UAV project.
Uses only fpdf2 built-in fonts (no external font download needed).
Run: python generate_setup_pdf.py
"""

from fpdf import FPDF
from fpdf.enums import XPos, YPos

# Colors
NAVY      = (15,  30, 100)
BLUE      = (28,  78, 183)
LIGHT_BG  = (235, 240, 255)
GREEN     = (22, 140,  70)
GREEN_BG  = (220, 245, 230)
ORANGE    = (190,  90,   0)
ORANGE_BG = (255, 240, 215)
RED       = (185,  30,  30)
RED_BG    = (255, 225, 225)
WHITE     = (255, 255, 255)
DARK      = ( 20,  20,  40)
GREY      = (110, 110, 130)
LGREY     = (245, 246, 250)


def clean(s):
    """Remove characters outside latin-1 range for Helvetica compatibility."""
    return s.encode("latin-1", errors="replace").decode("latin-1")


class PDF(FPDF):

    def __init__(self):
        super().__init__("P", "mm", "A4")
        self.set_auto_page_break(auto=True, margin=22)
        self.set_margins(18, 16, 18)
        self._cover_page = False

    def fc(self, rgb): self.set_fill_color(*rgb)
    def tc(self, rgb): self.set_text_color(*rgb)
    def dc(self, rgb): self.set_draw_color(*rgb)

    def header(self):
        if self._cover_page:
            return
        self.fc(NAVY)
        self.rect(0, 0, 210, 10, "F")
        self.set_font("Helvetica", "B", 7)
        self.tc(WHITE)
        self.set_y(2)
        self.cell(0, 6, "FACE RECOGNITION UAV PROJECT  -  SETUP GUIDE", align="C")
        self.tc(DARK)
        self.ln(9)

    def footer(self):
        if self._cover_page:
            return
        self.set_y(-13)
        self.fc(LGREY)
        self.rect(0, self.get_y() - 1, 210, 16, "F")
        self.set_font("Helvetica", "", 7)
        self.tc(GREY)
        self.cell(0, 9,
            f"Page {self.page - 1}  |  Face Recognition UAV Project  "
            f"|  github.com/TENSEI3011/face-recognition-skewed-images",
            align="C")
        self.tc(DARK)

    def cover(self):
        self._cover_page = True
        self.add_page()
        self.fc(NAVY); self.rect(0, 0, 210, 297, "F")
        self.fc((28, 55, 160)); self.rect(0, 0, 210, 6, "F")

        self.set_font("Helvetica", "B", 10); self.tc((130, 160, 255))
        self.set_y(22)
        self.cell(0, 8, "[ FACE RECOGNITION ON SKEWED UAV IMAGES ]",
                  align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.ln(10)
        self.set_font("Helvetica", "B", 32); self.tc(WHITE)
        self.cell(0, 15, "SETUP GUIDE",
                  align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.set_font("Helvetica", "B", 14); self.tc((160, 190, 255))
        self.cell(0, 9, "Clone, Configure and Run on Any Windows Laptop",
                  align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.ln(6)
        self.set_font("Helvetica", "", 10); self.tc((190, 210, 255))
        self.set_x(30)
        self.multi_cell(150, 6,
            "Step-by-step instructions written so that even someone who has\n"
            "never seen this project before can get it running successfully.\n"
            "Follow every step in order. Do not skip any step.",
            align="C")

        self.ln(16)
        boxes = [
            ("PLATFORM", "Windows 10 / 11"),
            ("PYTHON",   "3.10 or 3.11"),
            ("FRAMEWORK","FastAPI + InsightFace"),
        ]
        bx, bw, by = 20, 52, self.get_y()
        for lbl, val in boxes:
            self.fc((30, 55, 140)); self.dc(BLUE)
            self.rect(bx, by, bw, 22, "FD")
            self.set_font("Helvetica", "B", 6.5); self.tc((130, 160, 255))
            self.set_xy(bx, by + 4); self.cell(bw, 5, lbl, align="C")
            self.set_font("Helvetica", "B", 9); self.tc(WHITE)
            self.set_xy(bx, by + 11); self.cell(bw, 7, val, align="C")
            bx += bw + 9

        self.fc((10, 20, 70)); self.rect(0, 262, 210, 35, "F")
        self.set_font("Helvetica", "B", 9); self.tc((130, 160, 255))
        self.set_xy(0, 270)
        self.cell(0, 6, "github.com/TENSEI3011/face-recognition-skewed-images", align="C")
        self.set_font("Helvetica", "", 8); self.tc((100, 130, 200))
        self.set_xy(0, 279)
        self.cell(0, 6, "Do NOT skip any step. Follow from top to bottom.", align="C")
        self._cover_page = False

    def step(self, num, title):
        self.ln(4)
        if self.get_y() > 245:
            self.add_page()
        self.fc(NAVY); self.rect(self.l_margin - 2, self.get_y(), 176, 9, "F")
        self.set_font("Helvetica", "B", 11); self.tc(WHITE)
        self.set_x(self.l_margin)
        self.cell(0, 9, f"  STEP {num}  -  {title.upper()}")
        self.tc(DARK); self.ln(12)

    def sub(self, t):
        self.set_font("Helvetica", "B", 10); self.tc(BLUE)
        self.cell(0, 7, clean(t), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.tc(DARK)

    def body(self, t):
        self.set_font("Helvetica", "", 9.5); self.tc(DARK)
        self.multi_cell(0, 5.5, clean(t))
        self.ln(1)

    def bullet(self, t, indent=4):
        self.set_font("Helvetica", "", 9.5); self.tc(DARK)
        x = self.l_margin + indent
        self.set_x(x)
        self.cell(4, 5.5, "-")
        self.set_x(x + 5)
        self.multi_cell(0, 5.5, clean(t))

    def code(self, lines, label=None):
        if label:
            self.set_font("Helvetica", "B", 7.5); self.tc(GREY)
            self.cell(0, 5, clean(f"  {label}"),
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.fc((22, 27, 58))
        bg_y = self.get_y()
        h = len(lines) * 5.5 + 8
        if bg_y + h > 272:
            self.add_page(); bg_y = self.get_y()
        self.rect(self.l_margin, bg_y, 174, h, "F")
        self.set_font("Courier", "", 8.5); self.tc((130, 220, 160))
        for line in lines:
            self.set_x(self.l_margin + 4)
            self.cell(0, 5.5, clean(line), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.tc(DARK); self.ln(4)

    def box(self, kind, title, text):
        cfg = {
            "tip":   (GREEN,  GREEN_BG,  "[TIP]"),
            "warn":  (ORANGE, ORANGE_BG, "[WARNING]"),
            "error": (RED,    RED_BG,    "[COMMON ERROR]"),
            "info":  (BLUE,   LIGHT_BG,  "[NOTE]"),
        }
        col, bg, tag = cfg.get(kind, cfg["info"])
        text_lines = clean(text).split("\n")
        h = max(18, len(text_lines) * 5.2 + 18)
        y0 = self.get_y()
        if y0 + h > 272:
            self.add_page(); y0 = self.get_y()
        self.fc(bg); self.rect(self.l_margin, y0, 174, h, "F")
        self.dc(col); self.set_line_width(0.9)
        self.line(self.l_margin, y0, self.l_margin, y0 + h)
        self.set_line_width(0.2)
        self.set_xy(self.l_margin + 5, y0 + 3)
        self.set_font("Helvetica", "B", 8.5); self.tc(col)
        self.cell(0, 5, clean(f"{tag}  {title}"),
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_x(self.l_margin + 5)
        self.set_font("Helvetica", "", 9); self.tc(DARK)
        self.multi_cell(165, 5.2, clean(text))
        self.ln(4)

    def table(self, headers, rows, widths):
        self.fc(NAVY); self.tc(WHITE); self.set_font("Helvetica", "B", 8)
        for h, w in zip(headers, widths):
            self.cell(w, 7, f" {clean(h)}", border=0, fill=True)
        self.ln()
        self.set_font("Helvetica", "", 8)
        for i, row in enumerate(rows):
            self.fc(LGREY if i % 2 == 0 else WHITE); self.tc(DARK)
            for c, w in zip(row, widths):
                self.cell(w, 6.5, f" {clean(c)}", border=0, fill=True)
            self.ln()
        self.ln(4)

    def divider(self, color=BLUE, my=3):
        self.dc(color); self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), 210 - self.r_margin, self.get_y())
        self.ln(my)


# =============================================================================
#  BUILD PDF
# =============================================================================

pdf = PDF()

# ---- COVER ------------------------------------------------------------------
pdf.cover()

# ---- PAGE 2: OVERVIEW -------------------------------------------------------
pdf.add_page()

pdf.fc(LIGHT_BG); pdf.rect(pdf.l_margin, pdf.get_y(), 174, 30, "F")
pdf.set_xy(pdf.l_margin + 4, pdf.get_y() + 4)
pdf.set_font("Helvetica", "B", 11); pdf.tc(NAVY)
pdf.cell(0, 6, "What This Guide Covers", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_x(pdf.l_margin + 4); pdf.set_font("Helvetica", "", 9.5); pdf.tc(DARK)
pdf.multi_cell(166, 5.5,
    "This guide walks you through setting up and running the Face Recognition UAV project "
    "on a brand-new Windows laptop. It is written for someone who has never used this project "
    "before. Follow every step in order and do not skip anything.")
pdf.ln(8)

pdf.sub("What the Laptop Needs BEFORE You Start")
pdf.table(
    headers=["Requirement", "Version / Details", "How to Check"],
    rows=[
        ["Python",      "3.10 or 3.11  (NOT 3.12 or 3.13)", "Open CMD: python --version"],
        ["Git",         "Any version",                        "Open CMD: git --version"],
        ["Internet",    "Required for first setup",           "Or transfer files via USB"],
        ["RAM",         "Minimum 8 GB",                       "Task Manager > Performance"],
        ["Disk Space",  "Minimum 5 GB free",                  "Windows Explorer"],
        ["Browser",     "Chrome or Edge",                     "Already installed on Windows"],
    ],
    widths=[34, 78, 62]
)

pdf.box("warn", "Python Version is Critical",
    "Python 3.13 causes dlib-bin to FAIL during installation.\n"
    "Only Python 3.10 or 3.11 are guaranteed to work with all dependencies.\n"
    "If the laptop already has Python 3.13, install Python 3.11 from python.org alongside it.")

pdf.box("info", "Fully Offline After First Setup",
    "Run  python offline_setup.py  once on any machine with internet.\n"
    "It downloads all models and fonts. After that, the system runs with ZERO internet.\n"
    "For air-gapped / defence deployment: copy the full project folder + ~/.insightface/models/buffalo_l/ via USB.")

# ---- PAGE 3: STEPS 1-3 -----------------------------------------------------
pdf.add_page()

pdf.step(1, "Install Python 3.11")
pdf.body("1. Open Chrome or Edge on the scientist laptop.")
pdf.body("2. Go to:  https://www.python.org/downloads/release/python-3119/")
pdf.body("3. Scroll down to the Files section. Click: Windows installer (64-bit)")
pdf.body("4. Run the downloaded .exe file. On the FIRST installer screen:")
pdf.set_font("Helvetica", "B", 10); pdf.tc(RED)
pdf.cell(0, 7,
    "   --> TICK THE CHECKBOX:  'Add Python.exe to PATH'  <-- Do NOT miss this!",
    new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.tc(DARK)
pdf.body("5. Click 'Install Now'. Wait for it to complete. Click Close.")
pdf.ln(2)
pdf.sub("Verify Python installed correctly:")
pdf.body("Press Windows key, type cmd, press Enter to open Command Prompt. Type:")
pdf.code(["python --version"], label="Command Prompt")
pdf.body("You should see:   Python 3.11.9   (or any 3.11.x number)")
pdf.box("error", "python is not recognized as an internal or external command",
    "Python was not added to PATH during installation.\n"
    "Uninstall Python (Windows Settings > Apps), re-run the installer,\n"
    "and make sure you tick 'Add Python.exe to PATH' on the first screen.")

pdf.step(2, "Install Git")
pdf.body("1. Open browser and go to:  https://git-scm.com/download/win")
pdf.body("2. Click '64-bit Git for Windows Setup' to download.")
pdf.body("3. Run the installer. Click Next on every screen (all defaults are fine).")
pdf.body("4. Click Install, then Finish.")
pdf.ln(2)
pdf.sub("Verify Git is installed:")
pdf.code(["git --version"], label="Command Prompt")
pdf.body("You should see:   git version 2.x.x.windows.x")

pdf.step(3, "Clone the GitHub Repository")
pdf.body("Open Command Prompt (Windows key > type cmd > Enter). Type these commands:")
pdf.code([
    "cd Desktop",
    "git clone https://github.com/TENSEI3011/face-recognition-skewed-images.git",
    "cd face-recognition-skewed-images",
], label="Command Prompt - type each line and press Enter")
pdf.body("A folder called 'face-recognition-skewed-images' will appear on the Desktop.")
pdf.box("tip", "Slow Internet or No Internet?",
    "Copy the entire 'face-recognition-skewed-images' folder from Lucky's laptop\n"
    "to the USB drive and paste it on the scientist's Desktop.\n"
    "Then open CMD and type:  cd Desktop\\face-recognition-skewed-images")

# ---- PAGE 4: STEPS 4-5 -----------------------------------------------------
pdf.add_page()

pdf.step(4, "Create a Virtual Environment")
pdf.body("A virtual environment is an isolated Python workspace. "
         "Make sure you are inside the project folder (from Step 3). Run:")
pdf.code([
    "python -m venv venv",
    "",
    "venv\\Scripts\\activate",
], label="Command Prompt - run inside the project folder")
pdf.body("After the second command, you will see (venv) at the start of the prompt:")
pdf.code(["(venv) C:\\Users\\...\\face-recognition-skewed-images>"],
         label="What the prompt looks like when venv is active")
pdf.body("(venv) MUST be visible before you continue to the next step.")
pdf.box("warn", "Always activate (venv) every time you open a new CMD window",
    "Every time you close and reopen Command Prompt, you must run\n"
    "venv\\Scripts\\activate  again before running any python commands.\n"
    "If you forget, you will see ModuleNotFoundError errors.")
pdf.box("error", "venv\\Scripts\\activate gives an execution policy error",
    "This can happen in PowerShell. Run this command once:\n"
    "  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned\n"
    "Press Y then Enter. Then try activate again.\n"
    "Or use Command Prompt (cmd.exe) instead of PowerShell - it has no such restriction.")

pdf.step(5, "Install All Python Packages")
pdf.body("Make sure (venv) is showing in your prompt. Then run these two commands:")
pdf.code([
    "pip install --upgrade pip",
    "pip install -r requirements.txt",
], label="Command Prompt (venv must be active)")
pdf.body("This downloads and installs everything: ArcFace, FAISS, OpenCV, scikit-learn,\n"
         "FastAPI, dlib, and many more. It takes 10 to 20 minutes on first run.\n"
         "Hundreds of lines of output are completely normal. Wait for 'Successfully installed'.")
pdf.box("error", "dlib-bin install failed - Microsoft Visual C++ 14.0 is required",
    "Run these commands in order:\n"
    "  pip install cmake\n"
    "  pip install dlib\n"
    "Then try again:  pip install -r requirements.txt")
pdf.box("error", "ERROR: No matching distribution found (Python 3.12 or 3.13)",
    "Your Python version is incompatible. Check: python --version\n"
    "If it shows 3.12 or 3.13, install Python 3.11 from python.org (Step 1).\n"
    "Then delete the venv folder and run:\n"
    "  py -3.11 -m venv venv\n"
    "  venv\\Scripts\\activate\n"
    "  pip install -r requirements.txt")

# ---- PAGE 5: USB TRANSFER ---------------------------------------------------
pdf.add_page()

pdf.step(6, "Download Models and Fonts (One-Time Internet Setup)")
pdf.body(
    "Run this single command to download EVERYTHING needed for offline operation: "
    "dlib landmark model, InsightFace buffalo_l (ArcFace + SCRFD), and all 9 UI fonts.")
pdf.code(["python offline_setup.py"],
         label="Command Prompt (venv active, internet required)")
pdf.body("This downloads and caches:\n"
         "  - dlib 68-point landmark model (~100 MB) -> models/ in project folder\n"
         "  - InsightFace buffalo_l model (~500 MB) -> C:\\Users\\<name>\\.insightface\\models\\buffalo_l\\\n"
         "  - All 9 UI fonts (.ttf) -> web/frontend/fonts/  (no more internet needed for UI)\n"
         "After this completes, the system runs FULLY OFFLINE.")
pdf.box("tip", "Slow Internet or No Internet? Copy via USB Instead",
    "Prepare these items on Lucky's laptop and copy to USB:\n"
    "  1. models\\shape_predictor_68_face_landmarks.dat (~100 MB)\n"
    "  2. C:\\Users\\lucky\\.insightface\\models\\buffalo_l\\ (~500 MB)\n"
    "  3. web\\frontend\\fonts\\ (all .ttf files, ~1.5 MB)\n"
    "  4. data\\gallery\\ (enrolled identity photos)\n"
    "  5. results\\baseline\\models\\ (trained SVM + FAISS)\n"
    "Paste each to the same relative location on the scientist laptop.")

pdf.sub("How to Copy the buffalo_l InsightFace Folder (Most Important):")
pdf.bullet("On Lucky's laptop, open File Explorer")
pdf.bullet("In the top address bar, type this path and press Enter:")
pdf.code(["C:\\Users\\lucky\\.insightface\\models\\"],
         label="Type this in File Explorer address bar")
pdf.bullet("If you cannot see the .insightface folder: click View tab > tick 'Hidden items'")
pdf.bullet("Copy the entire 'buffalo_l' folder to your USB drive")
pdf.bullet("On the scientist's laptop, open File Explorer and navigate to C:\\Users\\<name>\\")
pdf.bullet("Create a new folder '.insightface', inside it create a folder 'models'")
pdf.bullet("Paste the 'buffalo_l' folder into:  C:\\Users\\<name>\\.insightface\\models\\")

# ---- PAGE 6: STEPS 7-8 -----------------------------------------------------
pdf.add_page()

pdf.step(7, "Create the .env Configuration File")
pdf.body("The .env file stores the secret key and database connection. Run:")
pdf.code(["copy .env.example .env"],
         label="Command Prompt (inside the project folder)")
pdf.body("Then open the file in Notepad to edit it:")
pdf.code(["notepad .env"], label="Command Prompt")
pdf.body("The file will open in Notepad. Edit it to look like this "
         "(choose ONE MongoDB option):")
pdf.code([
    "JWT_SECRET_KEY=face-recognition-uav-secret-key-2025",
    "",
    "# OPTION A: Local MongoDB (recommended for offline / defence use)",
    "MONGO_URI=mongodb://localhost:27017",
    "",
    "# OPTION B: MongoDB Atlas (cloud - remove # to use)",
    "#MONGO_URI=mongodb+srv://user:password@cluster.mongodb.net/",
    "",
    "# OPTION C: No MongoDB (local disk only - comment out MONGO_URI)",
    "#MONGO_URI=",
    "",
    "MONGO_DB_NAME=facerecog_db",
    "ENV=development",
    "",
    "# --- Passive Anti-Spoofing (6-signal fusion) ---",
    "LIVENESS_ENABLED=True       # Set False to disable the check",
    "LIVENESS_THRESHOLD=0.45     # 0.0-1.0: lower = more permissive",
    "",
    "# --- Active Blink Challenge (webcam / live mode only) ---",
    "BLINK_ENABLED=True          # Set False to skip blink challenge",
    "BLINK_EAR_THRESHOLD=0.25    # Eye Aspect Ratio below this = blink",
    "BLINK_TIMEOUT_SEC=7.0       # Seconds user has to blink",
    "BLINK_REQUIRED_COUNT=1      # Number of blinks required",
    "BLINK_CONSEC_FRAMES=2       # Consecutive frames below EAR = 1 blink",
    "",
    "# --- Recognition Threshold ---",
    "FAISS_THRESHOLD=0.35        # Cosine similarity threshold (0.35 tuned for UAV)",
], label=".env file - save with Ctrl+S after editing")
pdf.body("Save the file (Ctrl+S) and close Notepad.")
pdf.box("tip", "Recommended for Defence / Offline Use: Local MongoDB",
    "Install MongoDB Community Server (free) from mongodb.com/try/download/community\n"
    "After installing, start it with:  mongod --dbpath C:\\data\\db\n"
    "Then set: MONGO_URI=mongodb://localhost:27017 in your .env file.\n"
    "This gives you full audit log, watchlist, and gallery features with NO internet.")
pdf.box("tip", "No MongoDB at All? Use Disk-Only Mode",
    "Remove the MONGO_URI line from .env entirely.\n"
    "Gallery images are stored as files in data/gallery/.\n"
    "Audit log and watchlist are unavailable, but identification still works.")

pdf.step(8, "Start the Web Server")
pdf.body("Make sure (venv) is active and you are in the project folder. Run this command:")
pdf.code([
    "python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload"
], label="Command Prompt - keep this window open while using the system")
pdf.body("Wait 20 to 30 seconds. If everything is set up correctly, you will see:")
pdf.code([
    "INFO:     Uvicorn running on http://0.0.0.0:8000",
    "[FaceDetector] SCRFD (buffalo_l) loaded successfully.",
    "[ArcFaceExtractor] Loaded model: buffalo_l",
    "[SVMClassifier] Loaded from results/baseline/models/svm_classifier.pkl",
    "[FAISSMatcher] Loaded: 75 vectors, 3 identities, threshold=0.35",
    "[LivenessService] Passive liveness detector initialised (threshold=0.45, 6-signal fusion)",
    "[LivenessService] BlinkDetector initialised -- status: OK",
    "INFO:     Application startup complete.",
], label="Expected output when everything is working correctly")
pdf.body("Now open Chrome or Edge and go to this address:")
pdf.code(["http://localhost:8000"], label="Browser address bar")
pdf.body("The Face Recognition dashboard will appear in the browser. The system is ready.")
pdf.box("error", "Website shows 'Pipeline not loaded'",
    "The trained model files are missing.\n"
    "Solution A: Copy results\\baseline\\models\\ from Lucky's laptop via USB (Step 6).\n"
    "Solution B: Run this command to retrain from scratch (takes 20-30 mins):\n"
    "  python experiments\\run_baseline.py\n"
    "(Make sure gallery photos are in data\\gallery\\ before running this.)")

# ---- PAGE 7: RETRAIN + USING ------------------------------------------------
pdf.add_page()

pdf.step(9, "Enroll Identities and Retrain")
pdf.body("SKIP THIS STEP if you already copied results\\baseline\\models\\ via USB in Step 6.")
pdf.body("If models are missing OR you want to add new people, use the Gallery page:")
pdf.sub("Option A: Enroll via Web UI (recommended):")
pdf.bullet("Open http://localhost:8000/gallery in your browser")
pdf.bullet("Enter the person's name and click 'Upload Images' or drag a video file")
pdf.bullet("Supported formats: JPG, PNG, BMP (images) and MP4, MOV, AVI (video)")
pdf.bullet("For video: frames are sampled every 15 frames; blurry frames are skipped automatically")
pdf.bullet("Model retraining starts AUTOMATICALLY after each upload (no manual click needed)")
pdf.bullet("First retrain takes ~15 seconds. After that: ~2 seconds thanks to embedding cache")
pdf.ln(2)
pdf.sub("Option B: Retrain with GridSearchCV (best accuracy, takes ~5 minutes extra):")
pdf.body("After your gallery is complete, run a GridSearch retrain to find the best SVM settings:")
pdf.code(["# Via browser at http://localhost:8000/docs -> POST /api/pipeline/retrain",
          "# Set use_grid_search=true in the request body",
          "",
          "# OR via command line:",
          'curl -X POST "http://localhost:8000/api/pipeline/retrain?use_grid_search=true"'],
         label="GridSearchCV retrain - improves Rank-1 accuracy by 5-15%")
pdf.ln(2)
pdf.sub("Option C: Full Baseline Experiment (most thorough, 20-30 minutes):")
pdf.code(["python experiments\\run_baseline.py"],
         label="Command Prompt (venv active) - trains SVM, FAISS, and generates accuracy reports")
pdf.box("warn", "Gallery Photos Must Be Present for Retraining",
    "data\\gallery\\ must contain identity subfolders with photos.\n"
    "Copy from Lucky's laptop if empty. Use at least 10-30 photos per person for best accuracy.\n"
    "Blurry / dark images are automatically rejected during enrollment.")

pdf.step(10, "Using the System")
pdf.table(
    headers=["Page", "URL", "What It Does"],
    rows=[
        ["Dashboard",  "http://localhost:8000/",          "System status and metrics overview"],
        ["Identify",   "http://localhost:8000/identify",  "Upload photo - passive liveness (6-signal) checked automatically"],
        ["Gallery",    "http://localhost:8000/gallery",   "Enroll via images OR video files (MP4/MOV/AVI) - auto-retrains"],
        ["Demo",       "http://localhost:8000/demo",      "Upload video or webcam - blink challenge active in webcam mode"],
        ["Results",    "http://localhost:8000/results",   "Experiment accuracy graphs (CMC, ROC)"],
        ["Analytics",  "http://localhost:8000/analytics", "System usage stats"],
        ["Audit Log",  "http://localhost:8000/audit",     "All identification events (incl. spoof / blink-fail detections)"],
        ["Watchlist",  "http://localhost:8000/watchlist", "Alert watchlist management"],
        ["Config",     "http://localhost:8000/config",    "Threshold, pipeline settings, anti-spoofing + blink toggle"],
        ["API Docs",   "http://localhost:8000/docs",      "Full REST API reference"],
    ],
    widths=[26, 68, 80]
)
pdf.sub("Identify a Face in a Photo:")
pdf.bullet("Go to http://localhost:8000/identify in your browser")
pdf.bullet("Click 'Choose File' and select any .jpg or .png photo containing a face")
pdf.bullet("Click 'Identify'")
pdf.bullet("The system shows the person's name, confidence score, and a box around the face")
pdf.ln(3)
pdf.sub("Enroll a New Person from Video:")
pdf.bullet("Go to http://localhost:8000/gallery")
pdf.bullet("Enter the person's name and upload a 10-30 second MP4 or MOV video clip")
pdf.bullet("Set frame interval to 15 (default) — captures ~2 faces per second")
pdf.bullet("Click Upload. Only sharp, clear frames are stored (blurry ones are rejected)")
pdf.bullet("Click 'Retrain from Gallery' to rebuild the model with the new identity")
pdf.ln(3)
pdf.sub("Process a Video File:")
pdf.bullet("Go to http://localhost:8000/demo")
pdf.bullet("Click the 'Upload Video' tab")
pdf.bullet("Click to upload and select a .mp4 video")
pdf.bullet("Adjust 'Process every N frames' slider (default 6 = fast; lower = more accurate but slower)")
pdf.bullet("Click 'Process Video' and wait for processing")
pdf.bullet("Download the output video with face names labelled on each detected person")
pdf.ln(3)
pdf.sub("To Stop the Server:")
pdf.body("Go to the Command Prompt window where uvicorn is running and press Ctrl+C.")

# ---- PAGE 8: TROUBLESHOOTING ------------------------------------------------
pdf.add_page()

pdf.fc(LIGHT_BG); pdf.rect(pdf.l_margin, pdf.get_y(), 174, 9, "F")
pdf.set_font("Helvetica", "B", 12); pdf.tc(NAVY); pdf.set_x(pdf.l_margin + 3)
pdf.cell(0, 9, "  TROUBLESHOOTING  -  Common Errors and Their Fixes")
pdf.tc(DARK); pdf.ln(13)

errors = [
    ("python is not recognized as an internal or external command",
     "Python is not installed or not in PATH.\n"
     "Re-install Python 3.11 from python.org.\n"
     "On the first installer screen, tick 'Add Python.exe to PATH' before clicking Install."),
    ("No module named 'web'",
     "You are running the uvicorn command from the wrong folder.\n"
     "Make sure you first cd into the project folder:\n"
     "  cd Desktop\\face-recognition-skewed-images\n"
     "Then run the uvicorn command again."),
    ("Website shows 'Pipeline not loaded'",
     "The trained SVM and FAISS model files are missing.\n"
     "Copy results\\baseline\\models\\ from Lucky's laptop via USB (Step 6).\n"
     "OR run: python experiments\\run_baseline.py to retrain (needs gallery photos)."),
    ("shape_predictor_68_face_landmarks.dat not found",
     "The dlib facial landmark model file is missing.\n"
     "Copy models\\shape_predictor_68_face_landmarks.dat from Lucky's laptop via USB."),
    ("InsightFace is downloading models very slowly at startup",
     "The buffalo_l model folder is not in the right location.\n"
     "Copy the buffalo_l folder from Lucky's laptop (Step 6, item 2).\n"
     "Place it at: C:\\Users\\<scientist-name>\\.insightface\\models\\buffalo_l\\"),
    ("Port 8000 is already in use",
     "Another program is using port 8000. Use port 8001 instead:\n"
     "  python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8001 --reload\n"
     "Then open http://localhost:8001 in the browser."),
    ("dlib install failed: Microsoft Visual C++ 14.0 is required",
     "Install Microsoft C++ Build Tools:\n"
     "  1. Search 'Visual Studio Build Tools' download in browser\n"
     "  2. Download and run vs_BuildTools.exe\n"
     "  3. Select 'Desktop development with C++' and click Install\n"
     "  4. After it finishes, run: pip install cmake && pip install dlib\n"
     "  5. Then: pip install -r requirements.txt"),
    ("PowerShell: venv\\Scripts\\activate gives execution policy error",
     "Run this in PowerShell first:\n"
     "  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned\n"
     "Press Y then Enter. Then run venv\\Scripts\\activate again.\n"
     "Alternatively, use Command Prompt (cmd.exe) instead of PowerShell."),
]

for title, text in errors:
    pdf.box("error", title, text)

# ---- PAGE 9: CHEAT SHEET ----------------------------------------------------
pdf.add_page()

pdf.fc(NAVY); pdf.rect(pdf.l_margin, pdf.get_y(), 174, 10, "F")
pdf.set_font("Helvetica", "B", 12); pdf.tc(WHITE); pdf.set_x(pdf.l_margin + 3)
pdf.cell(0, 10, "  QUICK START CHEAT SHEET  -  Print This Page and Keep It Handy")
pdf.tc(DARK); pdf.ln(14)

pdf.sub("Every Time You Open a New Terminal Window, Run These 3 Commands:")
pdf.code([
    "cd Desktop\\face-recognition-skewed-images",
    "venv\\Scripts\\activate",
    "python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload",
], label="3 commands to start the server")
pdf.body("Then open Chrome and go to:  http://localhost:8000")
pdf.divider()

pdf.sub("All One-Time First-Day Setup Commands (Run Once Only):")
pdf.code([
    "# 1. Clone the repository",
    "cd Desktop",
    "git clone https://github.com/TENSEI3011/face-recognition-skewed-images.git",
    "cd face-recognition-skewed-images",
    "",
    "# 2. Create virtual environment",
    "python -m venv venv",
    "venv\\Scripts\\activate",
    "",
    "# 3. Install all packages (takes 10-20 minutes)",
    "pip install --upgrade pip",
    "pip install -r requirements.txt",
    "",
    "# 4. Download ALL models + fonts (one-time internet step)",
    "python offline_setup.py",
    "",
    "# 5. Create configuration file",
    "copy .env.example .env",
    "notepad .env    <- set MONGO_URI=mongodb://localhost:27017",
    "",
    "# 6. Start the server",
    "python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload",
], label="Complete first-time setup from scratch")

pdf.divider()
pdf.sub("USB Files Checklist - Prepare These Before Going (if no internet on target machine):")
pdf.table(
    headers=["Item", "File / Folder to Copy", "Destination on New Laptop", "Done?"],
    rows=[
        ["1", "models\\shape_predictor_68_face_landmarks.dat",
         "models\\ in project folder", "[ ]"],
        ["2", "C:\\Users\\lucky\\.insightface\\models\\buffalo_l\\",
         "C:\\Users\\<name>\\.insightface\\models\\buffalo_l\\", "[ ]"],
        ["3", "web\\frontend\\fonts\\  (all .ttf files)",
         "web\\frontend\\fonts\\ in project folder", "[ ]"],
        ["4", "data\\gallery\\ (all identity subfolders)",
         "data\\ in project folder", "[ ]"],
        ["5", "results\\baseline\\models\\ (SVM + FAISS)",
         "results\\baseline\\ in project folder", "[ ]"],
    ],
    widths=[10, 72, 72, 20]
)

# ---- SAVE -------------------------------------------------------------------
output = r"c:\Users\lucky\OneDrive\Desktop\Face Recognition\Setup_Guide_Face_Recognition.pdf"
pdf.output(output)
print(f"\nPDF generated successfully!")
print(f"Saved to: {output}")
print(f"Total pages: {pdf.page}")
