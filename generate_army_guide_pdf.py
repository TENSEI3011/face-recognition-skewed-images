"""
generate_army_guide_pdf.py
Army / Defence Offline Deployment Guide  -  clean, natural, easy to follow.
Run: python generate_army_guide_pdf.py
"""

from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os

OLIVE  = (55,  75, 20)
GREEN  = (0,  120, 60)
RED    = (180, 30, 30)
AMBER  = (175, 88,  0)
WHITE  = (255, 255, 255)
DARK   = (28,  30, 38)
MID    = (90,  95, 105)
LIGHT  = (247, 248, 245)
RULE   = (200, 208, 190)


def c(s):
    return str(s).encode("latin-1", errors="replace").decode("latin-1")


class PDF(FPDF):

    def __init__(self):
        super().__init__("P", "mm", "A4")
        self.set_auto_page_break(auto=True, margin=24)
        self.set_margins(20, 18, 20)
        self._on_cover = False

    def header(self):
        if self._on_cover:
            return
        self.set_fill_color(*OLIVE)
        self.rect(0, 0, 210, 8, "F")
        self.set_font("Helvetica", "B", 6.5)
        self.set_text_color(*WHITE)
        self.set_y(1.5)
        self.cell(0, 5, "Face Recognition UAV Project   -   Offline Deployment Guide   -   RESTRICTED", align="C")
        self.set_text_color(*DARK)
        self.ln(10)

    def footer(self):
        if self._on_cover:
            return
        self.set_y(-12)
        self.set_draw_color(*RULE)
        self.set_line_width(0.2)
        self.line(20, self.get_y(), 190, self.get_y())
        self.ln(2)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*MID)
        self.cell(0, 5, f"Page {self.page - 1}   |   Face Recognition UAV Project   -   Army Offline Deployment", align="C")
        self.set_text_color(*DARK)

    def cover(self):
        self._on_cover = True
        self.add_page()
        self.set_fill_color(*OLIVE)
        self.rect(0, 0, 210, 297, "F")
        self.set_fill_color(40, 60, 12)
        self.rect(0, 0, 210, 5, "F")

        self.set_y(30)
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(170, 205, 110)
        self.cell(0, 6, "FACE RECOGNITION ON SKEWED UAV IMAGES", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.ln(6)
        self.set_font("Helvetica", "B", 32)
        self.set_text_color(*WHITE)
        self.cell(0, 15, "Deployment Guide", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(190, 220, 140)
        self.cell(0, 8, "Offline / Air-Gapped Installation", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.ln(8)
        self.set_x(28)
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(195, 220, 160)
        self.multi_cell(154, 6,
            "Step-by-step guide for setting up the face recognition system\n"
            "on a laptop with little or no internet access.\n"
            "Covers everything from installation to field operation.",
            align="C")

        self.ln(10)
        chips = ["100% Offline", "Anti-Spoof", "FAISS Matching", "Temporal Voting"]
        cx = 22
        for chip in chips:
            cw = 42
            self.set_fill_color(40, 65, 15)
            self.rect(cx, self.get_y(), cw, 10, "F")
            self.set_font("Helvetica", "", 7.5)
            self.set_text_color(*WHITE)
            self.set_xy(cx, self.get_y() + 1.5)
            self.cell(cw, 7, c(chip), align="C")
            cx += cw + 4

        self.set_fill_color(28, 45, 8)
        self.rect(0, 268, 210, 29, "F")
        self.set_y(274)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(170, 205, 110)
        self.cell(0, 6, "github.com/TENSEI3011/face-recognition-skewed-images", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(140, 170, 100)
        self.cell(0, 6, "Follow every step in order. Do not skip.", align="C")
        self._on_cover = False

    def h1(self, num, title):
        self.ln(5)
        if self.get_y() > 248:
            self.add_page()
        self.set_fill_color(*OLIVE)
        self.rect(self.l_margin - 2, self.get_y(), 174, 10, "F")
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*WHITE)
        self.set_x(self.l_margin + 2)
        self.cell(0, 10, c(f"Step {num}   -   {title}"))
        self.set_text_color(*DARK)
        self.ln(13)

    def h2(self, text):
        if self.get_y() > 260:
            self.add_page()
        self.ln(2)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*GREEN)
        self.cell(0, 7, c(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*DARK)

    def para(self, text):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*DARK)
        self.multi_cell(0, 5.8, c(text))
        self.ln(2)

    def bullet(self, text, indent=5):
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(*DARK)
        x = self.l_margin + indent
        self.set_x(x)
        self.cell(4, 5.8, "-")
        self.set_x(x + 5)
        self.multi_cell(0, 5.8, c(text))

    def code(self, lines, note=None):
        if note:
            self.set_font("Helvetica", "I", 7.5)
            self.set_text_color(*MID)
            self.cell(0, 5, c(f"   {note}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        h = len(lines) * 5.5 + 8
        y0 = self.get_y()
        if y0 + h > 272:
            self.add_page()
            y0 = self.get_y()
        self.set_fill_color(22, 28, 18)
        self.rect(self.l_margin, y0, 170, h, "F")
        self.set_font("Courier", "", 8.5)
        self.set_text_color(140, 210, 110)
        for line in lines:
            self.set_x(self.l_margin + 5)
            self.cell(0, 5.5, c(line), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*DARK)
        self.ln(4)

    def note(self, kind, text):
        cfg = {
            "tip":  (GREEN,  (230, 248, 235), ">> "),
            "warn": (AMBER,  (255, 246, 220), "!! "),
            "err":  (RED,    (255, 233, 233), "XX "),
        }
        col, bg, icon = cfg.get(kind, cfg["tip"])
        lines = c(text).split("\n")
        h = max(16, len(lines) * 5.3 + 14)
        y0 = self.get_y()
        if y0 + h > 272:
            self.add_page()
            y0 = self.get_y()
        self.set_fill_color(*bg)
        self.rect(self.l_margin, y0, 170, h, "F")
        self.set_draw_color(*col)
        self.set_line_width(0.8)
        self.line(self.l_margin, y0, self.l_margin, y0 + h)
        self.set_line_width(0.2)
        self.set_xy(self.l_margin + 5, y0 + 4)
        self.set_font("Helvetica", "B", 8.5)
        self.set_text_color(*col)
        self.cell(5, 5, c(icon))
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*DARK)
        self.multi_cell(157, 5.3, c(text))
        self.ln(4)

    def rule(self):
        self.set_draw_color(*RULE)
        self.set_line_width(0.2)
        self.line(self.l_margin, self.get_y(), 190, self.get_y())
        self.ln(5)

    def table(self, headers, rows, widths):
        self.set_fill_color(*OLIVE)
        self.set_text_color(*WHITE)
        self.set_font("Helvetica", "B", 8)
        for h, w in zip(headers, widths):
            self.cell(w, 7, f"  {c(h)}", fill=True)
        self.ln()
        self.set_font("Helvetica", "", 8)
        for i, row in enumerate(rows):
            bg = LIGHT if i % 2 == 0 else WHITE
            self.set_fill_color(*bg)
            self.set_text_color(*DARK)
            for val, w in zip(row, widths):
                self.cell(w, 6.5, f"  {c(val)}", fill=True)
            self.ln()
        self.ln(4)

    def page_title(self, text):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*OLIVE)
        self.cell(0, 10, c(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*GREEN)
        self.set_line_width(0.8)
        self.line(self.l_margin, self.get_y(), 190, self.get_y())
        self.set_line_width(0.2)
        self.set_text_color(*DARK)
        self.ln(6)


# ══════════════════════════════════════════════════════════════════════════════
pdf = PDF()
pdf.cover()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2  -  Overview
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.page_title("Overview")

pdf.para(
    "This guide sets up the Face Recognition UAV system on a deployment laptop, "
    "which may have limited or no internet access. "
    "The project code, trained models, and enrolled person photos are all on GitHub  -  "
    "so a git clone handles most of the work. "
    "Only two large AI model files (~600 MB total) may need to be copied via USB "
    "if internet isn't available."
)

pdf.h2("What the deployment laptop needs")
pdf.table(
    ["Requirement", "Details"],
    [
        ["Operating system", "Windows 10 or Windows 11 (64-bit)"],
        ["Python", "Version 3.10 or 3.11  -  NOT 3.12 or 3.13"],
        ["RAM", "Minimum 8 GB (16 GB recommended)"],
        ["Free disk space", "At least 5 GB"],
        ["CPU", "Intel i5 / Ryzen 5 or better  -  no GPU needed"],
        ["Webcam", "Optional  -  only for live camera mode"],
    ],
    [50, 120]
)

pdf.h2("What this guide covers")
pdf.bullet("Installing Python, Git, and all packages")
pdf.bullet("Copying or downloading the two large AI model files")
pdf.bullet("Setting up MongoDB for audit logs and watchlists")
pdf.bullet("Creating the configuration file")
pdf.bullet("Starting the server and using the system")
pdf.bullet("Firewall settings for restricted networks")
pdf.bullet("Enrolling new people in the field")
pdf.bullet("Adjusting liveness and recognition settings")
pdf.bullet("Troubleshooting common problems")

pdf.h2("Current system features")
pdf.bullet("Passive liveness detection  -  rejects printed photos and screen replays in ~3ms")
pdf.bullet("FAISS open-set matching  -  unknown people show as UNKNOWN, never wrongly matched")
pdf.bullet("Test-Time Augmentation  -  5-variant embedding average, better accuracy on UAV footage")
pdf.bullet("Temporal voting  -  smooths out flickering identities in video")
pdf.bullet("Active blink challenge  -  live camera asks user to blink, defeats photo spoofing")
pdf.bullet("Auto-retrain  -  adding a new person triggers retraining automatically")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3  -  Preparation
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.page_title("Preparation  -  What to Bring")

pdf.para(
    "Before heading to the deployment site, decide which setup method applies:"
)

pdf.h2("Method A  -  Deployment laptop has internet access  (easiest)")
pdf.para(
    "If the laptop can reach the internet, even briefly, this is all you need on USB:"
)
pdf.bullet("Python 3.11 installer  (python-3.11.9-amd64.exe, ~27 MB)")
pdf.bullet("MongoDB installer  (mongodb-windows-x86_64-7.0.x.msi, ~500 MB)")
pdf.bullet("Git installer  (optional, Git-2.x.x-64-bit.exe)")
pdf.bullet("A .env file with the correct settings (copy from the project folder)")
pdf.para("")
pdf.para(
    "Everything else  -  the code, trained models, gallery photos, and fonts  -  "
    "downloads automatically when you run git clone and python offline_setup.py."
)

pdf.h2("Method B  -  Deployment laptop has NO internet  (air-gapped)")
pdf.para("Copy all of the following onto USB from a laptop that already has the system working:")
pdf.table(
    ["#", "What to copy", "Where to put it on the deployment laptop"],
    [
        ["1", "The entire project folder (face-recognition-skewed-images)",
         "Desktop"],
        ["2", "C:\\Users\\lucky\\.insightface\\models\\buffalo_l\\  (~500 MB)",
         "C:\\Users\\<name>\\.insightface\\models\\buffalo_l\\"],
        ["3", "models\\shape_predictor_68_face_landmarks.dat  (~95 MB)",
         "models\\ folder inside the project"],
        ["4", ".env file  (pre-configured settings)",
         "Rename to .env in the project root"],
        ["5", "Python 3.11 installer", "Run on deployment laptop"],
        ["6", "MongoDB installer", "Run on deployment laptop"],
        ["7", "pip wheel cache  (optional, for offline pip install)",
         "USB:\\wheels\\"],
    ],
    [6, 82, 82]
)

pdf.note("tip",
    "Items already included when you copy the project folder (item 1 above):\n"
    "  - results\\baseline\\models\\  (trained SVM + FAISS + PCA models)\n"
    "  - data\\gallery\\             (all enrolled people: Aditi, Samridhi, Siddhant, Stuti)\n"
    "  - web\\frontend\\fonts\\      (UI fonts for offline use)\n"
    "These are on GitHub and come with the project folder automatically."
)

pdf.note("warn",
    "The .insightface folder is hidden on Windows.\n"
    "In File Explorer, click View → tick 'Hidden items' to see it.\n"
    "Navigate to:  C:\\Users\\lucky\\.insightface\\models\\\n"
    "Copy the entire buffalo_l folder to USB."
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4  -  Steps 1-2
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1(1, "Install Python 3.11")

pdf.para(
    "Python 3.11 must be installed. Only versions 3.10 and 3.11 work  -  "
    "3.12 and 3.13 cause errors during package installation."
)
pdf.para("If the laptop has internet, download from:")
pdf.code(["https://www.python.org/downloads/release/python-3119/"],
         note="Click 'Windows installer (64-bit)'")
pdf.para("If no internet, run the installer from USB.")

pdf.note("err",
    "On the very first installer screen, tick:\n"
    "'Add Python.exe to PATH'\n\n"
    "Without this, Python won't work from Command Prompt."
)
pdf.para("Click 'Install Now'. Wait, then click Close.")

pdf.h2("Verify it worked")
pdf.code(["python --version"], note="Open Command Prompt  -  should show Python 3.11.x")

pdf.note("tip",
    "Laptop already has Python 3.12 or 3.13? Don't uninstall it.\n"
    "Install 3.11 alongside it. In Step 4, use:\n"
    "  py -3.11 -m venv venv\n"
    "This forces the project to use 3.11."
)

pdf.ln(4)
pdf.h1(2, "Get the Project Code")

pdf.h2("Option A  -  With internet  (preferred)")
pdf.code([
    "cd Desktop",
    "git clone https://github.com/TENSEI3011/face-recognition-skewed-images.git",
    "cd face-recognition-skewed-images",
], note="Command Prompt  -  press Enter after each line")
pdf.para(
    "This downloads the code along with the trained recognition models and gallery photos. "
    "The only things not included are the two large AI model files (buffalo_l and dlib), "
    "which you'll download in Step 5."
)

pdf.h2("Option B  -  From USB, no internet")
pdf.para("Copy the project folder from USB to the Desktop, then:")
pdf.code(["cd Desktop\\face-recognition-skewed-images"], note="Command Prompt")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5  -  Steps 3-4
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1(3, "Copy the AI Model Files  (USB only  -  skip if using internet)")

pdf.para(
    "If you're doing an offline USB deployment, you need to copy two files manually. "
    "If the laptop has internet, skip this step  -  Step 5 handles it automatically."
)

pdf.h2("File 1: InsightFace buffalo_l  (~500 MB)")
pdf.para("Create this folder on the deployment laptop if it doesn't exist:")
pdf.code(["C:\\Users\\<your-name>\\.insightface\\models\\"],
         note="In File Explorer  -  enable hidden items first (View → Hidden items)")
pdf.para("Copy from USB:")
pdf.code(["USB:\\insightface\\buffalo_l\\  -->  C:\\Users\\<name>\\.insightface\\models\\buffalo_l\\"])

pdf.h2("File 2: dlib facial landmark model  (~95 MB)")
pdf.code(["USB:\\models\\shape_predictor_68_face_landmarks.dat  -->  project\\models\\"])

pdf.note("tip",
    "Not doing a USB deployment? Just run  python offline_setup.py  in Step 5.\n"
    "It downloads both files automatically if the laptop has any internet."
)

pdf.ln(4)
pdf.h1(4, "Create Virtual Environment and Install Packages")

pdf.para("Navigate into the project folder if you haven't already:")
pdf.code(["cd Desktop\\face-recognition-skewed-images"])

pdf.h2("Create the virtual environment")
pdf.code([
    "python -m venv venv          # or: py -3.11 -m venv venv  if multiple versions",
    "venv\\Scripts\\activate",
])
pdf.para("You should see (venv) at the start of your prompt.")

pdf.h2("Install packages  -  with internet")
pdf.code([
    "pip install --upgrade pip",
    "pip install -r requirements.txt",
], note="Takes 10-20 minutes")

pdf.h2("Install packages  -  offline from USB wheel cache")
pdf.para("This requires preparing the wheel cache on another machine first:")
pdf.code([
    "# On Lucky's laptop (run once to prepare USB):",
    "pip download -r requirements.txt -d D:\\wheels",
    "",
    "# On deployment laptop (from USB):",
    "pip install --upgrade pip --no-index --find-links=D:\\wheels",
    "pip install --no-index --find-links=D:\\wheels -r requirements.txt",
], note="Replace D:\\wheels with your USB drive letter")

pdf.note("warn",
    "Every time you open a new Command Prompt, activate (venv) first:\n"
    "  cd Desktop\\face-recognition-skewed-images\n"
    "  venv\\Scripts\\activate\n"
    "If (venv) isn't showing, the environment isn't active."
)
pdf.note("err",
    "Execution policy error when activating? Use Command Prompt (cmd.exe), not PowerShell.\n"
    "Windows key → type cmd → Enter. Then try activating again."
)
pdf.note("err",
    "dlib-bin install failed? Run:\n"
    "  pip install cmake\n"
    "  pip install dlib\n"
    "  pip install -r requirements.txt\n"
    "Still failing? Install 'Visual Studio Build Tools 2022' (C++ workload) and retry."
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6  -  Step 5: Models
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1(5, "Download AI Models  (if not copied from USB in Step 3)")

pdf.para(
    "If the laptop has internet, run this one command to download the two large AI model files. "
    "If you already copied them from USB in Step 3, skip this step."
)
pdf.code(["python offline_setup.py"],
         note="Needs internet  -  downloads the buffalo_l model and dlib model (~600 MB total)")
pdf.para("A progress bar shows the download. Wait for all three items to complete:")
pdf.bullet("InsightFace buffalo_l  (face detection + recognition, ~500 MB)")
pdf.bullet("dlib 68-point landmark model  (~95 MB)")
pdf.bullet("UI fonts  (a few MB, needed for offline web interface)")
pdf.para("")
pdf.para(
    "After this finishes, the system can run with no internet at all. "
    "These files are cached and never need to be downloaded again."
)
pdf.note("tip",
    "The trained recognition models (SVM, FAISS, PCA) and enrolled photos\n"
    "were included in the git clone  -  you don't need to do anything extra for those."
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7  -  Step 6: MongoDB
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1(6, "Install MongoDB  (recommended for full features)")

pdf.para(
    "MongoDB stores audit logs, watchlist entries, and user accounts. "
    "Without it, you can still identify faces  -  but audit logs and watchlists won't work."
)

pdf.h2("Install from the internet")
pdf.para("Download from:")
pdf.code(["https://www.mongodb.com/try/download/community"],
         note="Select: Version 7.0, Platform: Windows, Package: msi")
pdf.para("Or run the installer from USB if no internet is available.")

pdf.para("Run the installer:")
pdf.bullet("Click Next. Accept the licence. Choose 'Complete'.")
pdf.note("err",
    "On the 'Service Configuration' screen  -  tick 'Install MongoDB as a Service'.\n"
    "This makes it start automatically with Windows. You won't need to start it manually."
)
pdf.bullet("Leave everything else as default. Install. Finish.")
pdf.para("")
pdf.para("Create the data folder (only needed once):")
pdf.code(["mkdir C:\\data\\db"])

pdf.h2("Verify MongoDB is running")
pdf.code([
    "sc query MongoDB          # should show: STATE: RUNNING",
    "net start MongoDB         # starts it if it's stopped",
])

pdf.note("tip",
    "If MongoDB was installed as a Windows Service, it starts automatically on every boot.\n"
    "You never need to touch it  -  just start the face recognition server and it works."
)

pdf.note("tip",
    "Skipping MongoDB for a quick field test? Fine.\n"
    "In Step 7, leave out the MONGO_URI line from your .env file.\n"
    "Face identification still works. Audit logs and watchlists won't."
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8  -  Step 7: .env
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1(7, "Create the Configuration File")

pdf.para(
    "The .env file is a short text file that tells the system your database address, "
    "a secret key, and various settings. You create it once and rarely need to change it."
)

pdf.h2("Create it from the template")
pdf.code([
    "copy .env.example .env",
    "notepad .env",
], note="Command Prompt inside the project folder")
pdf.para("Or copy the pre-configured .env from USB.")
pdf.para("Edit it to look like this:")

pdf.code([
    "# Change this to any long string  -  keep it secret",
    "JWT_SECRET_KEY=army-secret-key-change-this-2025",
    "",
    "# Database (choose one):",
    "",
    "# Option A: Local MongoDB  -  recommended if you installed MongoDB above",
    "MONGO_URI=mongodb://localhost:27017",
    "",
    "# Option B: Skip MongoDB (remove or comment out the line above)",
    "# MONGO_URI=mongodb://localhost:27017",
    "",
    "MONGO_DB_NAME=facerecog_db",
    "",
    "# Liveness detection",
    "LIVENESS_ENABLED=True",
    "LIVENESS_THRESHOLD=0.45    # Lower if rejecting real people, raise for stricter checking",
    "",
    "# Blink challenge (webcam / live mode only)",
    "BLINK_ENABLED=True",
    "BLINK_TIMEOUT_SEC=7.0",
    "",
    "# Recognition threshold  -  raise if too many UNKNOWN results",
    "FAISS_THRESHOLD=0.35",
], note=".env  -  save with Ctrl+S then close Notepad")

pdf.note("warn",
    "Change JWT_SECRET_KEY to something unique before operational deployment.\n"
    "Generate a random one:\n"
    "  python -c \"import secrets; print(secrets.token_hex(32))\"\n"
    "Paste the output as the key value."
)

pdf.note("tip",
    "FAISS_THRESHOLD tuning guide:\n"
    "  0.35 (default)  -  tuned for compressed UAV drone footage with TTA\n"
    "  0.45 to 0.50    -  better for high-quality ground-level photos\n"
    "  0.25 to 0.30    -  better for extreme blur or very high altitude\n"
    "Adjust live without restarting at  http://localhost:8000/config"
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 9  -  Step 8: Start
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1(8, "Start the System")

pdf.para("Before running, confirm:")
pdf.bullet("(venv) is active in your Command Prompt")
pdf.bullet("You're inside the project folder")
pdf.bullet("MongoDB is running (if you installed it)")
pdf.para("")
pdf.para("Start the server:")
pdf.code([
    "python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload"
], note="Keep this window open while the system is in use")

pdf.para("Wait 20-30 seconds. When it's ready you'll see:")
pdf.code([
    "INFO:     Uvicorn running on http://0.0.0.0:8000",
    "[FaceDetector] SCRFD loaded successfully.",
    "[FAISSMatcher] Loaded: 4 identities, threshold=0.35",
    "[LivenessService] Ready.",
    "[MongoDB] Connected to mongodb://localhost:27017",
    "INFO:     Application startup complete.",
])

pdf.para("Open Chrome or Edge and go to:")
pdf.code(["http://localhost:8000"])
pdf.para("The dashboard appears. The system is ready.")

pdf.note("err",
    "Website shows 'Pipeline not loaded'?\n\n"
    "The trained model files are missing from results\\baseline\\models\\\n"
    "Solution A: Copy the models folder from another machine via USB.\n"
    "Solution B: Train from scratch  -  needs photos in data\\gallery\\ first:\n"
    "  python experiments\\run_baseline.py   (takes 20-30 minutes)"
)
pdf.note("err",
    "Port 8000 in use? Use 8001:\n"
    "  python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8001 --reload\n"
    "Then open http://localhost:8001"
)
pdf.note("tip",
    "To stop the server: go to the CMD window and press Ctrl+C."
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 10  -  Firewall
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1(9, "Firewall and Security Settings")

pdf.para(
    "On military or institutional laptops, the firewall may block incoming connections. "
    "Do this if the browser shows 'connection refused' or if another device can't reach the server."
)

pdf.h2("Allow port 8000 through Windows Firewall")
pdf.bullet("Windows key → type 'Windows Defender Firewall' → Enter")
pdf.bullet("Click 'Advanced settings' on the left")
pdf.bullet("Click 'Inbound Rules' → 'New Rule...' on the right")
pdf.bullet("Select 'Port' → Next")
pdf.bullet("TCP, enter 8000 in 'Specific local ports' → Next")
pdf.bullet("'Allow the connection' → Next")
pdf.bullet("Tick Domain, Private, and Public → Next")
pdf.bullet("Name it 'Face Recognition' → Finish")

pdf.h2("Antivirus blocking the AI models or packages?")
pdf.bullet("Settings → Windows Security → Virus & Threat Protection → Manage Settings")
pdf.bullet("Add or remove exclusions → Add exclusion → Folder")
pdf.bullet("Exclude the project folder on the Desktop")
pdf.bullet("Also exclude:  C:\\Users\\<your-name>\\.insightface\\")

pdf.h2("Allow access from other devices on the same network")
pdf.para("The server already listens on all network interfaces. Find this laptop's IP:")
pdf.code(["ipconfig"], note="Look for IPv4 Address under WiFi or Ethernet")
pdf.para("From any device on the same network: http://<this-laptop-IP>:8000")

pdf.note("warn",
    "For classified environments where only this laptop should access the system:\n"
    "Start the server with --host 127.0.0.1 instead of 0.0.0.0:\n"
    "  python -m uvicorn web.backend.main:app --host 127.0.0.1 --port 8000 --reload\n"
    "This blocks all external access  -  only the server laptop can use it."
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 11  -  Enrollment
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1(10, "Enrolling People in the Field")

pdf.para(
    "The system comes pre-loaded with 4 enrolled identities (from the git clone). "
    "To add new people on the deployment laptop, follow these steps."
)

pdf.h2("Adding a new person via the browser")
pdf.bullet("Go to  http://localhost:8000/gallery")
pdf.bullet("Type the person's full name")
pdf.bullet("Upload 10-30 clear photos  (or a 20-60 second video of their face)")
pdf.bullet("The system retrains automatically after upload  -  wait for 'Retrain complete'")
pdf.bullet("The person is now in the system and will be identified going forward")
pdf.para("")
pdf.note("tip",
    "Photo quality tips:\n"
    "  - 15-30 photos per person gives much better accuracy than 5-10\n"
    "  - Include slightly different angles  -  not just straight-on\n"
    "  - Good lighting, no sunglasses, face at least 80px wide in the photo\n"
    "  - Blurry or very dark photos are automatically rejected  -  that's normal"
)

pdf.h2("After all enrollments are done  -  optional accuracy boost")
pdf.code([
    "python experiments\\run_baseline.py",
], note="Takes 20-30 minutes  -  gives the highest possible accuracy")
pdf.para(
    "This runs a full grid-search retrain. The quick auto-retrain after each upload "
    "is good for immediate use, but this full retrain can improve accuracy by 5-15% more."
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 12  -  Webcam
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1(11, "Using Live Camera in the Field")

pdf.para(
    "The demo page supports real-time face identification from a webcam or USB camera. "
    "The blink challenge is enabled by default to prevent photo spoofing."
)

pdf.h2("Starting live camera mode")
pdf.bullet("Go to  http://localhost:8000/demo  and click 'Live Stream'")
pdf.bullet("The browser asks 'Allow this site to use your camera?'  -  click Allow")
pdf.bullet("Point the camera at the person's face")
pdf.bullet("The system shows 'Please blink'  -  the person blinks once")
pdf.bullet("Face identification runs and shows the result")
pdf.para("")

pdf.h2("What the blink challenge does")
pdf.para(
    "A printed photo or a face on a screen cannot blink on command. "
    "By asking the subject to blink, the system confirms they are a real live person "
    "before running identification. "
    "They have 7 seconds to blink. If they don't blink in time, the system shows 'SPOOF DETECTED'."
)

pdf.h2("Disabling the blink challenge for field conditions")
pdf.para(
    "In some situations  -  bad camera angle, subject can't cooperate, or you're processing "
    "existing footage rather than live video  -  you can disable the blink challenge:"
)
pdf.code([
    "# Option 1: Disable in the Config page  -  no restart needed",
    "http://localhost:8000/config  ->  Blink Challenge  ->  toggle off",
    "",
    "# Option 2: Disable permanently in .env",
    "BLINK_ENABLED=False",
    "# Then restart the server",
])

pdf.note("warn",
    "Camera permission denied?\n"
    "Chrome: Settings → Privacy and Security → Site Settings → Camera\n"
    "Find http://localhost:8000 in Blocked and change it to Allow.\n"
    "Reload the demo page after granting permission."
)
pdf.note("tip",
    "Processing a drone video file? Go to the Demo page → Upload Video tab.\n"
    "Upload any .mp4 file. Adjust 'Process every N frames'  -  lower = more accurate.\n"
    "Download the output video with identity labels overlaid on each face."
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 13  -  Field adjustments
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.page_title("Field Adjustments  -  Tuning Without Restarting")

pdf.para(
    "All sensitivity settings can be adjusted live from the Config page. "
    "Go to  http://localhost:8000/config  in the browser. "
    "Changes take effect immediately  -  no restart needed."
)

pdf.h2("Liveness sensitivity  (Anti-Spoofing slider)")
pdf.table(
    ["Setting", "When to use it"],
    [
        ["0.45 (default)", "Balanced  -  works for most indoor and outdoor conditions"],
        ["0.35 to 0.40",   "Use if real people are being wrongly rejected (poor lighting, sun glare)"],
        ["0.55 to 0.65",   "Use in a controlled environment with good lighting"],
        ["0.0",            "Disable liveness check entirely (any photo passes  -  for testing only)"],
    ],
    [42, 128]
)

pdf.h2("Recognition threshold  (FAISS slider)")
pdf.table(
    ["Setting", "When to use it"],
    [
        ["0.35 (default)", "Tuned for UAV drone footage with image augmentation"],
        ["0.45 to 0.50",   "Use for high-quality ground-level photos or clear video"],
        ["0.25 to 0.30",   "Use for extreme blur or very high altitude footage"],
    ],
    [42, 128]
)

pdf.h2("Quick emergency adjustments")
pdf.note("err",
    "Too many real people being rejected as SPOOF?\n"
    "Config page → Anti-Spoofing → move slider left (lower value).\n"
    "Or toggle liveness off entirely for that session."
)
pdf.note("err",
    "Enrolled people showing as UNKNOWN?\n"
    "1. Config page → Recognition Threshold → move slider right (raise value to 0.45-0.50).\n"
    "2. Gallery page → Retrain from Gallery button.\n"
    "3. If accuracy is still poor, enroll more photos per person (15-30 is much better than 5)."
)

pdf.h2("All available pages")
pdf.table(
    ["Page", "Address", "What it does"],
    [
        ["Dashboard",  "http://localhost:8000/",           "System health at a glance"],
        ["Identify",   "http://localhost:8000/identify",   "Upload a single photo to identify"],
        ["Gallery",    "http://localhost:8000/gallery",    "Enroll people, manage photos, retrain"],
        ["Demo",       "http://localhost:8000/demo",       "Live webcam or video file processing"],
        ["Results",    "http://localhost:8000/results",    "Accuracy charts from training"],
        ["Audit Log",  "http://localhost:8000/audit",      "Full history of all identifications"],
        ["Watchlist",  "http://localhost:8000/watchlist",  "Set alerts for specific individuals"],
        ["Config",     "http://localhost:8000/config",     "Live threshold and sensitivity tuning"],
    ],
    [22, 62, 86]
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 14  -  Troubleshooting
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.page_title("Troubleshooting")

problems = [
    ("python is not recognized", "err",
     "Python isn't installed or wasn't added to PATH.\n"
     "Uninstall Python via Windows Settings → Apps, reinstall from python.org or USB.\n"
     "On the first installer screen, tick 'Add Python.exe to PATH'.\n"
     "Open a brand new CMD window and try again."),

    ("No module named 'web'  or  No module named 'fastapi'", "err",
     "The virtual environment isn't active.\n"
     "Run:\n"
     "  cd Desktop\\face-recognition-skewed-images\n"
     "  venv\\Scripts\\activate\n"
     "Then try your command again."),

    ("Website shows 'Pipeline not loaded'", "err",
     "The trained model files are missing from results\\baseline\\models\\\n"
     "Copy them from another machine via USB, or train from scratch:\n"
     "  python experiments\\run_baseline.py\n"
     "(Needs photos in data\\gallery\\  -  takes 20-30 minutes.)"),

    ("Face detection slow at startup / downloading models at startup", "err",
     "InsightFace can't find the buffalo_l model files.\n"
     "They should be at:  C:\\Users\\<name>\\.insightface\\models\\buffalo_l\\\n"
     "Copy from USB (Step 3) or run:  python offline_setup.py"),

    ("MongoDB connection error  /  ServerSelectionTimeoutError", "err",
     "MongoDB isn't running.\n"
     "If installed as a service:  net start MongoDB\n"
     "Or run in a separate CMD window:  mongod --dbpath C:\\data\\db\n"
     "(Keep that window open while using the system.)\n"
     "Or remove MONGO_URI from .env to skip MongoDB entirely."),

    ("SPOOF DETECTED for a real live person", "warn",
     "The liveness threshold is too strict for current conditions.\n"
     "Config page → Anti-Spoofing → lower the sensitivity slider.\n"
     "Try 0.35. Strong backlighting or unusual reflections can trigger false rejections."),

    ("Enrolled person keeps showing as UNKNOWN", "warn",
     "1. Raise the threshold: Config page → Recognition Threshold → move slider right.\n"
     "2. Retrain: Gallery page → Retrain from Gallery.\n"
     "3. Enroll more photos  -  15-30 per person is much better than 5-10.\n"
     "4. Check photo quality  -  face must be at least 20px wide, reasonably sharp."),

    ("No faces detected in drone video", "warn",
     "Face is likely too small in the frame.\n"
     "SCRFD can detect faces down to about 15px wide  -  below that it struggles.\n"
     "Reduce drone altitude, increase camera focal length, or reduce frame skip."),

    ("Camera not working in browser", "warn",
     "1. Chrome: Settings → Privacy → Camera → Allow for http://localhost:8000.\n"
     "2. Plug in USB camera before opening the demo page.\n"
     "3. Close Teams, Zoom, or any other app using the camera.\n"
     "4. Try Edge browser. Reload the page after granting permission."),

    ("Port 8000 already in use", "warn",
     "Use port 8001:\n"
     "  python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8001 --reload\n"
     "Then open http://localhost:8001"),
]

for title, kind, text in problems:
    pdf.note(kind, f"{title}\n\n{text}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 15  -  Quick Reference
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.page_title("Quick Reference   -   Print This Page")

pdf.h2("Every session  -  start the system")
pdf.code([
    "cd Desktop\\face-recognition-skewed-images",
    "venv\\Scripts\\activate",
    "python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload",
], note="Then open http://localhost:8000")

pdf.rule()

pdf.h2("If MongoDB isn't a Windows Service  -  start it first (separate CMD window)")
pdf.code(["mongod --dbpath C:\\data\\db"], note="Keep open while using the system")

pdf.rule()

pdf.h2("Full setup from scratch  -  internet method")
pdf.code([
    "# 1. Get the project (includes trained models + gallery)",
    "cd Desktop",
    "git clone https://github.com/TENSEI3011/face-recognition-skewed-images.git",
    "cd face-recognition-skewed-images",
    "",
    "# 2. Virtual environment",
    "python -m venv venv          # or: py -3.11 -m venv venv",
    "venv\\Scripts\\activate",
    "",
    "# 3. Install packages (10-20 minutes)",
    "pip install --upgrade pip",
    "pip install -r requirements.txt",
    "",
    "# 4. Download the 2 large AI models (needs internet, ~600 MB)",
    "python offline_setup.py",
    "",
    "# 5. MongoDB data folder (one time)",
    "mkdir C:\\data\\db",
    "",
    "# 6. Config file",
    "copy .env.example .env",
    "notepad .env",
    "",
    "# 7. Start",
    "python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload",
])

pdf.rule()

pdf.h2("Internet vs USB  -  what needs what")
pdf.table(
    ["Item", "How to get it"],
    [
        ["Code, trained models, gallery (from GitHub)", "Automatic with git clone"],
        ["InsightFace buffalo_l model (~500 MB)", "python offline_setup.py  OR  USB copy"],
        ["dlib landmark model (~95 MB)", "python offline_setup.py  OR  USB copy"],
        ["Python packages", "pip install -r requirements.txt  OR  offline wheel cache"],
        ["Python installer", "python.org  OR  USB"],
        ["MongoDB installer", "mongodb.com  OR  USB"],
    ],
    [82, 88]
)

# ══════════════════════════════════════════════════════════════════════════════
output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Army_Offline_Deployment_Guide.pdf")
pdf.output(output)
print(f"\nArmy Deployment Guide PDF generated successfully!")
print(f"Saved to: {output}")
print(f"Total pages: {pdf.page}")
