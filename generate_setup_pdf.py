"""
generate_setup_pdf.py
Generates the Setup Guide PDF for the Face Recognition UAV project.
Clean, natural, friendly tone. Easy to follow for anyone.
Run: python generate_setup_pdf.py
"""

from fpdf import FPDF
from fpdf.enums import XPos, YPos

NAVY   = (18, 38, 100)
TEAL   = (0, 120, 110)
RED    = (180, 30, 30)
AMBER  = (180, 90, 0)
WHITE  = (255, 255, 255)
DARK   = (30, 30, 45)
MID    = (90, 90, 110)
LIGHT  = (248, 249, 252)
RULE   = (210, 215, 225)


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
        self.set_fill_color(*NAVY)
        self.rect(0, 0, 210, 8, "F")
        self.set_font("Helvetica", "B", 6.5)
        self.set_text_color(*WHITE)
        self.set_y(1.5)
        self.cell(0, 5, "Face Recognition UAV Project  -  Setup Guide", align="C")
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
        self.cell(0, 5, f"Page {self.page - 1}   |   github.com/TENSEI3011/face-recognition-skewed-images", align="C")
        self.set_text_color(*DARK)

    # ── Cover ─────────────────────────────────────────────────────────────────
    def cover(self):
        self._on_cover = True
        self.add_page()

        # Background
        self.set_fill_color(*NAVY)
        self.rect(0, 0, 210, 297, "F")

        # Accent bar
        self.set_fill_color(*TEAL)
        self.rect(0, 0, 210, 5, "F")

        # Label
        self.set_y(30)
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(120, 160, 220)
        self.cell(0, 6, "FACE RECOGNITION ON SKEWED UAV IMAGES", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Title
        self.ln(6)
        self.set_font("Helvetica", "B", 34)
        self.set_text_color(*WHITE)
        self.cell(0, 16, "Setup Guide", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Subtitle
        self.set_font("Helvetica", "", 12)
        self.set_text_color(160, 195, 240)
        self.cell(0, 8, "How to install and run the system on any Windows laptop",
                  align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Description
        self.ln(8)
        self.set_x(30)
        self.set_font("Helvetica", "", 9.5)
        self.set_text_color(180, 205, 240)
        self.multi_cell(150, 6,
            "This guide walks you through every step  -  from installing Python\n"
            "to running the system for the first time. Follow the steps in order\n"
            "and you'll have everything working in about an hour.",
            align="C")

        # Chips
        self.ln(12)
        chips = ["Windows 10 / 11", "Python 3.10 or 3.11", "FastAPI + InsightFace", "MongoDB (optional)"]
        cx = 20
        for chip in chips:
            cw = 40 if len(chip) < 14 else 46
            self.set_fill_color(30, 55, 120)
            self.rect(cx, self.get_y(), cw, 10, "F")
            self.set_font("Helvetica", "", 7.5)
            self.set_text_color(*WHITE)
            self.set_xy(cx, self.get_y() + 1.5)
            self.cell(cw, 7, c(chip), align="C")
            cx += cw + 4

        # Footer strip
        self.set_fill_color(10, 22, 65)
        self.rect(0, 268, 210, 29, "F")
        self.set_y(276)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(130, 165, 215)
        self.cell(0, 6, "github.com/TENSEI3011/face-recognition-skewed-images", align="C")

        self._on_cover = False

    # ── Helpers ───────────────────────────────────────────────────────────────

    def h1(self, num, title):
        """Big step heading."""
        self.ln(5)
        if self.get_y() > 248:
            self.add_page()
        self.set_fill_color(*NAVY)
        self.rect(self.l_margin - 2, self.get_y(), 174, 10, "F")
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*WHITE)
        self.set_x(self.l_margin + 2)
        self.cell(0, 10, c(f"Step {num}   -   {title}"))
        self.set_text_color(*DARK)
        self.ln(13)

    def h2(self, text):
        """Section sub-heading."""
        if self.get_y() > 260:
            self.add_page()
        self.ln(2)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*TEAL)
        self.cell(0, 7, c(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*DARK)

    def para(self, text):
        """Normal paragraph."""
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
        self.set_fill_color(22, 28, 52)
        self.rect(self.l_margin, y0, 170, h, "F")
        self.set_font("Courier", "", 8.5)
        self.set_text_color(120, 210, 150)
        for line in lines:
            self.set_x(self.l_margin + 5)
            self.cell(0, 5.5, c(line), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(*DARK)
        self.ln(4)

    def note(self, kind, text):
        """kind: 'tip', 'warn', 'err'"""
        cfg = {
            "tip":  (TEAL,  (230, 248, 244), ">> "),
            "warn": (AMBER, (255, 245, 220), "!! "),
            "err":  (RED,   (255, 232, 232), "XX "),
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
        self.set_fill_color(*NAVY)
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
        self.set_text_color(*NAVY)
        self.cell(0, 10, c(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*TEAL)
        self.set_line_width(0.8)
        self.line(self.l_margin, self.get_y(), 190, self.get_y())
        self.set_line_width(0.2)
        self.set_text_color(*DARK)
        self.ln(6)


# ══════════════════════════════════════════════════════════════════════════════
pdf = PDF()

# ── Cover ─────────────────────────────────────────────────────────────────────
pdf.cover()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2  -  What you'll need
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.page_title("Before You Start")

pdf.para(
    "This guide installs the Face Recognition system on your Windows laptop step by step. "
    "Everything is explained in plain English  -  you don't need to be a programmer to follow it. "
    "Just read each step carefully and type the commands exactly as shown."
)

pdf.h2("What the laptop needs")
pdf.table(
    ["What", "Details", "How to check"],
    [
        ["Operating system", "Windows 10 or Windows 11 (64-bit)", "Settings → About"],
        ["Python", "Version 3.10 or 3.11  (NOT 3.12 or 3.13)", "Open CMD, type: python --version"],
        ["RAM", "At least 8 GB  (16 GB is better)", "Task Manager → Performance"],
        ["Free disk space", "At least 5 GB", "File Explorer → This PC"],
        ["Internet", "Needed once for setup, not after", " - "],
        ["Web browser", "Chrome or Edge (already on Windows)", " - "],
        ["Webcam", "Optional  -  only for live camera mode", " - "],
    ],
    [38, 78, 54]
)

pdf.note("warn",
    "Python version matters a lot.\n"
    "Only Python 3.10 or 3.11 work with all the packages this project needs.\n"
    "Python 3.12 and 3.13 will cause errors during installation.\n"
    "If you're not sure which version you have, open Command Prompt and type:  python --version"
)

pdf.para("")
pdf.h2("How long does this take?")
pdf.para(
    "About 60 to 90 minutes for a fresh laptop with no Python installed. "
    "Most of that time is waiting for packages to download  -  you don't need to watch it. "
    "If Python is already installed and working, you can be done in 30 minutes."
)

pdf.h2("What does this system do?")
pdf.para(
    "It identifies people from photos or video  -  including footage from drones flying overhead. "
    "You upload a photo, and it tells you who the person is (from the people you've enrolled). "
    "It also checks whether the face in the photo is a real live person or a printed photo/screen. "
    "The whole system runs on your laptop  -  no internet connection needed after setup."
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3  -  Step 1: Python
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1(1, "Install Python 3.11")

pdf.para(
    "Python is the programming language this project runs on. "
    "You need version 3.11 specifically."
)

pdf.h2("Download and install")
pdf.para("Open Chrome or Edge and go to:")
pdf.code(["https://www.python.org/downloads/release/python-3119/"], note="Type or paste this in your browser")
pdf.para(
    "Scroll to the bottom of the page to the 'Files' section. "
    "Click 'Windows installer (64-bit)' to download it. "
    "Once downloaded, double-click the file to run it."
)
pdf.note("err",
    "On the very first screen of the installer, you'll see a checkbox at the bottom:\n"
    "'Add Python.exe to PATH'\n\n"
    "You MUST tick this box before clicking Install Now.\n"
    "If you miss it, Python won't work from Command Prompt."
)
pdf.para("Click 'Install Now'. Wait for it to finish, then click Close.")

pdf.h2("Check it worked")
pdf.para("Open Command Prompt (press Windows key, type cmd, press Enter) and type:")
pdf.code(["python --version"])
pdf.para("You should see something like:  Python 3.11.9")
pdf.note("err",
    "Seeing 'python is not recognized as an internal or external command'?\n\n"
    "This means Python wasn't added to PATH. Uninstall Python from Windows Settings → Apps,\n"
    "then reinstall it and make sure to tick 'Add Python.exe to PATH' this time.\n"
    "Open a new CMD window after reinstalling."
)
pdf.note("tip",
    "Already have Python 3.12 or 3.13 on the laptop? Don't uninstall it.\n"
    "Install Python 3.11 alongside it from the same website.\n"
    "When you get to Step 4 (virtual environment), use this command instead:\n"
    "  py -3.11 -m venv venv\n"
    "This makes sure the project uses 3.11 even if another version is the default."
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4  -  Steps 2-3
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1(2, "Install Git")

pdf.para(
    "Git is a tool that downloads the project code from GitHub. "
    "You only need to install it once."
)
pdf.para("Open your browser and go to:")
pdf.code(["https://git-scm.com/download/win"], note="Download the '64-bit Git for Windows Setup'")
pdf.para(
    "Run the installer. Click Next on every screen  -  all the default settings are fine. "
    "Click Install, then Finish when it's done."
)

pdf.h2("Check it worked")
pdf.code(["git --version"])
pdf.para("You should see something like:  git version 2.45.0.windows.1")

pdf.note("tip",
    "No internet on the laptop? Download the Git installer on another machine,\n"
    "copy the .exe file to USB, and run it on this laptop.\n"
    "The installer itself doesn't need internet."
)

pdf.ln(4)
pdf.h1(3, "Download the Project")

pdf.h2("Option A  -  If this laptop has internet (recommended)")
pdf.para("Open Command Prompt and type these commands one at a time:")
pdf.code([
    "cd Desktop",
    "git clone https://github.com/TENSEI3011/face-recognition-skewed-images.git",
    "cd face-recognition-skewed-images",
], note="Press Enter after each line")
pdf.para(
    "A folder called 'face-recognition-skewed-images' will appear on your Desktop. "
    "This downloads the code, the trained recognition models, and all the enrolled person photos."
)

pdf.h2("Option B  -  No internet, copying from USB")
pdf.para("Copy the 'face-recognition-skewed-images' folder from USB to the Desktop. Then:")
pdf.code(["cd Desktop\\face-recognition-skewed-images"], note="Navigate into the folder")

pdf.note("tip",
    "Not sure if you're in the right folder? Type  dir  and press Enter.\n"
    "You should see files like requirements.txt, web, src, data, models.\n"
    "If you see 'path not found', double-check the folder name in File Explorer\n"
    "and adjust the cd command to match the exact name."
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5  -  Steps 4-5
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1(4, "Create a Virtual Environment")

pdf.para(
    "A virtual environment is like a clean, separate workspace for this project. "
    "It keeps the project's packages away from anything else on your laptop. "
    "You must do this before installing the packages."
)
pdf.para("Make sure you're still inside the project folder from Step 3, then run:")

pdf.h2("If Python 3.11 is the only version on this laptop")
pdf.code([
    "python -m venv venv",
    "venv\\Scripts\\activate",
])

pdf.h2("If you have multiple Python versions (e.g., 3.11 and 3.13)")
pdf.code([
    "py -3.11 -m venv venv",
    "venv\\Scripts\\activate",
])

pdf.para("After activating, your Command Prompt line will start with (venv):")
pdf.code(["(venv) C:\\Users\\...\\face-recognition-skewed-images>"],
         note="This (venv) prefix means the environment is active  -  required!")

pdf.note("warn",
    "Every time you open a new Command Prompt window, you need to activate the\n"
    "virtual environment again before running any project commands:\n\n"
    "  cd Desktop\\face-recognition-skewed-images\n"
    "  venv\\Scripts\\activate\n\n"
    "If (venv) isn't showing, the environment isn't active."
)
pdf.note("err",
    "Getting an 'execution policy' error? This happens in PowerShell, not CMD.\n"
    "Use Command Prompt instead: press Windows key, type cmd, press Enter.\n"
    "The activate command works fine in cmd.exe."
)

pdf.ln(4)
pdf.h1(5, "Install the Packages")

pdf.para("Make sure (venv) is visible in your prompt, then run:")
pdf.code([
    "pip install --upgrade pip",
    "pip install -r requirements.txt",
], note="This downloads and installs all required libraries  -  takes 10 to 20 minutes")

pdf.para(
    "You'll see a lot of text scrolling past  -  that's normal. "
    "Just wait until it finishes. It's done when you see 'Successfully installed ...' at the end."
)

pdf.note("err",
    "Error: 'dlib-bin install failed' or 'Microsoft Visual C++ required'?\n\n"
    "Run these in order:\n"
    "  pip install cmake\n"
    "  pip install dlib\n"
    "  pip install -r requirements.txt\n\n"
    "Still failing? You may need to install 'Visual Studio Build Tools 2022'.\n"
    "Search for it in your browser, download it, select 'Desktop development\n"
    "with C++', and install. Then retry the pip commands above."
)
pdf.note("err",
    "Error: 'No matching distribution found'?\n\n"
    "Your Python version is incompatible. Check:  python --version\n"
    "If it shows 3.12 or 3.13, delete the venv folder and redo Step 4 using:\n"
    "  py -3.11 -m venv venv"
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6  -  Step 6: Models
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1(6, "Download the AI Models")

pdf.para(
    "The system needs two large AI model files to work. "
    "One detects faces in images, the other recognises who the person is. "
    "These are downloaded once and then work offline forever."
)

pdf.para("Run this command with (venv) active:")
pdf.code(["python offline_setup.py"], note="Needs internet  -  downloads about 600 MB total")

pdf.para("It will download:")
pdf.bullet("InsightFace buffalo_l  (face detection + recognition model, ~500 MB)")
pdf.bullet("dlib facial landmark model  (~95 MB, used for blink detection)")
pdf.bullet("UI fonts  (~2 MB, so the website works offline)")
pdf.para("")
pdf.para("The download shows a progress bar. Just wait for all three to complete.")

pdf.note("tip",
    "Slow internet or no internet?\n\n"
    "You only need to copy 2 things via USB from another machine that already ran setup:\n\n"
    "  1. The buffalo_l folder (~500 MB):\n"
    "     Copy from:  C:\\Users\\lucky\\.insightface\\models\\buffalo_l\\\n"
    "     Paste to:   C:\\Users\\<your-name>\\.insightface\\models\\buffalo_l\\\n\n"
    "  2. The dlib model (~95 MB):\n"
    "     Copy from:  models\\shape_predictor_68_face_landmarks.dat  (in project folder)\n"
    "     Paste to:   models\\  folder inside this project\n\n"
    "The .insightface folder is hidden  -  in File Explorer, click View → tick 'Hidden items'.\n"
    "Everything else (trained models, gallery) already came with the git clone."
)

pdf.note("tip",
    "Already completed setup elsewhere? The trained models and enrolled person gallery\n"
    "were included when you ran git clone  -  no extra steps needed for those."
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7  -  Step 7: MongoDB
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1(7, "Set Up MongoDB  (optional but recommended)")

pdf.para(
    "MongoDB is the database that stores audit logs, alert watchlists, and user accounts. "
    "Without it, the system still works for face identification  -  but you won't have "
    "the audit log or watchlist features."
)

pdf.h2("Install MongoDB")
pdf.para("Open your browser and go to:")
pdf.code(["https://www.mongodb.com/try/download/community"], note="Select: Version 7.0, Platform: Windows, Package: msi")
pdf.para("Download the .msi file and run it.")
pdf.bullet("Click Next. Accept the license. Choose 'Complete'.")
pdf.note("err",
    "On the 'Service Configuration' screen  -  tick 'Install MongoDB as a Service'.\n"
    "This makes MongoDB start automatically with Windows  -  no manual steps each time."
)
pdf.bullet("Leave everything else as default. Click Next, then Install.")
pdf.bullet("Wait about 5-10 minutes. Click Finish.")

pdf.h2("Create the data folder (once only)")
pdf.code(["mkdir C:\\data\\db"], note="Command Prompt  -  creates the folder MongoDB stores data in")

pdf.h2("Check MongoDB is running")
pdf.code(["sc query MongoDB"], note="Should show STATE: RUNNING")
pdf.para("If it's not running:")
pdf.code(["net start MongoDB"])

pdf.note("tip",
    "If MongoDB is installed as a Windows Service (the recommended option above),\n"
    "it starts automatically every time Windows boots. You never need to start it manually."
)

pdf.note("tip",
    "Skipping MongoDB? That's fine for a basic install.\n"
    "In Step 8, just remove the MONGO_URI line from your .env file.\n"
    "Face identification, liveness checking, and all the core features still work.\n"
    "You just won't have the Audit Log or Watchlist pages."
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8  -  Step 8: .env file
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1(8, "Create the Configuration File")

pdf.para(
    "The .env file is a simple text file that tells the system your settings  -  "
    "things like the database address and a secret key. "
    "You create it once and never need to touch it again unless you want to change something."
)

pdf.h2("Create it from the template")
pdf.code([
    "copy .env.example .env",
    "notepad .env",
], note="Run in Command Prompt inside the project folder")

pdf.para("Notepad will open. Edit the file to look like this:")
pdf.code([
    "# Secret key  -  change this to anything you like, keep it private",
    "JWT_SECRET_KEY=my-secret-key-change-this-2025",
    "",
    "# Database  -  choose one option:",
    "",
    "# Option A: Local MongoDB (recommended if you installed MongoDB in Step 7)",
    "MONGO_URI=mongodb://localhost:27017",
    "",
    "# Option B: Skip MongoDB (just remove or comment out MONGO_URI)",
    "# MONGO_URI=mongodb://localhost:27017",
    "",
    "MONGO_DB_NAME=facerecog_db",
    "",
    "# Liveness detection (detects printed photos and screen replays)",
    "LIVENESS_ENABLED=True",
    "LIVENESS_THRESHOLD=0.45",
    "",
    "# Blink challenge (webcam mode only  -  user must blink to prove they're real)",
    "BLINK_ENABLED=True",
    "",
    "# Recognition sensitivity  -  raise if too many 'UNKNOWN' results",
    "FAISS_THRESHOLD=0.35",
], note=".env file  -  save with Ctrl+S then close Notepad")

pdf.note("warn",
    "Change JWT_SECRET_KEY to something unique  -  it's the key that signs all login tokens.\n"
    "Don't share it with anyone or commit it to GitHub.\n"
    "(The .gitignore already excludes .env from being uploaded.)"
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 9  -  Step 9: Start server
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1(9, "Start the System")

pdf.para("Before starting, make sure:")
pdf.bullet("(venv) is active in your Command Prompt")
pdf.bullet("You are inside the project folder")
pdf.bullet("MongoDB is running (if you installed it)")
pdf.para("")
pdf.para("Then run:")
pdf.code([
    "python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload"
], note="Keep this window open while you're using the system")

pdf.para(
    "The first startup takes 20-30 seconds while the AI models load. "
    "When it's ready, you'll see something like:"
)
pdf.code([
    "INFO:     Uvicorn running on http://0.0.0.0:8000",
    "[FaceDetector] SCRFD loaded successfully.",
    "[FAISSMatcher] Loaded: 4 identities",
    "[LivenessService] Ready.",
    "INFO:     Application startup complete.",
])

pdf.para("Now open Chrome or Edge and go to:")
pdf.code(["http://localhost:8000"])
pdf.para("The dashboard will load and the system is ready to use.")

pdf.note("err",
    "Website shows 'Pipeline not loaded'?\n\n"
    "The trained model files are missing from results\\baseline\\models\\\n"
    "Solution A: Copy them from another machine via USB.\n"
    "Solution B: Train from scratch (needs photos in data\\gallery\\ first):\n"
    "  python experiments\\run_baseline.py\n"
    "This takes 20-30 minutes."
)
pdf.note("err",
    "Port 8000 already in use? Try port 8001:\n"
    "  python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8001 --reload\n"
    "Then open http://localhost:8001 in the browser."
)
pdf.note("tip",
    "To stop the server, go to the Command Prompt window running it and press Ctrl+C."
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 10  -  Step 10: Firewall
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1(10, "Firewall Settings  (if the browser can't connect)")

pdf.para(
    "On university or corporate laptops, Windows Firewall may block the connection. "
    "You only need to do this if the website isn't loading."
)

pdf.h2("Allow port 8000 through Windows Firewall")
pdf.bullet("Press Windows key, type 'Windows Defender Firewall', press Enter")
pdf.bullet("Click 'Advanced settings' on the left")
pdf.bullet("Click 'Inbound Rules', then 'New Rule...' on the right")
pdf.bullet("Select 'Port' → Next")
pdf.bullet("Select 'TCP', enter 8000 in 'Specific local ports' → Next")
pdf.bullet("Select 'Allow the connection' → Next")
pdf.bullet("Tick all three boxes (Domain, Private, Public) → Next")
pdf.bullet("Name it 'Face Recognition' → Finish")

pdf.h2("If antivirus is blocking the AI models")
pdf.para("Add an exclusion in Windows Defender:")
pdf.bullet("Settings → Windows Security → Virus & Threat Protection")
pdf.bullet("Manage Settings → Add or remove exclusions → Add exclusion → Folder")
pdf.bullet("Exclude the entire project folder on the Desktop")
pdf.bullet("Also exclude:  C:\\Users\\<your-name>\\.insightface\\")

pdf.h2("Accessing from another device on the same network")
pdf.para(
    "Because the server runs on --host 0.0.0.0, any device on your WiFi can access it. "
    "Find this laptop's IP address:"
)
pdf.code(["ipconfig"], note="Look for 'IPv4 Address' under WiFi or Ethernet")
pdf.para("From another device, open:  http://<this-laptop's-IP>:8000")
pdf.para("Example:  http://192.168.1.42:8000")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 11  -  Step 11: Enroll people
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1(11, "Enroll People  (add someone to the system)")

pdf.para(
    "The system comes with 4 people already enrolled (Aditi, Samridhi, Siddhant, Stuti). "
    "To add new people, follow these steps."
)

pdf.h2("The easiest way  -  using the browser")
pdf.bullet("Go to  http://localhost:8000/gallery")
pdf.bullet("Type the person's full name in the Name field")
pdf.bullet("Click 'Upload Images' and select 10-30 clear photos of their face")
pdf.bullet("Or drag and drop a short video clip (20-60 seconds of their face)")
pdf.bullet("The system automatically retrains after the upload  -  wait for 'Retrain complete'")
pdf.para("")
pdf.note("tip",
    "Better photos = better recognition accuracy.\n\n"
    "- Use 15-30 photos per person if you can\n"
    "- Include slightly different angles (not just straight-on)\n"
    "- Good lighting, no sunglasses\n"
    "- Face should be at least 80 pixels wide in the photo\n"
    "- Blurry or very dark photos are automatically rejected  -  that's normal"
)

pdf.h2("The folder method  -  for bulk uploads")
pdf.para("Create a subfolder for each person inside  data\\gallery\\  like this:")
pdf.code([
    "data\\",
    "  gallery\\",
    "    person_name\\",
    "      photo1.jpg   photo2.jpg   photo3.jpg ...",
])
pdf.para("Then go to the Gallery page and click 'Retrain from Gallery'.")

pdf.h2("For best accuracy after final enrollment")
pdf.para("After enrolling everyone, run a full retrain from Command Prompt:")
pdf.code(["python experiments\\run_baseline.py"], note="Takes 20-30 minutes, produces the most accurate models")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 12  -  Step 12: Using the system
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.h1(12, "Using the System")

pdf.h2("Pages you can visit")
pdf.table(
    ["Page", "Address", "What it does"],
    [
        ["Dashboard",  "http://localhost:8000/",           "System status at a glance"],
        ["Identify",   "http://localhost:8000/identify",   "Upload a photo to identify someone"],
        ["Gallery",    "http://localhost:8000/gallery",    "Enroll people, add photos, retrain"],
        ["Demo",       "http://localhost:8000/demo",       "Process a video or use live webcam"],
        ["Results",    "http://localhost:8000/results",    "Accuracy graphs and experiment charts"],
        ["Audit Log",  "http://localhost:8000/audit",      "Full history of all identifications"],
        ["Watchlist",  "http://localhost:8000/watchlist",  "Set alerts for specific people"],
        ["Config",     "http://localhost:8000/config",     "Adjust thresholds without restarting"],
        ["API Docs",   "http://localhost:8000/docs",       "Technical API reference"],
    ],
    [22, 64, 84]
)

pdf.h2("Identifying someone from a photo")
pdf.bullet("Go to  http://localhost:8000/identify")
pdf.bullet("Click 'Choose File' and select a photo (JPG or PNG)")
pdf.bullet("Click Identify")
pdf.bullet("The system first checks if the face is real (not a printed photo)")
pdf.bullet("If it passes, it shows the person's name and a confidence score")

pdf.h2("Using the live webcam")
pdf.bullet("Go to  http://localhost:8000/demo  and click 'Live Stream'")
pdf.bullet("The browser will ask permission to use the camera  -  click Allow")
pdf.bullet("The system shows a 'Please blink' challenge  -  blink once to verify you're real")
pdf.bullet("After blinking, it identifies you automatically")

pdf.note("tip",
    "Getting too many 'UNKNOWN' results on enrolled people?\n"
    "Go to  http://localhost:8000/config  and raise the Recognition Threshold slider.\n"
    "Start at 0.45. You can adjust it live  -  no restart needed."
)
pdf.note("tip",
    "Liveness check rejecting real people? Lower the Anti-Spoofing sensitivity\n"
    "on the Config page. Or set LIVENESS_ENABLED=False in .env and restart."
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 13  -  Troubleshooting
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.page_title("Troubleshooting  -  Common Problems and Fixes")

problems = [
    (
        "python is not recognized as an internal or external command",
        "err",
        "Python isn't installed or wasn't added to PATH.\n"
        "Uninstall Python from Settings → Apps, then reinstall from python.org.\n"
        "Tick 'Add Python.exe to PATH' on the very first installer screen.\n"
        "Open a brand new CMD window after reinstalling and try again."
    ),
    (
        "No module named 'web'  or  No module named 'fastapi'",
        "err",
        "The virtual environment isn't active.\n"
        "Make sure (venv) appears at the start of your prompt.\n"
        "If not, run:\n"
        "  cd Desktop\\face-recognition-skewed-images\n"
        "  venv\\Scripts\\activate\n"
        "Then try your command again."
    ),
    (
        "Website shows 'Pipeline not loaded'",
        "err",
        "The trained recognition model files are missing.\n"
        "Solution A: Copy results\\baseline\\models\\ from another machine via USB.\n"
        "Solution B: Train from scratch  -  needs photos in data\\gallery\\ first:\n"
        "  python experiments\\run_baseline.py   (takes 20-30 min)"
    ),
    (
        "dlib model not found  /  shape_predictor_68 not found",
        "err",
        "The dlib model file is missing from the models\\ folder.\n"
        "Run:  python offline_setup.py\n"
        "Or copy shape_predictor_68_face_landmarks.dat from another machine into the models\\ folder."
    ),
    (
        "Face detection is very slow at startup / models downloading at startup",
        "err",
        "InsightFace is trying to download the buffalo_l model because it's not cached.\n"
        "Run:  python offline_setup.py   to cache it properly.\n"
        "Or copy the buffalo_l folder from another machine to:\n"
        "  C:\\Users\\<your-name>\\.insightface\\models\\buffalo_l\\"
    ),
    (
        "MongoDB connection error  /  ServerSelectionTimeoutError",
        "err",
        "MongoDB isn't running.\n"
        "If installed as a service:  net start MongoDB\n"
        "Or open the Services app (Windows key → type 'services') and start MongoDB.\n"
        "If not installed as a service, open a separate CMD window and run:\n"
        "  mongod --dbpath C:\\data\\db\n"
        "Keep that window open while using the system."
    ),
    (
        "SPOOF DETECTED for a real live person",
        "warn",
        "The liveness sensitivity is too strict for current lighting conditions.\n"
        "Go to  http://localhost:8000/config  → Anti-Spoofing → lower the sensitivity slider.\n"
        "Try 0.35 or 0.30 first. Bright backlighting often triggers false detections."
    ),
    (
        "Person keeps showing as UNKNOWN",
        "warn",
        "1. Retrain the models: Gallery page → Retrain from Gallery button.\n"
        "2. Raise the threshold: Config page → Recognition Threshold → move slider right.\n"
        "3. Enroll more photos: 15-30 good photos per person works much better than 5-10.\n"
        "4. Check photo quality: face must be at least 20 pixels wide, not too blurry."
    ),
    (
        "Camera not working in browser",
        "warn",
        "1. Check camera permission: Chrome → Settings → Privacy → Camera → Allow for localhost.\n"
        "2. Plug in a USB camera before opening the demo page.\n"
        "3. Close other apps using the camera (Teams, Zoom, etc.).\n"
        "4. Try Edge if Chrome doesn't work, or reload the page after granting permission."
    ),
    (
        "Port 8000 is already in use",
        "warn",
        "Use a different port:\n"
        "  python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8001 --reload\n"
        "Then open  http://localhost:8001  in the browser."
    ),
]

for title, kind, text in problems:
    pdf.note(kind, f"{title}\n\n{text}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 14  -  Quick Reference
# ══════════════════════════════════════════════════════════════════════════════
pdf.add_page()
pdf.page_title("Quick Reference   -   Print This Page")

pdf.h2("Every session  -  3 commands to start")
pdf.code([
    "cd Desktop\\face-recognition-skewed-images",
    "venv\\Scripts\\activate",
    "python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload",
], note="Then open http://localhost:8000 in Chrome or Edge")

pdf.rule()

pdf.h2("If MongoDB isn't a Windows Service  -  start it first (separate CMD window)")
pdf.code(["mongod --dbpath C:\\data\\db"], note="Keep this window open while using the system")

pdf.rule()

pdf.h2("Full first-time setup from scratch")
pdf.code([
    "# Step 3: Download the project",
    "cd Desktop",
    "git clone https://github.com/TENSEI3011/face-recognition-skewed-images.git",
    "cd face-recognition-skewed-images",
    "",
    "# Step 4: Create virtual environment",
    "python -m venv venv          # or: py -3.11 -m venv venv",
    "venv\\Scripts\\activate",
    "",
    "# Step 5: Install packages (10-20 minutes)",
    "pip install --upgrade pip",
    "pip install -r requirements.txt",
    "",
    "# Step 6: Download AI models (one time, needs internet)",
    "python offline_setup.py",
    "",
    "# Step 7: MongoDB data folder (one time)",
    "mkdir C:\\data\\db",
    "",
    "# Step 8: Create config file",
    "copy .env.example .env",
    "notepad .env",
    "",
    "# Step 9: Start",
    "python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload",
])

pdf.rule()

pdf.h2("What's on GitHub  vs  what needs internet/USB")
pdf.table(
    ["Item", "Where to get it"],
    [
        ["Project code (all Python, HTML, CSS, JS)", "Automatic from git clone"],
        ["Trained recognition models (SVM + FAISS + PCA, ~5 MB)", "Automatic from git clone"],
        ["Enrolled gallery photos (Aditi, Samridhi, Siddhant, Stuti)", "Automatic from git clone"],
        ["InsightFace buffalo_l model (~500 MB)", "python offline_setup.py  OR  USB copy"],
        ["dlib facial landmark model (~95 MB)", "python offline_setup.py  OR  USB copy"],
    ],
    [90, 80]
)

# ══════════════════════════════════════════════════════════════════════════════
output = r"c:\Users\lucky\OneDrive\Desktop\Face Recognition\Setup_Guide_Face_Recognition.pdf"
pdf.output(output)
print(f"\nSetup Guide PDF generated successfully!")
print(f"Saved to: {output}")
print(f"Total pages: {pdf.page}")
