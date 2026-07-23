"""
generate_explanation_pdf.py
Generates a clean, beginner-friendly explanation of the Face Recognition pipeline.
Run: python generate_explanation_pdf.py
"""

from fpdf import FPDF
from fpdf.enums import XPos, YPos

# ─── Colour palette ────────────────────────────────────────────────────────────
NAVY      = (15,  30, 100)
BLUE      = (28,  78, 183)
LIGHT_BG  = (235, 240, 255)
GREEN     = (22, 140,  70)
GREEN_BG  = (220, 245, 230)
ORANGE    = (190,  90,   0)
ORANGE_BG = (255, 240, 215)
PURPLE    = (100,  40, 180)
PURPLE_BG = (240, 230, 255)
TEAL      = (0,  130, 120)
TEAL_BG   = (220, 245, 242)
RED       = (185,  30,  30)
RED_BG    = (255, 225, 225)
WHITE     = (255, 255, 255)
DARK      = ( 20,  20,  40)
GREY      = (110, 110, 130)
LGREY     = (245, 246, 250)
YELLOW_BG = (255, 252, 220)
YELLOW    = (160, 120,   0)


def clean(s):
    """Strip characters that FPDF cannot encode in latin-1."""
    return s.encode("latin-1", errors="replace").decode("latin-1")


# ─── PDF class ─────────────────────────────────────────────────────────────────

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
        self.fc(NAVY); self.rect(0, 0, 210, 10, "F")
        self.set_font("Helvetica", "B", 7); self.tc(WHITE)
        self.set_y(2)
        self.cell(0, 6, "FACE RECOGNITION PIPELINE  -  SIMPLE LANGUAGE GUIDE", align="C")
        self.tc(DARK); self.ln(9)

    def footer(self):
        if self._cover_page:
            return
        self.set_y(-13)
        self.fc(LGREY); self.rect(0, self.get_y() - 1, 210, 16, "F")
        self.set_font("Helvetica", "", 7); self.tc(GREY)
        self.cell(0, 9,
                  f"Page {self.page - 1}  |  Face Recognition Pipeline  |  Simple Language Guide",
                  align="C")
        self.tc(DARK)

    def cover(self):
        self._cover_page = True
        self.add_page()
        self.fc(NAVY); self.rect(0, 0, 210, 297, "F")
        self.fc((28, 55, 160)); self.rect(0, 0, 210, 6, "F")

        self.set_font("Helvetica", "B", 10); self.tc((130, 160, 255))
        self.set_y(22)
        self.cell(0, 8, "[ FACE RECOGNITION ON SKEWED UAV IMAGES ]", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(8)
        self.set_font("Helvetica", "B", 30); self.tc(WHITE)
        self.cell(0, 14, "HOW IT WORKS", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Helvetica", "B", 14); self.tc((160, 190, 255))
        self.cell(0, 9, "The Complete Pipeline in Simple Language",
                  align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(6)
        self.set_font("Helvetica", "", 10); self.tc((190, 210, 255))
        self.set_x(25)
        self.multi_cell(160, 6,
            "Every step explained with plain words, real examples,\n"
            "and the reason why each component was chosen.\n"
            "Covers SCRFD, ArcFace+TTA, Liveness, FAISS, and Temporal Voting.",
            align="C")

        self.ln(14)
        badges = [
            ("STEP 1", "Detect"),
            ("STEP 2", "Align"),
            ("STEP 3", "Anti-Spoof"),
            ("STEP 4", "Features"),
            ("STEP 5", "Fuse+PCA"),
            ("STEP 6", "FAISS"),
            ("STEP 7", "Temporal Vote"),
        ]
        bx, bw, by = 12, 26, self.get_y()
        for lbl, val in badges:
            self.fc((30, 55, 140)); self.dc(BLUE)
            self.rect(bx, by, bw, 20, "FD")
            self.set_font("Helvetica", "B", 6); self.tc((130, 160, 255))
            self.set_xy(bx, by + 3); self.cell(bw, 5, lbl, align="C")
            self.set_font("Helvetica", "B", 7); self.tc(WHITE)
            self.set_xy(bx, by + 9); self.cell(bw, 7, val, align="C")
            bx += bw + 2

        self.fc((10, 20, 70)); self.rect(0, 262, 210, 35, "F")
        self.set_font("Helvetica", "B", 9); self.tc((130, 160, 255))
        self.set_xy(0, 270)
        self.cell(0, 6, "github.com/TENSEI3011/face-recognition-skewed-images", align="C")
        self.set_font("Helvetica", "", 8); self.tc((100, 130, 200))
        self.set_xy(0, 279)
        self.cell(0, 6, "SCRFD + ArcFace + FAISS + SVM + Liveness + Temporal Voting", align="C")
        self._cover_page = False

    # ── layout helpers ─────────────────────────────────────────────────────────

    def section(self, num, title, color=NAVY):
        self.ln(4)
        if self.get_y() > 248:
            self.add_page()
        self.fc(color); self.rect(self.l_margin - 2, self.get_y(), 176, 9, "F")
        self.set_font("Helvetica", "B", 11); self.tc(WHITE)
        self.set_x(self.l_margin)
        self.cell(0, 9, clean(f"  {num}  {title.upper()}"))
        self.tc(DARK); self.ln(12)

    def sub(self, t, color=BLUE):
        self.set_font("Helvetica", "B", 10); self.tc(color)
        self.cell(0, 7, clean(t), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.tc(DARK)

    def body(self, t):
        self.set_font("Helvetica", "", 9.5); self.tc(DARK)
        self.multi_cell(0, 5.8, clean(t)); self.ln(1)

    def bullet(self, t, indent=4):
        self.set_font("Helvetica", "", 9.5); self.tc(DARK)
        x = self.l_margin + indent
        self.set_x(x); self.cell(4, 5.5, "-")
        self.set_x(x + 5); self.multi_cell(0, 5.5, clean(t))

    def analogy(self, title, text):
        h = max(18, len(text) // 65 * 5.5 + 20)
        y0 = self.get_y()
        if y0 + h > 272:
            self.add_page(); y0 = self.get_y()
        self.fc(YELLOW_BG); self.rect(self.l_margin, y0, 174, h, "F")
        self.dc(YELLOW); self.set_line_width(0.8)
        self.line(self.l_margin, y0, self.l_margin, y0 + h)
        self.set_line_width(0.2)
        self.set_xy(self.l_margin + 5, y0 + 3)
        self.set_font("Helvetica", "B", 8.5); self.tc(YELLOW)
        self.cell(0, 5, clean(f"[ANALOGY]  {title}"),
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_x(self.l_margin + 5)
        self.set_font("Helvetica", "", 9); self.tc(DARK)
        self.multi_cell(165, 5.2, clean(text)); self.ln(4)

    def example(self, title, text):
        h = max(18, len(text) // 65 * 5.5 + 20)
        y0 = self.get_y()
        if y0 + h > 272:
            self.add_page(); y0 = self.get_y()
        self.fc(GREEN_BG); self.rect(self.l_margin, y0, 174, h, "F")
        self.dc(GREEN); self.set_line_width(0.8)
        self.line(self.l_margin, y0, self.l_margin, y0 + h)
        self.set_line_width(0.2)
        self.set_xy(self.l_margin + 5, y0 + 3)
        self.set_font("Helvetica", "B", 8.5); self.tc(GREEN)
        self.cell(0, 5, clean(f"[EXAMPLE]  {title}"),
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_x(self.l_margin + 5)
        self.set_font("Helvetica", "", 9); self.tc(DARK)
        self.multi_cell(165, 5.2, clean(text)); self.ln(4)

    def why_box(self, title, text):
        h = max(18, len(text) // 65 * 5.5 + 20)
        y0 = self.get_y()
        if y0 + h > 272:
            self.add_page(); y0 = self.get_y()
        self.fc(PURPLE_BG); self.rect(self.l_margin, y0, 174, h, "F")
        self.dc(PURPLE); self.set_line_width(0.8)
        self.line(self.l_margin, y0, self.l_margin, y0 + h)
        self.set_line_width(0.2)
        self.set_xy(self.l_margin + 5, y0 + 3)
        self.set_font("Helvetica", "B", 8.5); self.tc(PURPLE)
        self.cell(0, 5, clean(f"[WHY WE USE THIS]  {title}"),
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_x(self.l_margin + 5)
        self.set_font("Helvetica", "", 9); self.tc(DARK)
        self.multi_cell(165, 5.2, clean(text)); self.ln(4)

    def upgrade_box(self, old, new, reason):
        text = f"OLD: {old}\nNEW: {new}\nWHY: {reason}"
        h = max(24, len(text) // 60 * 5.5 + 26)
        y0 = self.get_y()
        if y0 + h > 272:
            self.add_page(); y0 = self.get_y()
        self.fc(TEAL_BG); self.rect(self.l_margin, y0, 174, h, "F")
        self.dc(TEAL); self.set_line_width(0.8)
        self.line(self.l_margin, y0, self.l_margin, y0 + h)
        self.set_line_width(0.2)
        self.set_xy(self.l_margin + 5, y0 + 3)
        self.set_font("Helvetica", "B", 8.5); self.tc(TEAL)
        self.cell(0, 5, "[WHAT CHANGED AND WHY]",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_x(self.l_margin + 5)
        self.set_font("Helvetica", "B", 8.5); self.tc(RED)
        self.cell(0, 5.5, clean(f"OLD:  {old}"),
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_x(self.l_margin + 5)
        self.set_font("Helvetica", "B", 8.5); self.tc(GREEN)
        self.cell(0, 5.5, clean(f"NEW:  {new}"),
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_x(self.l_margin + 5)
        self.set_font("Helvetica", "", 8.5); self.tc(DARK)
        self.multi_cell(165, 5.2, clean(f"WHY:  {reason}")); self.ln(4)

    def pipeline_step(self, num, name, desc, color=BLUE):
        y0 = self.get_y()
        if y0 + 22 > 272:
            self.add_page(); y0 = self.get_y()
        self.fc(color); self.rect(self.l_margin, y0, 14, 18, "F")
        self.set_font("Helvetica", "B", 14); self.tc(WHITE)
        self.set_xy(self.l_margin, y0 + 2)
        self.cell(14, 14, str(num), align="C")
        self.fc(LGREY); self.rect(self.l_margin + 14, y0, 160, 18, "F")
        self.set_font("Helvetica", "B", 10); self.tc(color)
        self.set_xy(self.l_margin + 17, y0 + 2)
        self.cell(0, 6, clean(name))
        self.set_font("Helvetica", "", 8.5); self.tc(DARK)
        self.set_xy(self.l_margin + 17, y0 + 9)
        self.multi_cell(155, 5, clean(desc))
        self.ln(4)

    def divider(self, color=BLUE, my=3):
        self.dc(color); self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), 210 - self.r_margin, self.get_y())
        self.ln(my)

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


# ==============================================================================
#  BUILD PDF
# ==============================================================================

pdf = PDF()
pdf.cover()

# ==============================================================================
# PAGE 2 – BIG PICTURE + 7-STEP OVERVIEW
# ==============================================================================
pdf.add_page()

pdf.fc(LIGHT_BG); pdf.rect(pdf.l_margin, pdf.get_y(), 174, 40, "F")
pdf.set_xy(pdf.l_margin + 4, pdf.get_y() + 4)
pdf.set_font("Helvetica", "B", 12); pdf.tc(NAVY)
pdf.cell(0, 7, "The Big Picture: What Does This System Do?",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_x(pdf.l_margin + 4); pdf.set_font("Helvetica", "", 9.5); pdf.tc(DARK)
pdf.multi_cell(166, 5.8,
    "This is a face recognition system built for drone (UAV) cameras. "
    "A drone flies over an area and records video of people below. "
    "The system automatically finds every face in the video, works out WHO each person is "
    "by comparing their face against a database of enrolled people, "
    "and displays the name in real-time. "
    "If the person is not in the database it says UNKNOWN instead of guessing wrong.")
pdf.ln(10)

pdf.analogy("Think of It Like a Very Smart Security Guard",
    "A security guard at an airport has a folder of photos of known people.\n"
    "When someone walks by, the guard:\n"
    "  1. Looks at the face  (DETECT)\n"
    "  2. Straightens the photo mentally to match the reference  (ALIGN)\n"
    "  3. Checks it is a real person, not a printed photo  (ANTI-SPOOF)\n"
    "  4. Notes key features: eye spacing, nose shape, jawline  (EXTRACT FEATURES)\n"
    "  5. Compares those features to the folder and finds the closest match  (FAISS MATCH)\n"
    "  6. Only announces the name after seeing the same face in several frames  (TEMPORAL VOTE)\n\n"
    "Our system does exactly this automatically for every video frame in under 1 second.")

pdf.section("", "THE 7-STEP PIPELINE AT A GLANCE", color=NAVY)
pdf.pipeline_step(1, "FACE DETECTION  (SCRFD  det_10g.onnx)",
    "Scan every frame. Draw a box around each face. Upscale tiny faces (<48 px) before processing.",
    BLUE)
pdf.pipeline_step(2, "FACE ALIGNMENT  (InsightFace 5-landmark affine warp)",
    "Rotate and resize each face to a standard 112x112 pixel image so all faces look the same.",
    (28, 120, 150))
pdf.pipeline_step(3, "LIVENESS / ANTI-SPOOFING  (6-signal passive + blink challenge)",
    "Check in ~3 ms that the face belongs to a real live person, not a photo or screen.",
    RED)
pdf.pipeline_step(4, "FEATURE EXTRACTION  (HOG + LBP + Geometry + ArcFace + TTA)",
    "Describe the face using 4 different methods. TTA averages 5 ArcFace variants to reduce noise.",
    GREEN)
pdf.pipeline_step(5, "FEATURE FUSION + PCA  (L2-normalise, concatenate, compress to 99% variance)",
    "Combine all 4 descriptions into one compact number vector. PCA removes noise.",
    ORANGE)
pdf.pipeline_step(6, "FAISS MATCHING  (cosine similarity, threshold 0.35, open-set UNKNOWN)",
    "Compare the 512-D ArcFace vector against all enrolled people. UNKNOWN if best score is too low.",
    PURPLE)
pdf.pipeline_step(7, "TEMPORAL VOTING  (3/5 frames video, 6/10 webcam, fast-confirm at 0.60)",
    "Only announce an identity after several frames agree. A very confident single frame confirms immediately.",
    TEAL)

# ==============================================================================
# PAGE 3 – STEP 1: FACE DETECTION
# ==============================================================================
pdf.add_page()

pdf.section("STEP 1:", "Face Detection  -  Finding the Face", BLUE)
pdf.body(
    "Before the system can work out who someone is, it must first FIND their face in the image. "
    "This step answers WHERE is the face, not WHO is it. "
    "The answer is a bounding box: a rectangle drawn around the face.")

pdf.analogy("Finding a Face is Like Finding Waldo in a Crowd",
    "If you have a photo of 50 people at a party, face detection finds the exact "
    "rectangle around each person's face. "
    "Once we have that box we can zoom in on just that face and examine it closely.")

pdf.sub("What We Use: SCRFD  (det_10g.onnx)")
pdf.body(
    "We use a detector called SCRFD, made by InsightFace in 2021. "
    "It is a deep learning model trained on millions of face photos. "
    "Along with the face box it also finds 5 key points on each face: "
    "left eye, right eye, nose tip, left mouth corner, right mouth corner. "
    "These 5 points are used in Step 2 for alignment.")

pdf.upgrade_box(
    "MTCNN - older detector, also found faces and landmarks",
    "SCRFD det_10g.onnx - same job but 3x faster and much better on tiny/tilted faces",
    "UAV faces at 20-30 metres altitude can be as small as 17-30 pixels wide. "
    "MTCNN missed most of these tiny faces. "
    "SCRFD was specifically designed for small and low-quality faces, "
    "making it far more reliable for drone footage.")

pdf.sub("The Key Fix: Send Full-Resolution Frames")
pdf.body(
    "SCRFD processes images on a 640x640 internal grid. "
    "If you shrink a 1920x1080 frame before sending it, a 30-pixel face becomes only 10 pixels "
    "and the detector misses it completely.\n\n"
    "Our fix: send the FULL 1920x1080 frame directly. "
    "This single change increased face detection from 0 out of 305 frames to 277 out of 305. "
    "A jump from 0% to 91% detection rate.")

pdf.example("Real Test Result",
    "Video: UAV footage at approximately 20-30 metres altitude\n"
    "Before fix: 0 faces detected across the entire video\n"
    "After full-resolution fix: 277 out of 305 frames had a face detected\n"
    "Improvement: 0% to 91% from one single change.")

pdf.sub("Tiny Face Upscaling")
pdf.body(
    "If the detected face box is smaller than 48 pixels wide, "
    "there are not enough pixels for the later steps to work well. "
    "We automatically enlarge such tiny crops to 112 pixels using bicubic interpolation "
    "before passing them to the next step. "
    "Think of it like zooming in on a blurry photo: it does not add detail "
    "but gives the later steps more pixels to work with.")

# ==============================================================================
# PAGE 4 – STEP 2: ALIGNMENT
# ==============================================================================
pdf.add_page()

pdf.section("STEP 2:", "Face Alignment  -  Standardising Every Face", (28, 120, 150))
pdf.body(
    "After detection gives us a face box, the faces inside can be tilted, "
    "at different sizes, and at different positions. "
    "Alignment fixes all of this: it rotates, scales, and crops every face "
    "to a standard fixed size of 112 x 112 pixels, always in the same orientation.")

pdf.analogy("Alignment is Like Putting Every ID Card Photo in the Same Format",
    "When you get a passport photo taken, the photographer asks you to:\n"
    "  - Look straight ahead\n"
    "  - Keep your face centred\n"
    "  - Have your eyes at a certain height in the photo\n\n"
    "Face alignment does the same thing automatically. "
    "It uses the 5 landmark points from Step 1 to calculate a transformation "
    "that warps the face into a standard position. "
    "After alignment both eyes are always at the same pixel coordinates in every image. "
    "This makes comparisons much more accurate.")

pdf.sub("Why Alignment Matters for UAV Images")
pdf.body(
    "UAV cameras look down at people from above at angles of 20-60 degrees. "
    "The face is often tilted, squished vertically, and at a different angle "
    "than the front-facing gallery photos used during enrolment.\n\n"
    "Without alignment, comparing a tilted aerial face against a straight ID photo "
    "would give a very low similarity score even for the same person. "
    "Alignment partially corrects for this before the comparison happens.")

pdf.example("Before vs After Alignment",
    "BEFORE: A face detected at 45-degree tilt, 200x150 pixels in size\n"
    "AFTER:  Same face, rotated to upright, cropped and resized to exactly 112x112 pixels\n\n"
    "Now this image can be fairly compared against enrolled gallery photos "
    "which were aligned the same way at enrolment time.")

# ==============================================================================
# PAGE 5 – STEP 3: LIVENESS / ANTI-SPOOFING
# ==============================================================================
pdf.add_page()

pdf.section("STEP 3:", "Liveness / Anti-Spoofing  -  Is This a Real Person?", RED)
pdf.body(
    "Before spending any time on feature extraction or matching, the system checks "
    "whether the face belongs to a LIVE PERSON or a FAKE "
    "(a printed photo held up, or a face shown on a phone screen). "
    "This check runs in about 3 milliseconds and needs no extra model files.")

pdf.analogy("Anti-Spoofing is Like Checking if a Coin is Real or Counterfeit",
    "A cashier can tell a fake coin from a real one by:\n"
    "  - Feeling the texture (a fake coin feels wrong)\n"
    "  - Looking at the edge patterns (different from a genuine coin)\n"
    "  - Checking the reflections (plastic or paper reflects differently)\n\n"
    "Our passive liveness detector uses SIX such checks fused into one score.\n"
    "Score above 0.45 = LIVE.  Score below 0.45 = SPOOF.")

pdf.sub("The Six Passive Checks")
pdf.body("Check 1 - LBP Texture Complexity:")
pdf.bullet("Looks at the micro-texture pattern of the face region")
pdf.bullet("Real skin has complex, irregular texture  (score 0.7-0.9)")
pdf.bullet("A printed photo has smoother, more uniform texture")
pdf.ln(2)
pdf.body("Check 2 - FFT Frequency Analysis:")
pdf.bullet("Analyses the frequency content of the face image")
pdf.bullet("Real faces have a smooth, natural frequency spread")
pdf.bullet("Screens have a regular pixel grid that creates tell-tale frequency peaks")
pdf.ln(2)
pdf.body("Check 3 - Gradient Coherence:")
pdf.bullet("Measures how smoothly brightness changes across the face")
pdf.bullet("Real 3D faces have structured depth gradients")
pdf.bullet("A flat printed photo has nearly uniform gradients with paper-texture edges")
pdf.ln(2)
pdf.body("Checks 4, 5, 6 - Colour, Specular Highlight, Chroma Noise:")
pdf.bullet("Colour: screens over-saturate colours; real skin stays in a natural HSV range")
pdf.bullet("Specular: screens produce a hard bright hotspot. Real skin has soft diffuse highlights")
pdf.bullet("Chroma: screen pixels sit on a regular grid making the chroma unnaturally smooth")
pdf.ln(2)

pdf.example("Passive Liveness in Action",
    "Scenario A: Someone holds a printed photo of Siddhant in front of the camera\n"
    "  Liveness score: 0.28  (below 0.45 threshold)\n"
    "  Decision: SPOOF DETECTED. ArcFace is NOT run. Gallery search is NOT attempted.\n\n"
    "Scenario B: Real Siddhant stands in front of the camera\n"
    "  Liveness score: 0.79  (above 0.45 threshold)\n"
    "  Decision: LIVE FACE. Proceed to feature extraction and matching.")

pdf.sub("Active Blink Challenge (Webcam / Live Mode Only)")
pdf.body(
    "For the live webcam mode the system adds a second liveness layer: a blink challenge. "
    "The user is asked to blink within 7 seconds. "
    "A printed photo cannot blink. A video replay cannot blink at a random unpredictable moment. "
    "Only a live person can pass.\n\n"
    "How it works: Eye Aspect Ratio (EAR) is measured using 6 dlib landmark points around each eye. "
    "An open eye gives EAR around 0.30-0.40. "
    "A blink makes EAR drop below 0.25 for 2 or more consecutive frames.")

pdf.why_box("Why Two Layers?",
    "Passive liveness (6 checks) runs silently on any image: photo uploads, video frames, UAV footage. "
    "No action is needed from the subject. Perfect for drone surveillance.\n\n"
    "The active blink challenge is for the webcam login mode only, "
    "where a known person is deliberately trying to authenticate. "
    "It defeats even high-quality screen replays that might fool passive texture analysis.\n\n"
    "Together the system works in both modes: "
    "passive for automatic drone use, active for live login sessions.")

pdf.sub("Embedding Cache  -  Faster Retraining")
pdf.body(
    "Every time a face image is processed for retraining, "
    "its ArcFace embedding is saved to a cache file keyed by the MD5 hash of the image. "
    "On the next retrain, if the image has not changed, the saved embedding is reused "
    "instead of re-running ArcFace (about 0.1 seconds per image).\n\n"
    "Result: first retrain takes about 15 seconds. "
    "All later retrains take about 2 seconds. That is a 10x speedup.")

# ==============================================================================
# PAGE 6 – STEP 4A: HOG FEATURES
# ==============================================================================
pdf.add_page()

pdf.section("STEP 4A:", "HOG Features  -  Reading the Shape of a Face", GREEN)
pdf.body(
    "HOG stands for Histogram of Oriented Gradients. "
    "Despite the technical name it does a simple thing: "
    "it looks at the direction of edges and lines in the face image "
    "and creates a compact description of the face shape.")

pdf.analogy("HOG is Like a Sketch Artist's Notes",
    "A police sketch artist does not draw every detail of a face. "
    "They note the direction and angle of key edges: "
    "the curve of the eyebrow, the slant of the nose, the angle of the jawline.\n\n"
    "HOG does exactly this mathematically. "
    "It divides the face into a grid of small blocks (8x8 pixels each). "
    "In each block it measures the direction of brightness changes (edges) "
    "and records how many edges point in each direction.\n\n"
    "The result is a list of numbers describing the edge directions across the whole face. "
    "A compact sketch of the face shape.")

pdf.sub("What HOG is Good At")
pdf.bullet("Captures the overall shape and structure of the face")
pdf.bullet("Works well even when the lighting changes")
pdf.bullet("Describes eyebrows, nose bridge, jawline, and lip shape")
pdf.bullet("Very fast to compute - adds almost no processing time")
pdf.ln(2)

pdf.sub("What HOG Struggles With")
pdf.bullet("Very blurry images - edges become unclear and HOG becomes noisy")
pdf.bullet("Very small faces - not enough pixels to compute meaningful edges")
pdf.bullet("Large pose changes - the shape profile looks very different from the side")
pdf.ln(2)

pdf.why_box("Why We Include HOG",
    "HOG captures shape information that deep learning features sometimes miss. "
    "When combined with the other features it adds complementary shape information "
    "that helps especially in moderate blur and lighting variation, "
    "both of which are common in UAV footage.")

# ==============================================================================
# PAGE 7 – STEP 4B: LBP FEATURES
# ==============================================================================
pdf.add_page()

pdf.section("STEP 4B:", "LBP Features  -  Reading the Texture of a Face", GREEN)
pdf.body(
    "LBP stands for Local Binary Patterns. "
    "It captures the TEXTURE of the face: skin pores, wrinkle patterns, stubble, "
    "and the fine grain of facial skin. "
    "Unlike HOG which looks at edges, LBP looks at tiny local patterns.")

pdf.analogy("LBP is Like Feeling the Texture of a Surface with Your Fingertips",
    "Close your eyes and touch a brick wall versus a smooth glass pane. "
    "You can tell them apart purely by texture without seeing the shape.\n\n"
    "LBP does something similar for face images. "
    "For every pixel in the face it looks at its 8 surrounding neighbours "
    "and creates a simple code: 1 if the neighbour is brighter, 0 if it is darker. "
    "This gives an 8-bit code for each pixel.\n\n"
    "These codes are counted up into histograms for different regions of the face. "
    "That histogram is the LBP feature vector.")

pdf.sub("What LBP is Good At")
pdf.bullet("Completely unaffected by overall brightness changes")
pdf.bullet("Works the same in artificial lighting and natural sunlight")
pdf.bullet("Very fast to compute")
pdf.bullet("Captures fine texture differences between people")
pdf.ln(2)

pdf.example("LBP in Action",
    "Person A has smooth skin with no beard.\n"
    "Person B has stubble and slight wrinkles.\n\n"
    "Even if both faces are photographed under different lighting, "
    "LBP produces different texture patterns for each person. "
    "The relative brightness patterns around each pixel stay the same "
    "regardless of whether the overall image is bright or dark. "
    "This makes LBP unusually reliable for outdoor UAV footage "
    "where lighting changes rapidly.")

pdf.why_box("Why We Include LBP",
    "UAV footage is captured outdoors where lighting changes dramatically: "
    "morning sun, afternoon shadows, cloudy versus sunny conditions. "
    "LBP is almost completely immune to these changes because it only looks at "
    "RELATIVE brightness (is this pixel brighter than its neighbour?). "
    "This makes it a valuable complement to HOG and ArcFace.")

# ==============================================================================
# PAGE 8 – STEP 4C: GEOMETRY FEATURES
# ==============================================================================
pdf.add_page()

pdf.section("STEP 4C:", "Geometry Features  -  Reading the Structure of a Face", GREEN)
pdf.body(
    "Geometry features use 68 precise landmark points on the face "
    "(corners of eyes, tip of nose, corners of mouth, jawline points, etc.) "
    "to compute distances, ratios, and angles between those points. "
    "This creates a mathematical blueprint of the face structure.")

pdf.analogy("Geometry Features are Like a Facial Measurement Chart",
    "Forensic artists can identify people by measuring facial proportions:\n"
    "  - How far apart are the eyes relative to the face width?\n"
    "  - How long is the nose relative to the face height?\n"
    "  - How wide is the jaw compared to the cheekbones?\n\n"
    "These proportions are unique to each person and stay relatively stable "
    "across different photos of the same person. "
    "We use dlib to detect 68 points and compute all pairwise distances and angles, "
    "normalised for scale and rotation, to create this blueprint.")

pdf.sub("What Geometry Features Capture")
pdf.bullet("Eye width and separation  (inter-ocular distance)")
pdf.bullet("Nose length and width relative to face")
pdf.bullet("Mouth width and lip thickness ratio")
pdf.bullet("Jawline shape and chin position")
pdf.bullet("Cheekbone width versus forehead width")
pdf.ln(2)

pdf.why_box("Why We Include Geometry",
    "Geometry features are completely independent of texture and lighting. "
    "They capture the structural proportions determined by bone structure, "
    "things that do not change with age, lighting, expression, or image quality.\n\n"
    "When ArcFace struggles with a very blurry or compressed image, "
    "geometric ratios often remain recognisable because the face structure "
    "is still present even in degraded images. "
    "Geometry acts as a robust backup feature.")

# ==============================================================================
# PAGE 9 – STEP 4D: ARCFACE + TTA
# ==============================================================================
pdf.add_page()

pdf.section("STEP 4D:", "ArcFace  -  The Deep Learning Brain", GREEN)
pdf.body(
    "ArcFace is the most powerful of the four features. "
    "It is a deep neural network (ResNet-50, 50 layers) "
    "trained on 600,000 different people's face photos. "
    "It converts any face image into a list of 512 numbers "
    "that uniquely represent that person's identity.")

pdf.analogy("ArcFace is Like Converting a Face into a Fingerprint",
    "A fingerprint gives a unique code for each person. "
    "No matter how you hold your finger, the fingerprint produces roughly the same code.\n\n"
    "ArcFace does the same for faces. Give it a face photo "
    "and it outputs 512 numbers called an embedding vector. "
    "The same person always produces similar 512 numbers. "
    "Different people produce very different 512 numbers.\n\n"
    "The key insight: you do not need to write rules about what makes a face unique. "
    "The network learned this automatically by studying 600,000 people.")

pdf.example("ArcFace Output Example",
    "Photo 1 of Siddhant: [0.12, -0.34, 0.89, 0.22, ...]  (512 numbers)\n"
    "Photo 2 of Siddhant: [0.11, -0.35, 0.87, 0.23, ...]  (very similar!)\n"
    "Photo of Aditi:      [-0.44, 0.21, -0.12, 0.77, ...]  (very different!)\n\n"
    "Similarity between two vectors is measured using cosine similarity:\n"
    "  Same person:      similarity 0.60 to 0.95  (vectors point in almost the same direction)\n"
    "  Different person: similarity 0.05 to 0.30  (vectors point in very different directions)")

pdf.sub("Test-Time Augmentation (TTA)  -  5 Variants, Averaged")
pdf.body(
    "Instead of extracting one ArcFace embedding from the aligned face, "
    "we extract 5 slightly different versions of the same face "
    "(original, mild left crop, mild right crop, slight flip, slight brightness shift) "
    "and average all 5 embeddings together.\n\n"
    "This averaging cancels out random noise from video compression and blur, "
    "giving a cleaner, more reliable embedding. "
    "Result: +3 to 8% recognition accuracy improvement on UAV images.")

pdf.upgrade_box(
    "Single ArcFace embedding per face image",
    "5-variant TTA average embedding",
    "Video compression and motion blur create slightly different noise in each frame. "
    "Averaging 5 variants reduces the effect of that noise. "
    "The computational cost is modest because all 5 variants run in parallel.")

# ==============================================================================
# PAGE 10 – STEP 5: FUSION + PCA
# ==============================================================================
pdf.add_page()

pdf.section("STEP 5:", "Feature Fusion + PCA  -  Combining All 4 Descriptions", ORANGE)
pdf.body(
    "After computing HOG, LBP, Geometry, and ArcFace features separately, "
    "we combine them into a single long vector of numbers. "
    "Before combining, each modality is L2-normalised (scaled to equal importance). "
    "The combined vector can be thousands of numbers long, "
    "so PCA then compresses it while keeping 99% of the important information.")

pdf.analogy("Fusion + PCA is Like Summarising a Long Report into Key Points",
    "If you have a 50-page report, a skilled editor can summarise the most important "
    "findings in a few bullet points without losing the key conclusions. "
    "PCA does this mathematically for number vectors.\n\n"
    "It finds which combinations of numbers carry the most signal (information about identity) "
    "and which carry mostly noise (random variation due to lighting, blur, compression). "
    "It keeps only the informative combinations.")

pdf.sub("Why L2-Normalise Before Combining?")
pdf.body(
    "HOG produces about 2,916 numbers. ArcFace produces 512 numbers. "
    "If we just add them together, the larger HOG vector would completely dominate "
    "simply because it has more numbers, not because it is more informative. "
    "L2-normalisation scales each modality to the same length before combining "
    "so every modality contributes equally.")

pdf.upgrade_box(
    "PCA retaining 95% of variance",
    "PCA retaining 99% of variance",
    "The extra 4% of variance retained carries genuine identity information "
    "that was previously discarded. Raising from 95% to 99% gave +2 to 5% accuracy "
    "on UAV probe images.")

pdf.sub("SVM Classifier  (Closed-Set Secondary Check)")
pdf.body(
    "An SVM (Support Vector Machine) is also trained on the fused PCA features. "
    "It learns a decision boundary between different people's feature clusters. "
    "However SVM always assigns a label, even for complete strangers. "
    "That is why FAISS (Step 6) is used as the primary identifier: "
    "FAISS can say UNKNOWN, SVM cannot.")

# ==============================================================================
# PAGE 11 – STEP 6: FAISS MATCHING
# ==============================================================================
pdf.add_page()

pdf.section("STEP 6:", "FAISS Matching  -  Who Is This Person?", PURPLE)
pdf.body(
    "FAISS (Facebook AI Similarity Search) is the core identification engine. "
    "It stores the 512-number ArcFace embedding of every enrolled person "
    "and instantly finds the closest match when a new face is presented.")

pdf.analogy("FAISS is Like a Very Fast Filing System",
    "Imagine a filing cabinet with a folder for every enrolled person. "
    "Each folder contains a numerical signature of that person's face. "
    "When a new face arrives, FAISS instantly compares its signature against every folder "
    "and tells you which folder is most similar.\n\n"
    "The crucial difference from SVM is the THRESHOLD:\n"
    "  - If the best match similarity is >= 0.35:  'This is Siddhant'\n"
    "  - If the best match similarity is <  0.35:  'UNKNOWN - no match found'\n\n"
    "This threshold is what allows the system to correctly reject strangers "
    "instead of wrongly labelling them as someone enrolled.")

pdf.sub("How Cosine Similarity Works")
pdf.body(
    "The 512-number ArcFace embedding is normalised to unit length. "
    "Cosine similarity measures the angle between two such vectors:\n\n"
    "  1.0 = vectors point in exactly the same direction = same person\n"
    "  0.0 = vectors are perpendicular = unrelated persons\n"
    " -1.0 = opposite directions = maximally different\n\n"
    "In practice genuine pairs (same person) score 0.55 to 0.95. "
    "Impostor pairs (different person) score 0.05 to 0.35.")

pdf.example("FAISS in Action",
    "Query: photo of Siddhant uploaded to the system\n"
    "Gallery: [aditi, siddhant, stuti]\n\n"
    "FAISS results:\n"
    "  Rank 1: siddhant - similarity 0.69  (>= 0.35: ACCEPTED)\n"
    "  Rank 2: stuti    - similarity 0.31\n"
    "  Rank 3: aditi    - similarity 0.29\n"
    "Decision: IDENTIFIED as siddhant  (69% confidence)\n\n"
    "If Rank 1 similarity was 0.28 (below 0.35):\n"
    "Decision: UNKNOWN  (no enrolled person matches closely enough)")

pdf.why_box("Why FAISS Threshold is 0.35  (Not Higher Like 0.60)",
    "In ideal conditions (clear, close-up, frontal photos) a threshold of 0.60 works well. "
    "For UAV footage at 20-30 metres altitude:\n"
    "  - The face is 17-30 pixels wide (very low resolution)\n"
    "  - H.264 video compression adds block artefacts\n"
    "  - Motion blur reduces sharpness\n"
    "  - The camera angle is oblique, not frontal\n\n"
    "All these factors reduce the cosine similarity even for the correct person. "
    "Genuine UAV pairs score 0.35-0.55 instead of 0.65-0.90. "
    "So we lowered the threshold to 0.35 to accept these lower-but-genuine scores "
    "while still rejecting most impostors who score below 0.30.")

pdf.upgrade_box(
    "SVM alone as the primary identifier  (cannot say UNKNOWN for strangers)",
    "FAISS cosine similarity with threshold as the primary identifier",
    "SVM will always pick the closest enrolled person even for a complete stranger. "
    "In surveillance, most faces will be unknown people. "
    "FAISS with a threshold correctly returns UNKNOWN for anyone below the cutoff. "
    "SVM is kept as a secondary signal for the ranked candidates list.")

# ==============================================================================
# PAGE 12 – STEP 7: TEMPORAL VOTING
# ==============================================================================
pdf.add_page()

pdf.section("STEP 7:", "Temporal Voting  -  Confirming Identity Across Frames", TEAL)
pdf.body(
    "When processing a video, the system gets one identification result per frame. "
    "Individual frames can be blurry, partially occluded, or badly compressed, "
    "causing incorrect single-frame predictions. "
    "Temporal voting solves this by waiting for consistent agreement across several frames.")

pdf.analogy("Temporal Voting is Like a Jury in a Court Case",
    "A single juror can make a mistake. "
    "But if most jurors independently reach the same conclusion, "
    "you can be much more confident they are right.\n\n"
    "Two modes:\n"
    "  VIDEO UPLOAD  (5/3 window):  Identity confirmed when 3 out of 5 consecutive frames agree.\n"
    "  LIVE WEBCAM  (10/6 window): Identity confirmed when 6 out of 10 consecutive frames agree.\n\n"
    "FAST CONFIRM: If any single frame scores >= 0.60 (very confident match), "
    "the voter is skipped and identity is confirmed immediately. "
    "This removes the delay at the start of a clear, well-lit video.\n\n"
    "Example with fast-confirm:\n"
    "  Frame 1: Siddhant  (blurry   - score 0.38)\n"
    "  Frame 2: UNKNOWN   (very blurry - score 0.28)\n"
    "  Frame 3: Siddhant  (clear    - score 0.71) -> FAST CONFIRM: SIDDHANT immediately!\n\n"
    "Example without fast-confirm:\n"
    "  Frames 1-3: Siddhant  (scores 0.38, 0.41, 0.45)\n"
    "  Frame 4: Siddhant  (score 0.52) -> 3 out of 4 votes = SIDDHANT confirmed.")

pdf.upgrade_box(
    "Temporal voter window 10/5  (needed 5 out of 10 frames to agree)",
    "5/3 window for video + 6/10 window for webcam + fast-confirm at score >= 0.60",
    "The old 10/5 window was too slow: the system showed UNKNOWN for a long time "
    "before confirming identity in short video clips. "
    "The new 5/3 window confirms faster. "
    "The fast-confirm path at 0.60 removes the delay entirely for clear frames. "
    "The webcam keeps 10/6 for extra stability in the live stream.")

pdf.upgrade_box(
    "PENDING label shown while the temporal voter is still deciding",
    "UNKNOWN shown immediately; confirmed identity shown once threshold is reached",
    "The word PENDING was confusing. "
    "Users did not understand it was an internal waiting state. "
    "Now faces show as UNKNOWN until the voter is confident enough. "
    "This matches user expectations: either identified or unknown, nothing in between.")

pdf.sub("Ranked Candidates  -  The Identity Shortlist")
pdf.body(
    "When the system identifies a face it also shows a ranked list of the next best matches "
    "with their similarity scores. "
    "This lets the user see how confident the system is and who the other candidates are.")

pdf.upgrade_box(
    "Ranked candidates came from SVM  (different source than the main label)",
    "Ranked candidates now come from FAISS  (same source as main label)",
    "Previously the main identity label came from FAISS "
    "but the ranked list below it came from SVM. "
    "These two systems often disagreed: the header might say Siddhant "
    "but rank 1 in the list said Aditi. "
    "Now both use FAISS as the single source of truth so they always agree.")

# ==============================================================================
# PAGE 13 – MULTI-MODAL ADVANTAGE + DATA FLOW TABLE
# ==============================================================================
pdf.add_page()

pdf.section("", "Why Use 4 Features Together?  The Multi-Modal Advantage", (100, 50, 150))
pdf.body(
    "Each of the 4 features is good at different things and struggles with different problems. "
    "Using all 4 together means that when one feature fails, the others compensate.")

pdf.table(
    headers=["Feature", "Best At", "Struggles With", "UAV Benefit"],
    rows=[
        ["HOG",      "Face shape, edge directions", "Heavy blur, tiny faces",
         "Captures shape even at moderate blur"],
        ["LBP",      "Texture, lighting changes",   "Very small faces (<20 px)",
         "Illumination-invariant for outdoor use"],
        ["Geometry", "Face structure, proportions", "Works best on frontal faces",
         "Proportions survive compression and blur"],
        ["ArcFace",  "Deep identity, best accuracy","Compressed/tiny faces lower scores",
         "Most discriminative - backbone of system"],
    ],
    widths=[22, 46, 52, 54]
)

pdf.example("Multi-Modal Compensation in Practice",
    "Siddhant's face at 25 metres altitude in compressed H.264 video:\n\n"
    "ArcFace score: 0.41  (low due to compression, but above 0.35 threshold)\n"
    "HOG: helps PCA separate Siddhant from Aditi based on brow and nose shape\n"
    "LBP: skin texture pattern remains even in the compressed image\n"
    "Geometry: eye separation and nose ratios are still correct\n\n"
    "Combined result: correctly identified with high confidence.\n"
    "ArcFace alone: borderline - might fail at 0.41 without the other signals.\n"
    "HOG/LBP/Geometry alone: would likely fail at this resolution.\n"
    "Combined: reliably passes because multiple independent signals agree.")

pdf.section("", "Complete Data Flow  -  Number by Number", NAVY)
pdf.table(
    headers=["Stage", "Input", "Output", "Size"],
    rows=[
        ["SCRFD Detection",     "1920x1080 video frame", "Face box + 5 landmarks",       "-"],
        ["Alignment",           "Detected face region",  "Aligned face crop",             "112x112 px"],
        ["Liveness Check",      "112x112 face",          "Live / Spoof decision",         "~3 ms"],
        ["HOG Extraction",      "112x112 face",          "HOG feature vector",            "~2,916 numbers"],
        ["LBP Extraction",      "112x112 face",          "LBP histogram vector",          "~1,568 numbers"],
        ["Geometry Extraction", "68 landmarks",          "Distance / angle vector",       "~100 numbers"],
        ["ArcFace + TTA",       "112x112 face x5",       "Averaged deep embedding",       "512 numbers"],
        ["L2 Fusion",           "4 separate vectors",    "One combined vector",           "~5,096 numbers"],
        ["PCA Reduction",       "~5,096 numbers",        "Compressed vector (99% var.)",  "~150-300 numbers"],
        ["FAISS Search",        "512-D ArcFace vector",  "Top-k matches + scores",        "-"],
        ["Temporal Vote",       "Per-frame decisions",   "Stable identity or UNKNOWN",    "-"],
    ],
    widths=[44, 50, 52, 28]
)

# ==============================================================================
# PAGE 14 – ONE-PAGE SUMMARY REFERENCE
# ==============================================================================
pdf.add_page()

pdf.fc(NAVY); pdf.rect(pdf.l_margin, pdf.get_y(), 174, 10, "F")
pdf.set_font("Helvetica", "B", 12); pdf.tc(WHITE); pdf.set_x(pdf.l_margin + 3)
pdf.cell(0, 10, "  COMPLETE SYSTEM SUMMARY  -  ONE PAGE REFERENCE")
pdf.tc(DARK); pdf.ln(14)

pdf.sub("The 7 Steps in Simple Words:")
steps_summary = [
    ("1. DETECT",     "SCRFD scans every frame and draws a box around each face. Tiny faces (<48 px) are upscaled."),
    ("2. ALIGN",      "Each face box is rotated, scaled, and cropped to a standard 112x112 pixel image."),
    ("3. ANTI-SPOOF", "6-signal passive liveness rejects printed photos and screens in ~3 ms. Webcam adds blink check."),
    ("4. DESCRIBE",   "HOG (shape) + LBP (texture) + Geometry (proportions) + ArcFace+TTA (deep embedding)."),
    ("5. COMBINE",    "All 4 descriptions are L2-normalised and merged into one vector. PCA compresses to 99% variance."),
    ("6. MATCH",      "FAISS compares the 512-D vector against enrolled people. Score < 0.35 = UNKNOWN."),
    ("7. DECIDE",     "3/5 frames agree (video) or 6/10 (webcam) = announce identity. Score >= 0.60 = instant confirm."),
]
for step, desc in steps_summary:
    pdf.fc(LGREY); pdf.rect(pdf.l_margin, pdf.get_y(), 174, 10, "F")
    pdf.set_font("Helvetica", "B", 9); pdf.tc(NAVY)
    pdf.set_x(pdf.l_margin + 2); pdf.cell(34, 10, clean(step))
    pdf.set_font("Helvetica", "", 9); pdf.tc(DARK)
    pdf.multi_cell(136, 10, clean(desc))
    pdf.ln(1)

pdf.ln(4)
pdf.divider()

pdf.sub("Why This Combination of Tools?")
pdf.table(
    headers=["Component", "Replaced What", "Why Better"],
    rows=[
        ["SCRFD Detector",                    "MTCNN Detector",
         "3x faster, detects small/tilted faces at UAV altitude"],
        ["6-Signal Passive Liveness",         "3-Signal passive check",
         "Added colour, specular, chroma signals; threshold lowered to 0.45"],
        ["Active Blink Challenge",            "No active check",
         "EAR blink detection for webcam mode defeats screen replay attacks"],
        ["FAISS + Threshold 0.35",            "SVM alone",
         "Enables UNKNOWN rejection - SVM always guesses someone"],
        ["ArcFace + TTA  (5 variants)",       "ArcFace single embedding",
         "+3-8% accuracy: average of 5 variants reduces compression noise"],
        ["PCA 99% variance",                  "PCA 95% variance",
         "+2-5% accuracy: more variance retained = better identity separation"],
        ["MD5 Embedding Cache",               "Re-extract every retrain",
         "Retrains drop from ~15 s to ~2 s after first run  (10x speedup)"],
        ["Temporal Voter 5/3 + fast-confirm", "Temporal voter 10/5",
         "Faster confirmation; fast-confirm removes UNKNOWN-at-start delay"],
        ["Full-res SCRFD input",              "Downscaled input",
         "Keeps 17-30 px UAV faces detectable on 640x640 grid"],
        ["FAISS for ranked list",             "SVM for ranked list",
         "Header and ranked candidates now always agree  (same source)"],
    ],
    widths=[48, 46, 80]
)

pdf.divider()
pdf.sub("Key Numbers to Remember:")
pdf.table(
    headers=["Value", "What It Is", "Why This Number"],
    rows=[
        ["512",    "ArcFace embedding size",         "Industry-standard ResNet-50 output dimension"],
        ["0.35",   "FAISS threshold  (UAV tuned)",   "Lowered because UAV faces score lower due to compression"],
        ["112x112","Standard aligned face size",     "ArcFace training input - every face resized to this"],
        ["3/5",    "Temporal voter  (video)",        "3 frames out of 5-frame window; fast-confirm at >= 0.60"],
        ["6/10",   "Temporal voter  (webcam)",       "6 frames out of 10-frame window for live stream"],
        ["99%",    "PCA variance retained",          "Raised from 95% for +2-5% accuracy improvement"],
        ["0.45",   "Liveness threshold",             "6-signal passive check; blink challenge adds hard gate"],
        ["0.25",   "Blink EAR threshold",            "Eye Aspect Ratio below 0.25 for 2+ frames = blink"],
        ["7.0 s",  "Blink challenge timeout",        "User has 7 seconds to blink; failure = session rejected"],
        ["5",      "TTA variants",                   "5 embedding variants averaged for +3-8% accuracy"],
        ["48 px",  "Tiny face upscale threshold",   "Faces below 48 px are upscaled before ArcFace"],
        ["~2 s",   "Retrain time with cache",        "MD5 cache: was ~15 s, now ~2 s after first run  (10x)"],
    ],
    widths=[22, 70, 82]
)

# ==============================================================================
#  SAVE
# ==============================================================================
output = r"c:\Users\lucky\OneDrive\Desktop\Face Recognition\Face_Recognition_Pipeline_Explanation.pdf"
pdf.output(output)
print(f"\nPDF generated: {output}")
print(f"Total pages: {pdf.page}")
