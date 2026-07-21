"""
generate_pipeline_pdf.py
Generates a simple, beginner-friendly explanation of the Face Recognition pipeline.
Run: python generate_pipeline_pdf.py
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
        if self._cover_page: return
        self.fc(NAVY); self.rect(0, 0, 210, 10, "F")
        self.set_font("Helvetica", "B", 7); self.tc(WHITE)
        self.set_y(2)
        self.cell(0, 6, "FACE RECOGNITION UAV PROJECT  -  HOW IT WORKS", align="C")
        self.tc(DARK); self.ln(9)

    def footer(self):
        if self._cover_page: return
        self.set_y(-13)
        self.fc(LGREY); self.rect(0, self.get_y()-1, 210, 16, "F")
        self.set_font("Helvetica", "", 7); self.tc(GREY)
        self.cell(0, 9, f"Page {self.page - 1}  |  Face Recognition Pipeline Explained  |  Simple Language Guide", align="C")
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
        self.set_font("Helvetica", "B", 28); self.tc(WHITE)
        self.cell(0, 14, "HOW IT WORKS", align="C",
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Helvetica", "B", 15); self.tc((160, 190, 255))
        self.cell(0, 9, "The Complete Pipeline Explained in Simple Language",
                  align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(6)
        self.set_font("Helvetica", "", 10); self.tc((190, 210, 255))
        self.set_x(25)
        self.multi_cell(160, 6,
            "Every step of the face recognition system explained with simple words,\n"
            "real examples, and the reason why each component was chosen.\n"
            "Now includes passive liveness detection, FAISS matching, TTA, and temporal voting.",
            align="C")

        # Feature badges
        self.ln(14)
        badges = [
            ("STEP 1", "Detect Face"), ("STEP 2", "Align Face"),
            ("STEP 3", "Anti-Spoof"), ("STEP 4", "Extract Features"),
            ("STEP 5", "FAISS Match"), ("STEP 6", "Temporal Vote"),
        ]
        bx, bw, by = 18, 30, self.get_y()
        for lbl, val in badges:
            self.fc((30, 55, 140)); self.dc(BLUE)
            self.rect(bx, by, bw, 20, "FD")
            self.set_font("Helvetica", "B", 6); self.tc((130, 160, 255))
            self.set_xy(bx, by+3); self.cell(bw, 5, lbl, align="C")
            self.set_font("Helvetica", "B", 7); self.tc(WHITE)
            self.set_xy(bx, by+9); self.cell(bw, 7, val, align="C")
            bx += bw + 4

        # bottom
        self.fc((10, 20, 70)); self.rect(0, 262, 210, 35, "F")
        self.set_font("Helvetica", "B", 9); self.tc((130, 160, 255))
        self.set_xy(0, 270); self.cell(0, 6, "github.com/TENSEI3011/face-recognition-skewed-images", align="C")
        self.set_font("Helvetica", "", 8); self.tc((100, 130, 200))
        self.set_xy(0, 279); self.cell(0, 6, "Built with SCRFD + ArcFace + FAISS + SVM", align="C")
        self._cover_page = False

    # --- section heading
    def section(self, num, title, color=NAVY):
        self.ln(4)
        if self.get_y() > 248: self.add_page()
        self.fc(color); self.rect(self.l_margin-2, self.get_y(), 176, 9, "F")
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
        self.set_x(x+5); self.multi_cell(0, 5.5, clean(t))

    def analogy(self, title, text):
        """Yellow analogy/example box"""
        h = max(18, len(text)//65 * 5.5 + 20)
        y0 = self.get_y()
        if y0 + h > 272: self.add_page(); y0 = self.get_y()
        self.fc(YELLOW_BG); self.rect(self.l_margin, y0, 174, h, "F")
        self.dc(YELLOW); self.set_line_width(0.8)
        self.line(self.l_margin, y0, self.l_margin, y0+h)
        self.set_line_width(0.2)
        self.set_xy(self.l_margin+5, y0+3)
        self.set_font("Helvetica","B",8.5); self.tc(YELLOW)
        self.cell(0, 5, clean(f"[ANALOGY]  {title}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_x(self.l_margin+5)
        self.set_font("Helvetica","",9); self.tc(DARK)
        self.multi_cell(165, 5.2, clean(text)); self.ln(4)

    def example(self, title, text):
        """Green example box"""
        h = max(18, len(text)//65 * 5.5 + 20)
        y0 = self.get_y()
        if y0 + h > 272: self.add_page(); y0 = self.get_y()
        self.fc(GREEN_BG); self.rect(self.l_margin, y0, 174, h, "F")
        self.dc(GREEN); self.set_line_width(0.8)
        self.line(self.l_margin, y0, self.l_margin, y0+h)
        self.set_line_width(0.2)
        self.set_xy(self.l_margin+5, y0+3)
        self.set_font("Helvetica","B",8.5); self.tc(GREEN)
        self.cell(0, 5, clean(f"[EXAMPLE]  {title}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_x(self.l_margin+5)
        self.set_font("Helvetica","",9); self.tc(DARK)
        self.multi_cell(165, 5.2, clean(text)); self.ln(4)

    def why_box(self, title, text):
        """Purple 'why we use this' box"""
        h = max(18, len(text)//65 * 5.5 + 20)
        y0 = self.get_y()
        if y0 + h > 272: self.add_page(); y0 = self.get_y()
        self.fc(PURPLE_BG); self.rect(self.l_margin, y0, 174, h, "F")
        self.dc(PURPLE); self.set_line_width(0.8)
        self.line(self.l_margin, y0, self.l_margin, y0+h)
        self.set_line_width(0.2)
        self.set_xy(self.l_margin+5, y0+3)
        self.set_font("Helvetica","B",8.5); self.tc(PURPLE)
        self.cell(0, 5, clean(f"[WHY WE USE THIS]  {title}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_x(self.l_margin+5)
        self.set_font("Helvetica","",9); self.tc(DARK)
        self.multi_cell(165, 5.2, clean(text)); self.ln(4)

    def upgrade_box(self, old, new, reason):
        """Teal 'why we upgraded' box"""
        text = f"OLD: {old}\nNEW: {new}\nWHY: {reason}"
        h = max(24, len(text)//60 * 5.5 + 26)
        y0 = self.get_y()
        if y0 + h > 272: self.add_page(); y0 = self.get_y()
        self.fc(TEAL_BG); self.rect(self.l_margin, y0, 174, h, "F")
        self.dc(TEAL); self.set_line_width(0.8)
        self.line(self.l_margin, y0, self.l_margin, y0+h)
        self.set_line_width(0.2)
        self.set_xy(self.l_margin+5, y0+3)
        self.set_font("Helvetica","B",8.5); self.tc(TEAL)
        self.cell(0, 5, "[WHY WE UPGRADED]", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_x(self.l_margin+5)
        self.set_font("Helvetica","B",8.5); self.tc(RED)
        self.cell(0, 5.5, clean(f"OLD:  {old}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_x(self.l_margin+5)
        self.set_font("Helvetica","B",8.5); self.tc(GREEN)
        self.cell(0, 5.5, clean(f"NEW:  {new}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_x(self.l_margin+5)
        self.set_font("Helvetica","",8.5); self.tc(DARK)
        self.multi_cell(165, 5.2, clean(f"WHY:  {reason}")); self.ln(4)

    def pipeline_step(self, num, name, desc, color=BLUE):
        """Visual pipeline step card"""
        y0 = self.get_y()
        if y0 + 22 > 272: self.add_page(); y0 = self.get_y()
        self.fc(color); self.rect(self.l_margin, y0, 14, 18, "F")
        self.set_font("Helvetica","B",14); self.tc(WHITE)
        self.set_xy(self.l_margin, y0+2)
        self.cell(14, 14, str(num), align="C")
        self.fc(LGREY); self.rect(self.l_margin+14, y0, 160, 18, "F")
        self.set_font("Helvetica","B",10); self.tc(color)
        self.set_xy(self.l_margin+17, y0+2)
        self.cell(0, 6, clean(name))
        self.set_font("Helvetica","",8.5); self.tc(DARK)
        self.set_xy(self.l_margin+17, y0+9)
        self.multi_cell(155, 5, clean(desc))
        self.ln(4)

    def divider(self, color=BLUE, my=3):
        self.dc(color); self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), 210-self.r_margin, self.get_y())
        self.ln(my)

    def table(self, headers, rows, widths):
        self.fc(NAVY); self.tc(WHITE); self.set_font("Helvetica","B",8)
        for h,w in zip(headers,widths):
            self.cell(w,7,f" {clean(h)}",border=0,fill=True)
        self.ln()
        self.set_font("Helvetica","",8)
        for i,row in enumerate(rows):
            self.fc(LGREY if i%2==0 else WHITE); self.tc(DARK)
            for c,w in zip(row,widths):
                self.cell(w,6.5,f" {clean(c)}",border=0,fill=True)
            self.ln()
        self.ln(4)


# =============================================================================
#  BUILD PDF
# =============================================================================

pdf = PDF()
pdf.cover()

# =========================================================================
# PAGE 2: OVERVIEW + THE BIG PICTURE
# =========================================================================
pdf.add_page()

# Big picture box
pdf.fc(LIGHT_BG); pdf.rect(pdf.l_margin, pdf.get_y(), 174, 38, "F")
pdf.set_xy(pdf.l_margin+4, pdf.get_y()+4)
pdf.set_font("Helvetica","B",12); pdf.tc(NAVY)
pdf.cell(0, 7, "The Big Picture: What Does This System Do?",
         new_x=XPos.LMARGIN, new_y=YPos.NEXT)
pdf.set_x(pdf.l_margin+4); pdf.set_font("Helvetica","",9.5); pdf.tc(DARK)
pdf.multi_cell(166, 5.8,
    "This system is a face recognition system for a drone (UAV) camera. "
    "When a drone flies over an area, it captures video of people below. "
    "The system automatically finds faces in each video frame, "
    "figures out WHO the person is by comparing their face against a database of enrolled people, "
    "and labels them in real-time. "
    "If the person is not in the database, it says UNKNOWN instead of guessing wrong.")
pdf.ln(10)

pdf.analogy("Think of It Like a Very Smart Security Guard",
    "Imagine a security guard at an airport who has a folder of photos of known people.\n"
    "When someone walks by, the guard:\n"
    "  1. Looks at their face (DETECT)\n"
    "  2. Straightens the photo mentally so it matches their reference photos (ALIGN)\n"
    "  3. Notes key features: eye spacing, nose shape, jawline (EXTRACT FEATURES)\n"
    "  4. Compares those features against the folder and finds the closest match (IDENTIFY)\n"
    "  5. If no match is close enough, says 'I don't recognise this person' (UNKNOWN)\n\n"
    "Our system does exactly this - but automatically, for every video frame, in under 1 second.")

pdf.section("", "THE 6-STEP PIPELINE AT A GLANCE", color=NAVY)
pdf.pipeline_step(1, "FACE DETECTION  (SCRFD Detector)",
    "Find where faces are in the image. Draw a box around each face. Upscale tiny faces (<48px) with bicubic interpolation.", BLUE)
pdf.pipeline_step(2, "FACE ALIGNMENT  (InsightFace 5-landmark)",
    "Straighten and normalise each face to a fixed 112x112 pixel size using affine transform.", (28,120,150))
pdf.pipeline_step(3, "LIVENESS / ANTI-SPOOFING CHECK  [UPDATED]",
    "Passive check: 6-signal fusion (LBP, FFT, gradient, colour, specular, chroma) in ~3ms. Active blink challenge for webcam mode.", RED)
pdf.pipeline_step(4, "FEATURE EXTRACTION  (HOG + LBP + Geometry + ArcFace + TTA)",
    "Describe the face using 4 mathematical modalities. TTA averages 5 variants for +3-8% accuracy.", GREEN)
pdf.pipeline_step(5, "FEATURE FUSION + PCA (99% variance)",
    "Combine all 4 descriptions into one compact vector. PCA raised to 99% variance for +2-5% accuracy.", ORANGE)
pdf.pipeline_step(6, "FAISS MATCHING  (Primary Identifier + Open-Set)",
    "Compare 512-D ArcFace vector against enrolled gallery. Returns UNKNOWN if best similarity < threshold.", PURPLE)
pdf.pipeline_step(7, "TEMPORAL VOTING  (5/3 window + fast-confirm @ 0.60)",
    "Confirm identity only after 3 out of 5 consecutive frames agree. Fast-confirm at 0.60 skips voter for clear frames.", TEAL)

# =========================================================================
# PAGE 3: STEP 1 - FACE DETECTION
# =========================================================================
pdf.add_page()

pdf.section("STEP 1:", "Face Detection - Finding the Face in the Image", BLUE)
pdf.body(
    "Before we can recognise who someone is, we first need to FIND their face in the image. "
    "This is called Face Detection. It answers the question: "
    "WHERE is the face? (not WHO is it - that comes later).")

pdf.analogy("Finding a Face is Like Finding Waldo in a Crowd",
    "If you have a photo of 50 people at a party, face detection finds the exact "
    "rectangular region in the photo where each person's face is located. "
    "It draws a box (bounding box) around every face it finds. "
    "Once we have that box, we can zoom in on just that face and examine it closely.")

pdf.sub("What We Use: SCRFD (det_10g.onnx)")
pdf.body(
    "We use a detector called SCRFD, developed by InsightFace in 2021. "
    "It is a deep learning model trained on millions of face images. "
    "It also detects 5 key landmark points on each face: "
    "left eye, right eye, nose tip, left mouth corner, right mouth corner. "
    "These 5 points are used in the next step for alignment.")

pdf.upgrade_box(
    "MTCNN (old detector) - also found faces and landmarks",
    "SCRFD det_10g.onnx - same job but 3x faster and much better on small/tilted faces",
    "For a UAV flying at 20-30 metres altitude, faces appear very small in the video frame "
    "(sometimes just 17-30 pixels wide). MTCNN missed many of these tiny faces. "
    "SCRFD was specifically designed and tested on very small and low-quality faces, "
    "making it far more reliable for our drone footage scenario.")

pdf.sub("The Key Technical Problem We Solved")
pdf.body(
    "SCRFD works on a 640x640 internal grid. If you give it a 1920x1080 video frame "
    "and it shrinks the image before processing (as many implementations do), "
    "a 30-pixel face becomes only 10 pixels on the grid - too small to detect.\n\n"
    "Our fix: We send the FULL RESOLUTION frame (1920x1080) directly to SCRFD "
    "so the 30-pixel face stays 30 pixels. This single change allowed the system "
    "to detect faces in 277 out of 305 video frames (previously 0 out of 305).")

pdf.example("Real Test Result",
    "Video: Lucky's UAV footage at approximately 20-30 metres altitude\n"
    "Before fix: 0 faces detected across the entire video\n"
    "After full-resolution fix: 277 out of 305 frames had successful face detection\n"
    "That is a 0% to 91% detection rate improvement from one single change.")

pdf.sub("Tiny Face Upscaling (Bonus Fix)")
pdf.body(
    "Even after detecting a face, if the face box is smaller than 48 pixels wide, "
    "it does not contain enough pixel detail for the next steps to work well. "
    "We automatically enlarge (upscale) such small face crops to 112 pixels "
    "using a mathematical technique called bicubic interpolation before processing them. "
    "Think of it like zooming into a blurry photo - it does not add detail, "
    "but it gives the later steps more pixels to work with.")

# =========================================================================
# PAGE 4: STEP 2 - ALIGNMENT
# =========================================================================
pdf.add_page()

pdf.section("STEP 2:", "Face Alignment - Standardising Every Face", (28,120,150))
pdf.body(
    "After detection gives us a face box, the faces in that box can be tilted, "
    "rotated, at different scales, and at different positions. "
    "Alignment fixes all of this: it rotates, scales, and crops every face "
    "to a standard fixed position and size of 112 x 112 pixels.")

pdf.analogy("Alignment is Like Putting Every ID Card Photo in the Same Format",
    "When you get a passport photo taken, the photographer asks you to:\n"
    "  - Look straight ahead\n"
    "  - Keep your face centred\n"
    "  - Have your eyes at a certain height in the photo\n\n"
    "Face alignment does the same thing automatically. "
    "It uses the 5 landmark points (eyes, nose, mouth corners) detected in Step 1 "
    "to calculate a mathematical transformation that warps the face image "
    "into a standard position - both eyes always at the same pixel coordinates, "
    "same scale, same orientation. This makes comparison much easier and more accurate.")

pdf.sub("Why Alignment is Critical for UAV Images")
pdf.body(
    "UAV cameras look down at people from above at angles of 20-60 degrees. "
    "This means the face is often tilted, foreshortened (squished vertically), "
    "and at a different angle than the front-facing gallery photos.\n\n"
    "Without alignment, comparing a tilted aerial face against a straight ID photo "
    "would give a very low similarity score even for the same person. "
    "Alignment partially corrects for this by mapping detected facial landmarks "
    "to fixed reference positions before the comparison happens.")

pdf.example("Before vs After Alignment",
    "BEFORE alignment: A face detected at 45-degree tilt, 200x150 pixels in size\n"
    "AFTER alignment: Same face, rotated to upright, cropped and resized to exactly 112x112 pixels\n\n"
    "Now this aligned image can be fairly compared against the enrolled gallery photos "
    "which were also aligned the same way at enrollment time.")

# =========================================================================
# PAGE 5: STEP 3 - HOG FEATURES
# =========================================================================
pdf.add_page()

pdf.section("STEP 3A:", "HOG Features - Reading the Shape of a Face", GREEN)
pdf.body(
    "HOG stands for Histogram of Oriented Gradients. "
    "Despite the complicated name, it does a simple thing: "
    "it looks at the direction of edges and lines in the face image "
    "and creates a compact description of the face shape.")

pdf.analogy("HOG is Like a Sketch Artist's Notes",
    "A police sketch artist does not draw every detail of a face. "
    "They note the direction and angle of key edges: "
    "the curve of the eyebrow, the slant of the nose, the angle of the jawline.\n\n"
    "HOG does exactly this mathematically. It divides the face image into a grid of "
    "small cells (e.g., 8x8 pixel blocks). In each cell, it measures the direction "
    "of brightness changes (called gradients or edges) and records how many "
    "edges point in each direction (0 degrees, 45 degrees, 90 degrees, etc.).\n\n"
    "The result is a list of numbers describing the edge directions throughout the face - "
    "a compact 'sketch' of the face shape.")

pdf.sub("What HOG is Good At")
pdf.bullet("Captures the overall shape and structure of the face")
pdf.bullet("Works well even when the lighting changes (bright vs dark room)")
pdf.bullet("Captures the shape of eyes, nose bridge, jawline, and lips")
pdf.bullet("Very fast to compute - adds almost no time to processing")
pdf.ln(2)

pdf.sub("What HOG Struggles With")
pdf.bullet("Very blurry images - edges become unclear, HOG becomes noisy")
pdf.bullet("Very small faces - not enough pixels to compute meaningful gradients")
pdf.bullet("Large pose changes - the shape profile looks very different from the side")
pdf.ln(2)

pdf.why_box("Why We Include HOG",
    "HOG is a classical computer vision feature that has been proven to work for over 20 years. "
    "It captures shape information that deep learning features sometimes miss. "
    "When combined with other features, it adds complementary information about face structure "
    "that improves accuracy especially in moderate blur and lighting variation - "
    "conditions that are common in UAV footage.")

# =========================================================================
# PAGE 6: LBP FEATURES
# =========================================================================
pdf.add_page()

pdf.section("STEP 3B:", "LBP Features - Reading the Texture of a Face", GREEN)
pdf.body(
    "LBP stands for Local Binary Patterns. "
    "It captures the TEXTURE of the face - things like skin pores, "
    "wrinkle patterns, stubble, and the fine grain of facial skin. "
    "Unlike HOG which looks at edges, LBP looks at tiny local patterns.")

pdf.analogy("LBP is Like Feeling the Texture of a Surface with Your Fingertips",
    "If you close your eyes and touch a brick wall vs a smooth glass pane, "
    "you can tell them apart purely by texture without seeing the shape. "
    "LBP does something similar for face images.\n\n"
    "For every single pixel in the face image, LBP looks at its 8 surrounding neighbours "
    "and creates a simple code: '1' if the neighbour is brighter than the centre pixel, "
    "'0' if it is darker. This gives an 8-bit binary code (e.g., 10110100) for each pixel.\n\n"
    "These codes are collected into histograms (counts of how often each code appears "
    "in different regions of the face). This histogram is the LBP feature vector.")

pdf.sub("What LBP is Good At")
pdf.bullet("Completely unaffected by overall brightness changes (dark vs bright room)")
pdf.bullet("Works the same in artificial lighting and natural sunlight")
pdf.bullet("Very fast computation - no complex maths required")
pdf.bullet("Captures fine texture differences between people")
pdf.ln(2)

pdf.example("LBP in Action",
    "Person A has smooth skin with no beard.\n"
    "Person B has stubble and slight wrinkles.\n\n"
    "Even if both faces are captured under different lighting, "
    "LBP will produce different texture patterns for each person. "
    "The relative brightness patterns around each pixel stay the same "
    "regardless of whether the overall image is bright or dark. "
    "This makes LBP unusually robust for outdoor UAV footage "
    "where lighting changes rapidly.")

pdf.why_box("Why We Include LBP",
    "UAV footage is captured outdoors where lighting changes dramatically - "
    "morning sun, afternoon shadows, cloudy vs sunny conditions. "
    "LBP is almost completely immune to these lighting changes because it only looks "
    "at RELATIVE brightness (is this pixel brighter or darker than its neighbour?). "
    "This makes it a valuable complement to HOG and ArcFace which can both be affected by "
    "extreme lighting changes.")

# =========================================================================
# PAGE 7: GEOMETRY FEATURES
# =========================================================================
pdf.add_page()

pdf.section("STEP 3C:", "Geometry Features - Reading the Structure of a Face", GREEN)
pdf.body(
    "Geometry features use 68 precise landmark points detected on the face "
    "(corners of eyes, tip of nose, corners of mouth, jawline points, etc.) "
    "to compute distances, ratios, and angles between these points. "
    "This creates a mathematical 'blueprint' of the face structure.")

pdf.analogy("Geometry Features are Like a Facial Measurement Chart",
    "Forensic artists and anthropologists can identify people by measuring facial proportions:\n"
    "  - How far apart are the eyes relative to the face width?\n"
    "  - How long is the nose relative to the face height?\n"
    "  - What is the ratio of the upper lip to the lower lip?\n"
    "  - How wide is the jaw compared to the cheekbones?\n\n"
    "These proportions are unique to each person and stay relatively stable "
    "across different photos of the same person. "
    "Our system uses dlib to detect 68 points and then computes all pairwise distances "
    "and angles, normalised for scale and rotation, to create this 'measurement blueprint'.")

pdf.sub("What Geometry Features Capture")
pdf.bullet("Eye width and separation (inter-ocular distance)")
pdf.bullet("Nose length and width relative to face")
pdf.bullet("Mouth width and lip thickness ratio")
pdf.bullet("Jawline shape and chin position")
pdf.bullet("Cheekbone width vs forehead width")
pdf.ln(2)

pdf.why_box("Why We Include Geometry",
    "Geometry features are completely independent of texture and lighting. "
    "They capture the structural proportions of the face that are determined by bone structure - "
    "things that do not change with age, lighting, expression changes, or image quality.\n\n"
    "When ArcFace struggles with a very blurry or compressed image, "
    "the geometric ratios often remain recognisable because the face structure "
    "is still present even in degraded images. "
    "This makes geometry a robust 'backup' feature.")

pdf.example("Geometry Feature Example",
    "Siddhant's facial measurements:\n"
    "  - Eye separation / Face width ratio: 0.42\n"
    "  - Nose length / Face height ratio: 0.31\n"
    "  - Mouth width / Eye separation ratio: 0.88\n\n"
    "These ratios stay approximately constant whether the photo is taken:\n"
    "  - From 5 metres or 25 metres altitude\n"
    "  - In bright sunlight or shade\n"
    "  - From slightly to the side or slightly from above")

# =========================================================================
# PAGE 8: ARCFACE FEATURES
# =========================================================================
pdf.add_page()

pdf.section("STEP 3D:", "ArcFace - The Deep Learning Brain", GREEN)
pdf.body(
    "ArcFace is the most powerful of the four features. "
    "It is a deep neural network with 50 layers (ResNet-50) "
    "trained on 600,000 different people's face photos. "
    "It converts any face image into a list of 512 numbers "
    "that uniquely represent that person's identity.")

pdf.analogy("ArcFace is Like Converting a Face into a Fingerprint",
    "A fingerprint is a unique numerical code for each person. "
    "No matter how you hold your finger, or how clean or dirty it is, "
    "the fingerprint always produces roughly the same code.\n\n"
    "ArcFace does the same for faces. You give it a face photo, "
    "and it outputs 512 numbers (called an 'embedding vector'). "
    "The same person always produces similar 512 numbers. "
    "Different people produce very different 512 numbers.\n\n"
    "The key insight: you do not need to write rules about what makes a face unique. "
    "The network learned this automatically by studying 600,000 people.")

pdf.sub("How ArcFace Was Trained")
pdf.body(
    "ArcFace was trained with a clever trick called Angular Margin Loss. "
    "During training, the network was forced to separate different people's "
    "embeddings by at least a fixed angle on a mathematical sphere. "
    "This makes the 512-number outputs very discriminative: "
    "same person = vectors that point in nearly the same direction, "
    "different person = vectors pointing in very different directions.")

pdf.example("ArcFace Output Example",
    "Photo 1 of Siddhant: ArcFace outputs [0.12, -0.34, 0.89, 0.22, ...] (512 numbers)\n"
    "Photo 2 of Siddhant: ArcFace outputs [0.11, -0.35, 0.87, 0.23, ...] (very similar!)\n"
    "Photo of Aditi:      ArcFace outputs [-0.44, 0.21, -0.12, 0.77, ...] (very different!)\n\n"
    "The similarity between two 512-number vectors is measured using cosine similarity:\n"
    "  Same person: cosine similarity = 0.60 to 0.95 (close to 1.0 = identical direction)\n"
    "  Different person: cosine similarity = 0.05 to 0.30 (close to 0.0 = different direction)")

pdf.upgrade_box(
    "Old approach: Hand-crafted HOG/LBP only (good for simple cases)",
    "ArcFace deep embedding (trained on 600K people, 512-D vector) + TTA",
    "Hand-crafted features require humans to decide what face features matter. "
    "ArcFace learned automatically from millions of examples what the most "
    "discriminative face features are. It consistently outperforms hand-crafted "
    "features by 15-30% on challenging datasets. "
    "We keep HOG, LBP, and Geometry alongside ArcFace because they provide "
    "complementary information that helps when ArcFace struggles (e.g., very small faces).\n\n"
    "Test-Time Augmentation (TTA): Instead of extracting one ArcFace embedding, "
    "we extract 5 variants (original + slight crops/flips) and average them. "
    "This reduces the effect of compression artefacts, giving +3-8% accuracy on UAV images.")

# =========================================================================
# PAGE 8B: LIVENESS DETECTION [NEW]
# =========================================================================
pdf.add_page()

pdf.section("STEP 3 (UPDATED):", "Liveness / Anti-Spoofing - Rejecting Fake Faces", RED)
pdf.body(
    "Before spending any time on feature extraction or matching, the system first checks "
    "if the face in the image belongs to a LIVE PERSON or a FAKE (e.g., a printed photo held up, "
    "or someone showing a face on a phone screen). This check happens in about 3 milliseconds "
    "and requires NO extra model files to download.")

pdf.analogy("Anti-Spoofing is Like Checking if a Coin is Real or Counterfeit",
    "A trained cashier can tell a fake coin from a real one by:\n"
    "  - Feeling the texture (a fake coin feels wrong)\n"
    "  - Looking at the edge patterns (different from a genuine coin)\n"
    "  - Checking the reflections (plastic or paper reflects differently)\n\n"
    "Our passive liveness detector uses SIX such tests fused into one score:\n"
    "  1. LBP TEXTURE ENTROPY: Real skin has complex micro-patterns. A print is smoother.\n"
    "  2. FFT FREQUENCY PEAK: Real faces have a natural frequency distribution. Screens/prints have pixel/dot grids.\n"
    "  3. GRADIENT COHERENCE: Real faces have coherent brightness changes. Printed faces do not.\n"
    "  4. COLOUR HISTOGRAM: Real skin has a natural HSV colour distribution. Screens can shift this.\n"
    "  5. SPECULAR REFLECTION: Screens have characteristic bright specular spots. Real faces do not.\n"
    "  6. CHROMA NOISE: Printed faces have different chroma (colour) noise compared to real faces.")

pdf.sub("What the Six Tests Measure")
pdf.body("Test 1 - LBP Texture Entropy:")
pdf.bullet("Divides the face into regions and measures how 'complex' the texture is")
pdf.bullet("Real skin: high complexity (0.7-0.9 on a 0-1 scale)")
pdf.bullet("Printed photo: lower complexity because printing smooths details")
pdf.ln(2)
pdf.body("Test 2 - FFT Frequency Peak Analysis:")
pdf.bullet("Analyses the frequency content of the face image")
pdf.bullet("Real faces have a smooth natural frequency distribution")
pdf.bullet("Screens have a regular pixel grid pattern; prints have a dot-matrix pattern")
pdf.ln(2)
pdf.body("Test 3 - Gradient Coherence:")
pdf.bullet("Measures how smoothly the brightness changes across the face")
pdf.bullet("Real faces: natural, coherent gradient flow")
pdf.bullet("Printed photo: introduces artificial edges from the paper texture")
pdf.ln(2)
pdf.body("Tests 4-6 - Colour, Specular & Chroma (NEW):")
pdf.bullet("Colour histogram: checks HSV skin-tone distribution (screens shift colours)")
pdf.bullet("Specular reflection: screens reflect bright spots in characteristic ways real faces do not")
pdf.bullet("Chroma noise: the fine colour noise pattern differs between real skin and printed/screen faces")
pdf.ln(2)

pdf.example("Passive Liveness Detection in Action",
    "Scenario: Someone holds up a photo of Siddhant on their phone in front of the camera\n\n"
    "Liveness score: 0.28 (below 0.45 threshold)\n"
    "Decision: SPOOF DETECTED - face labelled as 'SPOOF' with red box\n"
    "ArcFace embedding: NOT computed (saves time)\n"
    "Gallery match: NOT attempted\n\n"
    "Scenario: Real Siddhant stands in front of the camera\n"
    "Liveness score: 0.79 (above 0.45 threshold)\n"
    "Decision: LIVE FACE - proceed to feature extraction and matching")

pdf.sub("Active Blink Challenge (Webcam / Live Mode Only) [NEW]")
pdf.body(
    "For the live webcam mode, the system adds a second, active liveness layer: "
    "a blink challenge. The user is asked to blink within 7 seconds. "
    "A static printed photo cannot blink. A video replay cannot blink at a random, "
    "unpredictable moment. Only a live person can pass.\n\n"
    "Technical method: Eye Aspect Ratio (EAR) by Soukupova and Cech (2016).\n"
    "  EAR = (|p2-p6| + |p3-p5|) / (2 x |p1-p4|)\n"
    "  where p1...p6 are the six dlib 68-point eye landmark coordinates.\n"
    "  Open eye: EAR ~ 0.30-0.40.  Blink: EAR drops below 0.25 for 2+ frames.")

pdf.example("Blink Challenge Flow",
    "1. Webcam session starts. Frontend receives a session ID from /api/liveness/blink/new.\n"
    "2. Each webcam frame is sent to /api/liveness/blink/frame. Server returns EAR + blink count.\n"
    "3. If blink_count >= 1 within 7 seconds: challenge PASSED -> proceed to face matching.\n"
    "4. If 7 seconds elapse with 0 blinks: challenge FAILED -> session rejected.\n"
    "5. If dlib is unavailable (dlib-bin not installed): blink challenge is auto-passed\n"
    "   and passive 6-signal check continues as the only liveness gate.")

pdf.why_box("Why Two Layers? Passive + Active?",
    "Passive liveness (6-signal) runs silently on any image - photo uploads, video frames, UAV footage.\n"
    "It does NOT require cooperation from the subject. Perfect for drone surveillance.\n\n"
    "Active blink challenge is for the webcam login mode only, where a known person "
    "is deliberately trying to authenticate. It defeats even high-quality phone screen replays "
    "and photos that might fool passive texture analysis.\n\n"
    "Together: the system gets the best of both worlds. The passive check always runs "
    "(no user action needed). The active check adds a hard cryptographic-like gate "
    "for live authentication sessions.")

pdf.sub("Embedding Cache [NEW] - Faster Retraining")
pdf.body(
    "A related speed improvement: the MD5 embedding cache.\n"
    "Every time a face image is processed for retraining, its ArcFace embedding "
    "is saved to a cache file keyed by the MD5 hash of the image. "
    "On the next retrain, if the image has not changed, the saved embedding is reused "
    "instead of re-running ArcFace (which takes ~0.1s per image).\n\n"
    "Result: First retrain takes ~15 seconds. All subsequent retrains take ~2 seconds "
    "(a 10x speedup). This makes the auto-retrain after enrollment feel instant.")

# =========================================================================
# PAGE 9: PCA + SVM
# =========================================================================
pdf.add_page()

pdf.section("STEP 4:", "Feature Fusion + PCA - Combining All 4 Descriptions", ORANGE)
pdf.body(
    "After computing HOG, LBP, Geometry, and ArcFace features separately, "
    "we combine them all into a single long vector of numbers. "
    "This combined vector can be very long (thousands of numbers). "
    "PCA (Principal Component Analysis) then shrinks it to a smaller size "
    "while keeping 99% of the important information (raised from 95% for +2-5% accuracy).")

pdf.analogy("PCA is Like Summarising a Long Report into Key Bullet Points",
    "If you have a 50-page report about a topic, a skilled editor can "
    "summarise the most important information in 5 bullet points without "
    "losing the key findings. PCA does this mathematically for number vectors.\n\n"
    "It finds which combinations of numbers carry the most 'signal' (information about identity) "
    "and which carry mostly 'noise' (random variation due to lighting, blur, etc.). "
    "It keeps only the informative combinations and discards the noisy ones.")

pdf.why_box("Why PCA is Important for UAV Images",
    "UAV images have a lot of noise: motion blur, compression artefacts, lighting variation. "
    "Without PCA, this noise would be included in the feature vector and confuse the classifier. "
    "PCA helps filter out noise by focusing only on the directions of maximum variance "
    "in the feature space, which correspond to real identity differences rather than "
    "image quality variations.")

pdf.section("STEP 4B:", "SVM Classifier - Closed-Set Classification", ORANGE)
pdf.body(
    "SVM (Support Vector Machine) is a mathematical classifier that learns "
    "a decision boundary between different people's feature vectors. "
    "After training, given a new face's feature vector, it predicts "
    "which enrolled person it belongs to.")

pdf.analogy("SVM is Like Drawing Boundary Lines on a Map",
    "Imagine you have plotted all the feature vectors of Siddhant, Aditi, and Stuti "
    "on a map where each person's vectors form a cluster. "
    "SVM draws the optimal boundary lines between these clusters "
    "so that new face vectors falling in Siddhant's territory get classified as Siddhant, "
    "those in Aditi's territory get classified as Aditi, and so on.\n\n"
    "However, SVM always assigns a territory - it cannot say 'this point is outside "
    "all territories'. That is why we also use FAISS for open-set rejection.")

pdf.upgrade_box(
    "SVM alone as primary identifier (cannot say UNKNOWN for strangers)",
    "SVM for closed-set + FAISS for open-set identification",
    "SVM will always pick the closest enrolled person even for a complete stranger. "
    "In surveillance, most faces will be unknown people. "
    "We now use FAISS cosine similarity as the primary check: "
    "only if the similarity exceeds 0.35 do we accept the identity. "
    "Otherwise we say UNKNOWN. SVM is kept as a secondary signal.")

# =========================================================================
# PAGE 10: FAISS MATCHING
# =========================================================================
pdf.add_page()

pdf.section("STEP 5:", "FAISS Matching - Open-Set Identification", PURPLE)
pdf.body(
    "FAISS (Facebook AI Similarity Search) is the core identification engine. "
    "It stores the 512-number ArcFace embedding of every enrolled person "
    "and searches for the closest match when a new face is presented.")

pdf.analogy("FAISS is Like a Very Fast, Very Smart Filing System",
    "Imagine you have a filing cabinet with folders for every enrolled person. "
    "Each folder contains a numerical 'signature' of that person's face. "
    "When a new face arrives, FAISS instantly compares its signature against "
    "every folder in the cabinet and tells you which folder is most similar.\n\n"
    "The crucial difference from SVM is the THRESHOLD:\n"
    "  - If the best match similarity is >= 0.35: 'This is Siddhant'\n"
    "  - If the best match similarity is < 0.35: 'UNKNOWN - no match found'\n\n"
    "This threshold is what allows the system to correctly reject strangers "
    "instead of wrongly identifying them as someone enrolled.")

pdf.sub("How Cosine Similarity Works")
pdf.body(
    "The 512-number ArcFace embedding is normalised to have unit length "
    "(it sits on the surface of a 512-dimensional sphere). "
    "Cosine similarity measures the angle between two such vectors:\n\n"
    "  Similarity = 1.0 means the vectors point in exactly the same direction = same person\n"
    "  Similarity = 0.0 means the vectors are perpendicular = unrelated persons\n"
    "  Similarity = -1.0 means opposite directions = maximally different\n\n"
    "In practice, genuine pairs (same person) score 0.55-0.95. "
    "Impostor pairs (different person) score 0.05-0.35.")

pdf.example("FAISS in Action with Siddhant's Photo",
    "Query: Upload photo of Siddhant\n"
    "FAISS searches gallery of [aditi, siddhant, stuti]\n"
    "Results:\n"
    "  Rank 1: siddhant - similarity 0.691  [>= 0.35 threshold: ACCEPTED]\n"
    "  Rank 2: stuti    - similarity 0.312\n"
    "  Rank 3: aditi    - similarity 0.287\n"
    "Decision: IDENTIFIED as siddhant with 69.1% confidence\n\n"
    "If Rank 1 similarity was 0.28 (below 0.35):\n"
    "Decision: UNKNOWN (no enrolled person matches closely enough)")

pdf.why_box("Why FAISS_THRESHOLD = 0.35 (Not Higher Like 0.60)",
    "In ideal conditions (clear, close-up, frontal face photos), a threshold of 0.60 works well. "
    "But for UAV footage at 20-30 metres altitude:\n"
    "  - The face is 17-30 pixels wide (very low resolution)\n"
    "  - H.264 video compression adds block artefacts\n"
    "  - Motion blur reduces sharpness\n"
    "  - The camera angle is oblique (not frontal)\n\n"
    "All of these factors reduce the cosine similarity even for the correct person. "
    "We measured that genuine UAV pairs score 0.35-0.55 instead of 0.65-0.90. "
    "So we lowered the threshold to 0.35 to accept these lower but still genuine scores "
    "while still rejecting most impostors who score below 0.30.")

# =========================================================================
# PAGE 11: TEMPORAL VOTER + RANKED CANDIDATES
# =========================================================================
pdf.add_page()

pdf.section("STEP 6:", "Temporal Voting - Stable Video Identification", TEAL)
pdf.body(
    "When processing a video, we get one identification result per frame. "
    "Individual frames can be blurry, partially occluded, or compressed badly "
    "causing incorrect single-frame predictions. "
    "Temporal voting solves this by waiting for consistent agreement across frames.")

pdf.analogy("Temporal Voting is Like a Jury in a Court Case",
    "A single juror can make a mistake. But if 12 jurors all independently "
    "reach the same conclusion, you can be much more confident they are right.\n\n"
    "Our temporal voter uses two modes:\n\n"
    "  VIDEO UPLOAD (5/3 window): Identity confirmed when 3 out of 5 consecutive frames agree.\n"
    "  LIVE WEBCAM (10/6 window): Identity confirmed when 6 out of 10 consecutive frames agree.\n\n"
    "FAST CONFIRM: If any single frame has a similarity score >= 0.60 (very confident match), "
    "we skip the voter and confirm immediately. This removes the UNKNOWN-at-start delay "
    "for clear, frontal, well-lit faces.\n\n"
    "  Frame 1: Siddhant (blurry - score 0.38)\n"
    "  Frame 2: UNKNOWN  (very blurry - score 0.28)\n"
    "  Frame 3: Siddhant (CLEAR - score 0.71) -> FAST CONFIRM: SIDDHANT immediately!\n\n"
    "Or without fast-confirm:\n"
    "  Frames 1-3: Siddhant (scores 0.38, 0.41, 0.45)\n"
    "  Frame 4: Siddhant (score 0.52) -> 3/4 votes = confirmed: SIDDHANT")

pdf.upgrade_box(
    "PENDING label shown when temporal voter had not decided yet",
    "UNKNOWN shown immediately, confirmed identity shown once threshold reached",
    "The word PENDING was confusing to users who did not understand it was an internal state. "
    "Now faces show as UNKNOWN until the temporal voter is confident enough to confirm. "
    "This matches user expectations: either identified or unknown, nothing in between.")

pdf.section("", "Ranked Candidates - The Identity Shortlist", TEAL)
pdf.body(
    "When the system identifies a face, it not only shows the top match "
    "but also shows a ranked list of the next best matches with their similarity scores. "
    "This helps the user see how confident the system is and who the other candidates are.")

pdf.upgrade_box(
    "Ranked candidates came from SVM (different source than the main identity label)",
    "Ranked candidates now come from FAISS (same source as main identity label)",
    "Previously the main identity label (shown in the header) came from FAISS, "
    "but the ranked list below it came from SVM. These two systems often disagreed! "
    "The header might say 'Siddhant' but rank #1 would say 'Aditi'. "
    "Now both use FAISS as the single source of truth, so they always agree.")

# =========================================================================
# PAGE 12: WHY MULTI-MODAL (ALL FEATURES TOGETHER)
# =========================================================================
pdf.add_page()

pdf.section("", "Why Use 4 Features Together? The Multi-Modal Advantage", (100,50,150))
pdf.body(
    "Each of the 4 features (HOG, LBP, Geometry, ArcFace) is good at different things "
    "and struggles with different problems. Using all 4 together means that when one "
    "feature fails, the others compensate.")

pdf.table(
    headers=["Feature", "Best At", "Struggles With", "UAV Benefit"],
    rows=[
        ["HOG", "Face shape, edges", "Heavy blur, tiny faces",
         "Captures shape even at moderate blur"],
        ["LBP", "Texture, lighting changes", "Very small faces (<20px)",
         "100% illumination invariant for outdoor use"],
        ["Geometry", "Face structure, proportions", "Frontal faces only work well",
         "Proportions survive compression and blur"],
        ["ArcFace", "Deep identity features, best accuracy", "Compressed/tiny faces lower scores",
         "Most discriminative, backbone of system"],
    ],
    widths=[22, 46, 52, 54]
)

pdf.example("Multi-Modal Compensation in Practice",
    "Scenario: Siddhant's face is detected at 25 metres altitude in compressed H.264 video.\n\n"
    "ArcFace score: 0.41 (low due to compression and altitude, but above 0.35 threshold)\n"
    "HOG contribution: helps PCA separate Siddhant from Aditi based on brow/nose shape\n"
    "LBP contribution: robust skin texture pattern remains even in compressed image\n"
    "Geometry contribution: eye separation and nose ratios still correct\n\n"
    "Combined result: system correctly identifies Siddhant with high confidence\n"
    "Using ArcFace alone: borderline - might fail at 0.41 if threshold is not tuned\n"
    "Using HOG/LBP/Geometry alone: would likely fail at this resolution\n"
    "Combined: reliably passes because multiple independent signals agree")

pdf.section("", "Complete Data Flow - Number by Number", NAVY)
pdf.table(
    headers=["Stage", "Input", "Output", "Size"],
    rows=[
        ["SCRFD Detection", "1920x1080 video frame", "Face bounding box + 5 landmarks", "-"],
        ["Alignment", "Detected face region", "Aligned face crop", "112x112 px"],
        ["HOG Extraction", "112x112 face", "HOG feature vector", "~2,916 numbers"],
        ["LBP Extraction", "112x112 face", "LBP histogram vector", "~1,568 numbers"],
        ["Geometry Extraction", "68 landmarks", "Distance/angle vector", "~100 numbers"],
        ["ArcFace Extraction", "112x112 face", "Deep embedding", "512 numbers"],
        ["L2 Fusion", "4 separate vectors", "One combined vector", "~5,096 numbers"],
        ["PCA Reduction", "~5,096 numbers", "Compressed vector", "~150-300 numbers"],
        ["FAISS Search", "512-D ArcFace vector", "Top-k matches + similarity scores", "-"],
        ["Decision", "Best similarity + threshold", "Identity name or UNKNOWN", "-"],
    ],
    widths=[42, 52, 52, 28]
)

# =========================================================================
# PAGE 13: SUMMARY TABLE
# =========================================================================
pdf.add_page()

pdf.fc(NAVY); pdf.rect(pdf.l_margin, pdf.get_y(), 174, 10, "F")
pdf.set_font("Helvetica","B",12); pdf.tc(WHITE); pdf.set_x(pdf.l_margin+3)
pdf.cell(0, 10, "  COMPLETE SYSTEM SUMMARY - ONE PAGE REFERENCE")
pdf.tc(DARK); pdf.ln(14)

pdf.sub("The 7 Steps in Simple Words:")
steps_summary = [
    ("1. DETECT",    "SCRFD scans every frame and draws a box around each face. Tiny faces are upscaled."),
    ("2. ALIGN",     "Each face box is rotated, scaled, and cropped to a standard 112x112 pixel image."),
    ("3. ANTI-SPOOF","6-signal passive liveness rejects printed photos and screens in ~3ms. Webcam: blink challenge."),
    ("4. DESCRIBE",  "4 algorithms describe the face: HOG (shape), LBP (texture), Geometry (proportions), ArcFace+TTA."),
    ("5. COMBINE",   "All 4 descriptions are merged into one number vector, then PCA (99%) compresses it."),
    ("6. MATCH",     "FAISS compares the 512-D ArcFace vector against all enrolled people and finds the best match."),
    ("7. DECIDE",    "If best match similarity >= 0.35 and 3/5 frames agree: announce the identity. Else: UNKNOWN."),
]
for step, desc in steps_summary:
    pdf.fc(LGREY); pdf.rect(pdf.l_margin, pdf.get_y(), 174, 10, "F")
    pdf.set_font("Helvetica","B",9); pdf.tc(NAVY)
    pdf.set_x(pdf.l_margin+2); pdf.cell(32, 10, clean(step))
    pdf.set_font("Helvetica","",9); pdf.tc(DARK)
    pdf.multi_cell(138, 10, clean(desc))
    self_y = pdf.get_y()
    pdf.ln(1)

pdf.ln(4)
pdf.divider()

pdf.sub("Why This Combination of Tools?")
pdf.table(
    headers=["Component", "Replaced What", "Why Better"],
    rows=[
        ["SCRFD Detector", "MTCNN Detector",
         "3x faster, detects small/tilted faces at UAV altitude"],
        ["Passive Liveness (6-signal) [UPDATED]", "3-signal passive check",
         "Added colour, specular, chroma signals; threshold lowered to 0.45"],
        ["Active Blink Challenge [NEW]", "No active check",
         "EAR blink detection for webcam mode defeats screen replay attacks"],
        ["FAISS + Threshold", "SVM alone",
         "Enables UNKNOWN rejection - SVM always guesses someone"],
        ["ArcFace + TTA [NEW]", "ArcFace single embedding",
         "+3-8% accuracy: average of 5 embedding variants reduces compression noise"],
        ["PCA 99% [NEW]", "PCA 95%",
         "+2-5% accuracy: more variance retained means better identity separation"],
        ["MD5 Embedding Cache [NEW]", "Re-extract every retrain",
         "Retrains drop from 15s to ~2s after first run (10x speedup)"],
        ["Temporal Voter 5/3 + fast-confirm [NEW]", "Temporal voter 10/5",
         "Faster confirmation + 0.60 fast-confirm removes UNKNOWN-at-start delay"],
        ["Full-res SCRFD input", "Downscaled input",
         "Keeps 17-30px UAV faces detectable on 640x640 grid"],
        ["search_ranked() for candidates", "SVM for candidates",
         "Header and ranked list now always agree (same source)"],
    ],
    widths=[46, 46, 82]
)

pdf.divider()
pdf.sub("Key Numbers to Remember:")
pdf.table(
    headers=["Value", "What It Is", "Why This Number"],
    rows=[
        ["512",   "ArcFace embedding size",             "Industry standard ResNet-50 output dimension"],
        ["0.35",  "FAISS_THRESHOLD (UAV tuned)",        "Lowered from 0.45 - UAV faces score lower due to compression/altitude"],
        ["112x112", "Standard aligned face size",       "ArcFace training input size - every face resized to this"],
        ["3/5",   "Temporal voter threshold (video)",   "3 frames out of 5-frame window; fast-confirm at score >= 0.60"],
        ["6/10",  "Temporal voter threshold (webcam)",  "6 frames out of 10-frame window for live stream"],
        ["99%",   "PCA variance retained [NEW]",        "Raised from 95% for +2-5% accuracy improvement"],
        ["0.45",  "Liveness threshold [UPDATED]",       "6-signal passive: lowered from 0.50; blink adds hard active gate"],
        ["0.25",  "Blink EAR threshold [NEW]",          "Eye Aspect Ratio below 0.25 for 2+ frames = detected blink"],
        ["7.0s",  "Blink challenge timeout [NEW]",      "User has 7 seconds to blink; failure = session rejected"],
        ["5",     "TTA variants [NEW]",                 "5-variant embedding average gives +3-8% accuracy on UAV images"],
        ["48px",  "Tiny face upscale threshold",        "Faces below 48px get upscaled before ArcFace for better embedding"],
        ["~2s",   "Retrain time with cache [NEW]",      "MD5 hash cache: was 15s, now ~2s after first run (10x faster)"],
    ],
    widths=[22, 70, 82]
)

# SAVE
output = r"c:\Users\lucky\OneDrive\Desktop\Face Recognition\Face_Recognition_Pipeline_Explanation.pdf"
pdf.output(output)
print(f"\nPDF generated: {output}")
print(f"Total pages: {pdf.page}")
