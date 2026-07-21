"""
generate_army_guide_pdf.py - Army Offline Deployment Guide
"""

from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os

NAVY=(15,30,100);OLIVE=(60,80,30);OLIVE_BG=(230,235,200);BLUE=(28,78,183)
LIGHT_BG=(235,240,255);GREEN=(22,140,70);GREEN_BG=(220,245,230)
ORANGE=(190,90,0);ORANGE_BG=(255,240,215);RED=(185,30,30);RED_BG=(255,225,225)
PURPLE=(100,40,180);PURPLE_BG=(240,230,255);TEAL=(0,130,120);TEAL_BG=(220,245,242)
WHITE=(255,255,255);DARK=(20,20,40);GREY=(110,110,130);LGREY=(245,246,250)

def clean(s): return s.encode('latin-1',errors='replace').decode('latin-1')

class PDF(FPDF):
    def __init__(self):
        super().__init__('P','mm','A4')
        self.set_auto_page_break(auto=True,margin=22)
        self.set_margins(18,16,18)
        self._cover=False
    def fc(self,rgb): self.set_fill_color(*rgb)
    def tc(self,rgb): self.set_text_color(*rgb)
    def dc(self,rgb): self.set_draw_color(*rgb)
    def header(self):
        if self._cover: return
        self.fc(OLIVE);self.rect(0,0,210,10,'F')
        self.set_font('Helvetica','B',7);self.tc(WHITE);self.set_y(2)
        self.cell(0,6,'FACE RECOGNITION UAV  -  ARMY OFFLINE DEPLOYMENT GUIDE  -  RESTRICTED',align='C')
        self.tc(DARK);self.ln(9)
    def footer(self):
        if self._cover: return
        self.set_y(-13);self.fc(LGREY);self.rect(0,self.get_y()-1,210,16,'F')
        self.set_font('Helvetica','',7);self.tc(GREY)
        self.cell(0,9,f'Page {self.page-1}  |  Face Recognition UAV  |  Offline / Air-Gapped Deployment',align='C')
        self.tc(DARK)
    def cover(self):
        self._cover=True;self.add_page()
        self.fc(OLIVE);self.rect(0,0,210,297,'F')
        self.fc((50,70,20));self.rect(0,0,210,6,'F')
        self.set_font('Helvetica','B',9);self.tc((180,210,120));self.set_y(18)
        self.cell(0,7,'[ FACE RECOGNITION ON SKEWED UAV IMAGES ]',align='C',new_x=XPos.LMARGIN,new_y=YPos.NEXT)
        self.ln(10);self.set_font('Helvetica','B',28);self.tc(WHITE)
        self.cell(0,13,'ARMY OFFLINE',align='C',new_x=XPos.LMARGIN,new_y=YPos.NEXT)
        self.cell(0,13,'DEPLOYMENT GUIDE',align='C',new_x=XPos.LMARGIN,new_y=YPos.NEXT)
        self.set_font('Helvetica','B',13);self.tc((180,210,120))
        self.cell(0,9,'Air-Gapped / Classified Environment Setup',align='C',new_x=XPos.LMARGIN,new_y=YPos.NEXT)
        self.ln(6);self.set_font('Helvetica','',10);self.tc((200,220,160));self.set_x(25)
        self.multi_cell(160,6,'Complete guide for deploying on a laptop with NO internet.\nIncludes anti-spoofing, FAISS matching, TTA, and temporal voting.\nAll files transferred via USB from a previously configured laptop.',align='C')
        self.ln(14)
        badges=[('OFFLINE','100% Air-Gap'),('LIVENESS','Anti-Spoof'),('FAISS','Open-Set'),('TEMPORAL','Voting')]
        bx,bw,by=18,42,self.get_y()
        for lbl,val in badges:
            self.fc((50,80,20));self.dc((90,140,40));self.rect(bx,by,bw,22,'FD')
            self.set_font('Helvetica','B',6.5);self.tc((180,210,120))
            self.set_xy(bx,by+4);self.cell(bw,5,lbl,align='C')
            self.set_font('Helvetica','B',8);self.tc(WHITE)
            self.set_xy(bx,by+11);self.cell(bw,7,val,align='C')
            bx+=bw+9
        self.fc((30,50,10));self.rect(0,260,210,37,'F')
        self.set_font('Helvetica','B',10);self.tc((180,210,120));self.set_xy(0,266)
        self.cell(0,7,'github.com/TENSEI3011/face-recognition-skewed-images',align='C')
        self.set_font('Helvetica','',8);self.tc((140,170,100));self.set_xy(0,277)
        self.cell(0,6,'DO NOT CONNECT TO THE INTERNET ON THE DEPLOYMENT LAPTOP.',align='C')
        self._cover=False
    def step(self,num,title):
        self.ln(4)
        if self.get_y()>245: self.add_page()
        self.fc(OLIVE);self.rect(self.l_margin-2,self.get_y(),176,9,'F')
        self.set_font('Helvetica','B',11);self.tc(WHITE);self.set_x(self.l_margin)
        self.cell(0,9,clean(f'  STEP {num}  -  {title.upper()}'));self.tc(DARK);self.ln(12)
    def sub(self,t):
        self.set_font('Helvetica','B',10);self.tc(BLUE)
        self.cell(0,7,clean(t),new_x=XPos.LMARGIN,new_y=YPos.NEXT);self.tc(DARK)
    def body(self,t):
        self.set_font('Helvetica','',9.5);self.tc(DARK)
        self.multi_cell(0,5.5,clean(t));self.ln(1)
    def bullet(self,t,indent=4):
        self.set_font('Helvetica','',9.5);self.tc(DARK)
        x=self.l_margin+indent;self.set_x(x);self.cell(4,5.5,'-')
        self.set_x(x+5);self.multi_cell(0,5.5,clean(t))
    def code(self,lines,label=None):
        if label:
            self.set_font('Helvetica','B',7.5);self.tc(GREY)
            self.cell(0,5,clean(f'  {label}'),new_x=XPos.LMARGIN,new_y=YPos.NEXT)
        self.fc((22,27,58));bg_y=self.get_y()
        h=len(lines)*5.5+8
        if bg_y+h>272: self.add_page();bg_y=self.get_y()
        self.rect(self.l_margin,bg_y,174,h,'F')
        self.set_font('Courier','',8.5);self.tc((130,220,160))
        for line in lines:
            self.set_x(self.l_margin+4)
            self.cell(0,5.5,clean(line),new_x=XPos.LMARGIN,new_y=YPos.NEXT)
        self.tc(DARK);self.ln(4)
    def box(self,kind,title,text):
        cfg={'tip':(GREEN,GREEN_BG,'[TIP]'),'warn':(ORANGE,ORANGE_BG,'[WARNING]'),
             'error':(RED,RED_BG,'[CRITICAL ERROR]'),'info':(BLUE,LIGHT_BG,'[NOTE]'),
             'new':(PURPLE,PURPLE_BG,'[NEW FEATURE]')}
        col,bg,tag=cfg.get(kind,cfg['info'])
        text_lines=clean(text).split('\n');h=max(18,len(text_lines)*5.2+18)
        y0=self.get_y()
        if y0+h>272: self.add_page();y0=self.get_y()
        self.fc(bg);self.rect(self.l_margin,y0,174,h,'F')
        self.dc(col);self.set_line_width(0.9)
        self.line(self.l_margin,y0,self.l_margin,y0+h);self.set_line_width(0.2)
        self.set_xy(self.l_margin+5,y0+3)
        self.set_font('Helvetica','B',8.5);self.tc(col)
        self.cell(0,5,clean(f'{tag}  {title}'),new_x=XPos.LMARGIN,new_y=YPos.NEXT)
        self.set_x(self.l_margin+5);self.set_font('Helvetica','',9);self.tc(DARK)
        self.multi_cell(165,5.2,clean(text));self.ln(4)
    def table(self,headers,rows,widths):
        self.fc(OLIVE);self.tc(WHITE);self.set_font('Helvetica','B',8)
        for h,w in zip(headers,widths): self.cell(w,7,f' {clean(h)}',border=0,fill=True)
        self.ln();self.set_font('Helvetica','',8)
        for i,row in enumerate(rows):
            self.fc(LGREY if i%2==0 else WHITE);self.tc(DARK)
            for c,w in zip(row,widths): self.cell(w,6.5,f' {clean(c)}',border=0,fill=True)
            self.ln()
        self.ln(4)
    def banner(self,title):
        self.fc(LIGHT_BG);self.rect(self.l_margin,self.get_y(),174,9,'F')
        self.set_font('Helvetica','B',12);self.tc(NAVY);self.set_x(self.l_margin+3)
        self.cell(0,9,clean(f'  {title}'));self.tc(DARK);self.ln(13)
    def divider(self,color=OLIVE,my=3):
        self.dc(color);self.set_line_width(0.3)
        self.line(self.l_margin,self.get_y(),210-self.r_margin,self.get_y());self.ln(my)

pdf=PDF()
pdf.cover()

# PAGE 2: OVERVIEW
pdf.add_page()
pdf.fc(LIGHT_BG);pdf.rect(pdf.l_margin,pdf.get_y(),174,30,'F')
pdf.set_xy(pdf.l_margin+4,pdf.get_y()+4)
pdf.set_font('Helvetica','B',11);pdf.tc(NAVY)
pdf.cell(0,6,'What This Guide Covers',new_x=XPos.LMARGIN,new_y=YPos.NEXT)
pdf.set_x(pdf.l_margin+4);pdf.set_font('Helvetica','',9.5);pdf.tc(DARK)
pdf.multi_cell(166,5.5,'Full setup for deploying the Face Recognition UAV system on a laptop with ZERO internet access. All files are transferred via USB from a pre-configured source laptop. Covers installation, model copying, configuration, and operation.')
pdf.ln(8)

pdf.box('new','New in Current System Version',
    'PASSIVE LIVENESS DETECTION: Rejects printed photos and screen replays before running recognition. ~3ms CPU, no model download needed.\n'
    'FAISS OPEN-SET MATCHING: Unknown persons now correctly returned as UNKNOWN (not wrongly matched to gallery).\n'
    'TEST-TIME AUGMENTATION (TTA): 5 embedding variants averaged for +3-8% UAV accuracy.\n'
    'EMBEDDING CACHE: Retrains take ~2s after first run (was 15s). 10x speedup.\n'
    'TEMPORAL VOTING: Video uses 5/3 window; webcam uses 10/6 window. Fast-confirm at 0.60.\n'
    'AUTO-RETRAIN: Gallery enrollment now triggers retraining automatically -- no manual step.')

pdf.sub('System Requirements (Deployment Laptop)')
pdf.table(['Requirement','Minimum','Notes'],[
    ['OS','Windows 10 / 11 (64-bit)','No admin rights needed after Python install'],
    ['Python','3.10 or 3.11 ONLY','3.12/3.13 FAIL - dlib-bin incompatible'],
    ['RAM','8 GB minimum','16 GB recommended for smooth operation'],
    ['Disk','5 GB free space','Models ~600 MB + project ~200 MB + gallery'],
    ['CPU','Intel i5 / Ryzen 5+','No GPU needed - all inference on CPU'],
    ['Camera','USB webcam (optional)','Only for live stream mode'],
    ['Internet','NOT REQUIRED after setup','All files copied via USB first'],
],[22,56,96])

# PAGE 3: USB PREP
pdf.add_page()
pdf.banner('PREPARATION: Files to Copy from Source Laptop to USB Drive')
pdf.body('Before going to the deployment site, prepare the USB on Lucky laptop. Copy ALL items below. Missing any one item will cause failure.')
pdf.table(['#','Source Path (Lucky Laptop)','USB Folder','Required?'],[
    ['1','face-recognition-skewed-images\\  (entire project)','USB root','YES'],
    ['2','C:\\Users\\lucky\\.insightface\\models\\buffalo_l\\','USB:\\insightface\\buffalo_l\\','YES - ArcFace/SCRFD'],
    ['3','models\\shape_predictor_68_face_landmarks.dat','USB:\\models\\','YES - dlib'],
    ['4','data\\gallery\\  (all identity subfolders)','USB:\\gallery\\','YES'],
    ['5','results\\baseline\\models\\  (SVM+FAISS+PCA .pkl/.bin/.json)','USB:\\models_trained\\','YES'],
    ['6','web\\frontend\\fonts\\  (all .ttf files)','In project folder already','YES'],
    ['7','.env file (config)','USB:\\env_template.txt','YES'],
],[8,78,60,28])
pdf.box('warn','buffalo_l Folder is Hidden',
    'On Windows, .insightface is a hidden folder.\n'
    'Open File Explorer > View tab > tick Hidden items.\n'
    'Navigate to: C:\\Users\\lucky\\.insightface\\models\\\n'
    'Copy the buffalo_l folder to USB.')
pdf.box('warn','Critical: SVM + FAISS Model Files Must Exist',
    'results\\baseline\\models\\ must contain:\n'
    '  svm_classifier.pkl    faiss_index.bin    faiss_labels.json\n'
    '  pca_pipeline.pkl      label_encoder.pkl\n'
    'If missing, run: python experiments\\run_baseline.py on Lucky laptop first.')
pdf.box('tip','Run offline_setup.py First (on Source Laptop)',
    'python offline_setup.py\n'
    'Downloads and caches: buffalo_l, dlib model, all 9 UI fonts.\n'
    'After this completes, the project folder is ready for USB copy -- no internet needed on deployment laptop.')

# PAGE 4: INSTALLATION
pdf.add_page()
pdf.step(1,'Install Python 3.11 (if not already installed)')
pdf.body('Copy the Python 3.11 installer from USB to desktop and run it.')
pdf.set_font('Helvetica','B',10);pdf.tc(RED)
pdf.cell(0,7,'   --> TICK: Add Python.exe to PATH  on first screen  <-- CRITICAL',new_x=XPos.LMARGIN,new_y=YPos.NEXT)
pdf.tc(DARK)
pdf.body('Click Install Now. Wait. Click Close. Verify:')
pdf.code(['python --version'],label='Should show: Python 3.11.x')
pdf.box('error','python is not recognized',
    'Python was not added to PATH.\n'
    'Uninstall (Settings > Apps), re-run installer, tick PATH checkbox on first screen.')

pdf.step(2,'Copy the Project Folder from USB')
pdf.code(['Copy from USB to: C:\\Users\\<name>\\Desktop\\face-recognition-skewed-images'],label='File Explorer')

pdf.step(3,'Copy Model Files from USB to Correct Locations')
pdf.sub('3a - buffalo_l InsightFace model:')
pdf.code(['Create: C:\\Users\\<name>\\.insightface\\models\\','Paste: buffalo_l folder there'],label='File Explorer')
pdf.sub('3b - dlib landmark model:')
pdf.code(['Copy shape_predictor_68_face_landmarks.dat -> models\\ in project folder'],label='File Explorer')
pdf.sub('3c - Gallery photos:')
pdf.code(['Copy gallery\\ -> data\\ in project folder  (result: data\\gallery\\<name>\\)'],label='File Explorer')
pdf.sub('3d - Trained model files:')
pdf.code(['Copy USB:\\models_trained\\ -> results\\baseline\\models\\ in project folder'],label='File Explorer')

pdf.step(4,'Create Virtual Environment and Install Packages')
pdf.code([
    'cd Desktop\\face-recognition-skewed-images',
    'python -m venv venv',
    'venv\\Scripts\\activate',
    'pip install --upgrade pip',
    'pip install -r requirements.txt',
],label='Command Prompt - takes 10-20 minutes from wheel cache on USB')
pdf.box('warn','No Internet - Install from USB Wheel Cache',
    'pip install --no-index --find-links=D:\\wheels -r requirements.txt\n'
    '(Replace D:\\wheels with the path to your offline wheel folder on USB)')
pdf.box('error','venv\\Scripts\\activate gives execution policy error',
    'Run once in PowerShell:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned\n'
    'Press Y. Or use Command Prompt (cmd.exe) instead - no policy restriction.')

# PAGE 5: CONFIG + START
pdf.add_page()
pdf.step(5,'Create the .env Configuration File')
pdf.code(['copy .env.example .env','notepad .env'],label='Command Prompt')
pdf.code([
    'JWT_SECRET_KEY=face-recognition-uav-secret-key-2025',
    '',
    '# Local MongoDB (offline - recommended)',
    'MONGO_URI=mongodb://localhost:27017',
    '',
    '# OR disk-only mode (no MongoDB required)',
    '#MONGO_URI=',
    '',
    'MONGO_DB_NAME=facerecog_db',
    'ENV=development',
    '',
    '# Anti-Spoofing / Liveness Detection',
    'LIVENESS_ENABLED=True',
    'LIVENESS_THRESHOLD=0.50',
    '',
    '# FAISS Recognition Threshold',
    'FAISS_THRESHOLD=0.35',
],label='.env file - save Ctrl+S, close Notepad')
pdf.box('tip','Local MongoDB for Full Features',
    'Install MongoDB Community (portable) from USB.\n'
    'Start it: mongod --dbpath C:\\data\\db  (create C:\\data\\db folder first)\n'
    'Set MONGO_URI=mongodb://localhost:27017 in .env\n'
    'Gives audit log, watchlist, full gallery features - all offline.')
pdf.box('info','No MongoDB Mode',
    'Leave MONGO_URI blank. Gallery stored as files in data\\gallery\\\n'
    'Anti-spoofing, FAISS, and temporal voting all work without MongoDB.\n'
    'Audit log and watchlist pages will be unavailable.')

pdf.step(6,'Start the System')
pdf.code([
    'python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload'
],label='Command Prompt (venv active) - keep window open')
pdf.body('Wait 20-30 seconds. Correct startup:')
pdf.code([
    'INFO:     Uvicorn running on http://0.0.0.0:8000',
    '[FaceDetector] SCRFD (buffalo_l) loaded successfully.',
    '[ArcFaceExtractor] Loaded model: buffalo_l',
    '[SVMClassifier] Loaded from results/baseline/models/svm_classifier.pkl',
    '[FAISSMatcher] Loaded: XX vectors, N identities, threshold=0.35',
    '[LivenessDetector] Passive liveness initialized (threshold=0.50)',
    'INFO:     Application startup complete.',
],label='Expected output - all lines should appear')
pdf.body('Open Chrome or Edge:')
pdf.code(['http://localhost:8000'],label='Browser address bar')
pdf.box('error','Pipeline not loaded',
    'Trained models missing from results\\baseline\\models\\\n'
    'Solution A: Copy from USB (Step 3d).\n'
    'Solution B: python experiments\\run_baseline.py  (needs gallery photos).')

# PAGE 6: OPERATION
pdf.add_page()
pdf.banner('OPERATIONAL GUIDE - Using the System in the Field')
pdf.table(['Page','URL','What It Does'],[
    ['Dashboard','http://localhost:8000/','System status and model health'],
    ['Identify','http://localhost:8000/identify','Upload photo - liveness + face ID'],
    ['Gallery','http://localhost:8000/gallery','Enroll persons (images or video)'],
    ['Demo','http://localhost:8000/demo','Video upload or live webcam stream'],
    ['Audit Log','http://localhost:8000/audit','All events incl. spoof detections'],
    ['Watchlist','http://localhost:8000/watchlist','Define alert identities'],
    ['Config','http://localhost:8000/config','Threshold, liveness, TTA settings'],
    ['API Docs','http://localhost:8000/docs','REST API for integration'],
],[24,68,82])

pdf.sub('Identifying a Person from a Photo:')
pdf.bullet('Go to http://localhost:8000/identify')
pdf.bullet('Choose a JPG or PNG file containing a face. Click Identify.')
pdf.bullet('Liveness check runs first (~3ms): SPOOF DETECTED rejects printed photo / screen')
pdf.bullet('LIVE: shows name, confidence %, liveness_score, is_spoof, ranked candidates')
pdf.ln(3)
pdf.sub('Enrolling a New Person:')
pdf.bullet('Go to http://localhost:8000/gallery')
pdf.bullet('Enter name and upload 10+ clear face photos or a 30-60s video (MP4/MOV/AVI)')
pdf.bullet('Blurry images (sharpness < 40) are automatically rejected at enrollment')
pdf.bullet('Retrain triggers AUTOMATICALLY after upload - no manual click needed')
pdf.bullet('First retrain: ~15s. Subsequent: ~2s (embedding cache active)')
pdf.ln(3)
pdf.sub('Processing a Drone Video File:')
pdf.bullet('Go to http://localhost:8000/demo > Upload Video tab')
pdf.bullet("Upload .mp4 drone footage. Set 'Process every N frames' slider (6=fast, 2=accurate)")
pdf.bullet('Download the output video with identity labels annotated on each detected face')
pdf.ln(3)
pdf.sub('Live Webcam / Camera Stream:')
pdf.bullet('Go to http://localhost:8000/demo > Live Stream tab. Allow camera access.')
pdf.bullet('Temporal voting: 10/6 window - identity confirmed after 6/10 frames agree')
pdf.bullet('Fast-confirm: score >= 0.60 announces identity immediately (no waiting)')
pdf.bullet('Spoof faces shown with red SPOOF label - ArcFace NOT run on them')
pdf.ln(3)
pdf.sub('Adjusting Anti-Spoofing Sensitivity (Config Page):')
pdf.bullet('Go to http://localhost:8000/config > Anti-Spoofing card')
pdf.bullet('Toggle: Enable or Disable liveness check entirely')
pdf.bullet('Sensitivity slider: 0.0=accept all, 1.0=maximum rejection. Default: 0.50')
pdf.bullet('Increase to 0.65+ only if false-reject rate on real faces is acceptable')

# PAGE 7: TROUBLESHOOTING
pdf.add_page()
pdf.banner('TROUBLESHOOTING - Common Field Issues and Fixes')
errors=[
    ('python is not recognized',
     'Python not in PATH.\nRe-install Python 3.11. Tick Add Python.exe to PATH on first screen.'),
    ('Website shows Pipeline not loaded',
     'Trained models missing.\nSolution A: Copy from USB (Step 3d).\nSolution B: python experiments\\run_baseline.py (needs gallery photos in data\\gallery\\).'),
    ('SPOOF DETECTED for a real live person',
     'Liveness threshold set too high for current lighting.\nGo to Config > Anti-Spoofing > lower sensitivity slider.\nOr set LIVENESS_ENABLED=False in .env to disable completely.'),
    ('UNKNOWN shown for an enrolled person',
     '1. FAISS index missing: run retrain from gallery.\n2. FAISS_THRESHOLD too high in .env (default 0.35 for UAV).\n3. Too few gallery images (<3 per person): enroll more.\n4. UAV altitude too high (>25m): face too compressed for reliable match.'),
    ('InsightFace downloading at startup',
     'buffalo_l folder not at correct location.\nPlace at: C:\\Users\\<name>\\.insightface\\models\\buffalo_l\\\n(See Step 3a for exact instructions)'),
    ('No faces detected in drone video',
     '1. Video resolution too low: SCRFD needs faces at least 15px wide.\n2. Reduce altitude or increase camera zoom.'),
    ('Gallery upload fails: no face detected',
     'Image too blurry or no detectable face.\nUse clear well-lit photos, face at least 80px wide.\nProfile/sideways shots rejected automatically.'),
    ('Port 8000 already in use',
     'Use port 8001:\n  python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8001 --reload\nOpen http://localhost:8001 in browser.'),
]
for title,text in errors:
    pdf.box('error',title,text)

# PAGE 8: CHEAT SHEET
pdf.add_page()
pdf.fc(OLIVE);pdf.rect(pdf.l_margin,pdf.get_y(),174,10,'F')
pdf.set_font('Helvetica','B',12);pdf.tc(WHITE);pdf.set_x(pdf.l_margin+3)
pdf.cell(0,10,'  QUICK START CHEAT SHEET  -  Print This Page and Keep It Handy')
pdf.tc(DARK);pdf.ln(14)

pdf.sub('Every Session - 3 Commands to Start:')
pdf.code([
    'cd Desktop\\face-recognition-skewed-images',
    'venv\\Scripts\\activate',
    'python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload',
],label='Then open Chrome: http://localhost:8000')
pdf.divider()

pdf.sub('USB Files Checklist - Tick Before Leaving Source Laptop:')
pdf.table(['#','File / Folder','Destination on Deployment Laptop','Done?'],[
    ['1','Entire face-recognition-skewed-images\\ folder','Desktop\\','[ ]'],
    ['2','C:\\Users\\lucky\\.insightface\\models\\buffalo_l\\','C:\\Users\\<name>\\.insightface\\models\\buffalo_l\\','[ ]'],
    ['3','models\\shape_predictor_68_face_landmarks.dat','models\\ in project folder','[ ]'],
    ['4','data\\gallery\\ (all identity subfolders)','data\\ in project folder','[ ]'],
    ['5','results\\baseline\\models\\ (SVM+FAISS+PCA files)','results\\baseline\\ in project folder','[ ]'],
    ['6','Python 3.11 installer .exe','Run on deployment laptop','[ ]'],
    ['7','.env template with correct settings','Rename to .env in project root','[ ]'],
],[8,72,72,22])
pdf.divider()

pdf.sub('Current System Capabilities:')
pdf.table(['Capability','Status','Notes'],[
    ['Face detection','ACTIVE','SCRFD buffalo_l, down to ~20px face width'],
    ['Face alignment','ACTIVE','ArcFace 112x112 standard template'],
    ['Passive liveness check','ACTIVE','LBP+FFT+Gradient, ~3ms, threshold=0.50'],
    ['FAISS open-set matching','ACTIVE','Returns UNKNOWN for non-gallery persons'],
    ['Test-Time Augmentation (TTA)','ACTIVE','5-variant average, +3-8% UAV accuracy'],
    ['Temporal voting (video)','ACTIVE','5/3 window + fast-confirm at 0.60'],
    ['Temporal voting (webcam)','ACTIVE','10/6 window + fast-confirm at 0.60'],
    ['Auto-retrain on enrollment','ACTIVE','No manual click - triggers automatically'],
    ['Embedding cache','ACTIVE','~2s retrain after first run (was 15s)'],
    ['MongoDB audit log','OPTIONAL','Needs local mongod running'],
    ['Watchlist alerts','OPTIONAL','Needs MongoDB'],
    ['Extreme pose (>60 deg yaw)','LIMITED','3DDFA-V2 not yet integrated'],
    ['3D mask spoofing','LIMITED','Passive liveness only - active needed for adversarial'],
],[50,22,102])

output=os.path.join(os.path.dirname(os.path.abspath(__file__)),'Army_Offline_Deployment_Guide.pdf')
pdf.output(output)
print(f'PDF generated successfully!')
print(f'Saved to: {output}')
print(f'Total pages: {pdf.page}')
