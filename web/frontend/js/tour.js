/**
 * tour.js — Welcome / How-to-use step-by-step guide
 * Shows automatically on first visit; can be re-opened via the floating "?" button.
 * Saves state to localStorage so it won't show again once dismissed.
 */

const TOUR_KEY = 'facerecog_tour_done';

const TOUR_STEPS = [
  {
    icon: '👋',
    title: 'Welcome to Face Recognition UAV System',
    desc: 'A hybrid drone-based face recognition system with <strong>passive liveness detection</strong> (anti-spoofing), powered by SCRFD detection, HOG + LBP + ArcFace feature fusion, and FAISS cosine matching. This short tour (6 steps) will show you how to use every feature.',
  },
  {
    icon: '👤',
    title: 'Step 1 — Enroll Faces in the Gallery',
    desc: 'Go to <strong>Gallery</strong> in the sidebar. Create a new identity by entering a person\'s name, then upload clear face images or a short video clip. Enrolled images train the FAISS index — no retraining wait needed for immediate recognition.',
  },
  {
    icon: '🔍',
    title: 'Step 2 — Identify a Face',
    desc: 'Go to <strong>Identify Face</strong> in the sidebar. Upload any UAV/drone image. The pipeline detects the face with SCRFD, checks <strong>🛡️ liveness</strong> (rejects printed photos &amp; screens), then extracts features and matches using FAISS cosine similarity. Returns the identity with confidence score.',
  },
  {
    icon: '📷',
    title: 'Step 3 — Run the Live Demo',
    desc: 'Go to <strong>Live Demo</strong> in the sidebar. Stream from your webcam or upload a drone video. Every face is checked for liveness before identification. Spoofed faces are flagged <strong>⚠️ SPOOF</strong> in real-time with a distinct colour box.',
  },
  {
    icon: '📊',
    title: 'Step 4 — View Experiments &amp; Analytics',
    desc: 'Use <strong>Experiments &amp; Results</strong> to run ablation studies (HOG, LBP, ArcFace combinations) and degradation sweeps (CLEAN → EXTREME altitude profiles). Check <strong>Analytics</strong> for performance charts, accuracy trends, and evaluation metrics like Rank-1, EER, and AUC.',
  },
  {
    icon: '🚨',
    title: 'Step 5 — Watchlist, Audit &amp; Configuration',
    desc: 'Add persons of interest to the <strong>Watchlist</strong> — you\'ll be alerted when they are detected. The <strong>Audit Log</strong> records every identification event including liveness score and spoof flags. Use <strong>Configuration</strong> to tune FAISS threshold, PCA variance, and the 🛡️ liveness sensitivity.',
  },
  {
    icon: '✅',
    title: 'You\'re all set!',
    desc: 'The system is ready. Click the <strong>purple ? button</strong> at the bottom-right anytime to reopen this guide. Visit <strong>API Docs</strong> in the sidebar for programmatic access. 🎉',
  },
];

let tourStep = 0;

/* ── DOM refs ─────────────────────────────────────────────── */
const overlay   = document.getElementById('tour-overlay');
const icon      = document.getElementById('tour-icon');
const title     = document.getElementById('tour-title');
const desc      = document.getElementById('tour-desc');
const counter   = document.getElementById('tour-step-counter');
const fill      = document.getElementById('tour-progress-fill');
const dotsEl    = document.getElementById('tour-dots');
const btnPrev   = document.getElementById('tour-btn-prev');
const btnNext   = document.getElementById('tour-btn-next');
const btnSkip   = document.getElementById('tour-btn-skip');
const helpBtn   = document.getElementById('tour-help-btn');

/* ── Build dot indicators ─────────────────────────────────── */
function buildDots() {
  dotsEl.innerHTML = '';
  TOUR_STEPS.forEach((_, i) => {
    const dot = document.createElement('span');
    dot.className = 'tour-dot';
    dot.addEventListener('click', () => goToStep(i));
    dotsEl.appendChild(dot);
  });
}

/* ── Render current step ──────────────────────────────────── */
function renderStep(idx) {
  const step  = TOUR_STEPS[idx];
  const total = TOUR_STEPS.length;
  const pct   = Math.round(((idx + 1) / total) * 100);

  // Animate content swap
  const animTargets = [icon, title, desc];
  animTargets.forEach(el => {
    el.classList.remove('tour-step-fade');
    void el.offsetWidth; // reflow trick
    el.classList.add('tour-step-fade');
  });

  icon.textContent   = step.icon;
  title.textContent  = step.title;
  desc.innerHTML     = step.desc;
  counter.textContent = `Step ${idx + 1} of ${total}`;
  fill.style.width   = pct + '%';

  // Dots
  dotsEl.querySelectorAll('.tour-dot').forEach((dot, i) => {
    dot.className = 'tour-dot';
    if (i < idx)       dot.classList.add('done');
    else if (i === idx) dot.classList.add('active');
  });

  // Buttons
  btnPrev.style.display = idx === 0 ? 'none' : 'inline-flex';

  const isLast = idx === total - 1;
  btnNext.textContent = isLast ? '🎉 Finish' : 'Next →';
  btnSkip.style.display = isLast ? 'none' : 'inline-block';
}

/* ── Navigation ───────────────────────────────────────────── */
function goToStep(idx) {
  tourStep = Math.max(0, Math.min(idx, TOUR_STEPS.length - 1));
  renderStep(tourStep);
}

function tourNext() {
  if (tourStep < TOUR_STEPS.length - 1) {
    goToStep(tourStep + 1);
  } else {
    tourClose();
  }
}

function tourPrev() {
  if (tourStep > 0) {
    goToStep(tourStep - 1);
  }
}

function tourSkip() {
  tourClose();
}

/* ── Open / Close ─────────────────────────────────────────── */
function tourOpen() {
  tourStep = 0;
  buildDots();
  renderStep(0);
  overlay.classList.remove('hidden');
  helpBtn.style.display = 'none';
}

function tourClose() {
  overlay.classList.add('hidden');
  helpBtn.style.display = 'flex';
  localStorage.setItem(TOUR_KEY, '1');
}

/* ── Close on backdrop click ──────────────────────────────── */
overlay.addEventListener('click', function(e) {
  if (e.target === overlay) tourClose();
});

/* ── Keyboard navigation ──────────────────────────────────── */
document.addEventListener('keydown', function(e) {
  if (overlay.classList.contains('hidden')) return;
  if (e.key === 'ArrowRight' || e.key === 'Enter') tourNext();
  if (e.key === 'ArrowLeft')  tourPrev();
  if (e.key === 'Escape')     tourSkip();
});

/* ── Expose to global scope (HTML onclick attrs) ──────────── */
window.tourNext  = tourNext;
window.tourPrev  = tourPrev;
window.tourSkip  = tourSkip;
window.tourOpen  = tourOpen;
window.tourClose = tourClose;

/* ── Auto-show on first visit ─────────────────────────────── */
(function init() {
  buildDots();
  renderStep(0);

  if (!localStorage.getItem(TOUR_KEY)) {
    // First time: show immediately
    overlay.classList.remove('hidden');
    helpBtn.style.display = 'none';
  } else {
    // Already seen: hide overlay, show help button
    overlay.classList.add('hidden');
    helpBtn.style.display = 'flex';
  }
})();
