// demo.js - Live webcam stream + video upload processing

var ws           = null;
var webcamStream = null;
var totalFrames  = 0;
var totalFaces   = 0;
var currentMode  = 'live';
var _wsBusy      = false;      // true while waiting for server response
var _inferCanvas = null;
var _sendTimer   = null;        // setInterval for sending frames to server
var _rafHandle   = null;        // requestAnimationFrame handle for smooth display
var _video       = null;
var _canvas      = null;
var _ctx         = null;
var _lastFaces   = [];          // last face detections from server (for overlay)
var _scaleX      = 1;           // scale from inference canvas → display canvas (x)
var _scaleY      = 1;           // scale from inference canvas → display canvas (y)

// ── Liveness / Blink Challenge State ────────────────────────────────────────
var _livenessSessionId    = null;   // current challenge session UUID
var _livenessTimer        = null;   // setInterval for sending blink frames
var _livenessPassed       = false;  // true once challenge passed
var _livenessTimeoutSec   = 7;      // synced from server
var _livenessStartedAt    = 0;


// Init - wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
  _video  = document.getElementById('webcam-video');
  _canvas = document.getElementById('stream-canvas');
  _ctx    = _canvas ? _canvas.getContext('2d') : null;

  var videoInput = document.getElementById('video-input');
  var videoZone  = document.getElementById('video-upload-zone');
  if (videoInput) {
    videoInput.addEventListener('change', function() {
      var f  = videoInput.files[0];
      var el = document.getElementById('video-file-name');
      if (el) el.textContent = f ? ('Selected: ' + f.name) : '';
    });
  }
  if (videoZone && videoInput) {
    videoZone.addEventListener('dragover', function(e) {
      e.preventDefault(); videoZone.classList.add('drag-over');
    });
    videoZone.addEventListener('dragleave', function() {
      videoZone.classList.remove('drag-over');
    });
    videoZone.addEventListener('drop', function(e) {
      e.preventDefault();
      videoZone.classList.remove('drag-over');
      try { videoInput.files = e.dataTransfer.files; } catch(x) {}
      var f  = videoInput.files[0];
      var el = document.getElementById('video-file-name');
      if (el) el.textContent = f ? ('Selected: ' + f.name) : '';
    });
  }
});

// Mode switching
function setMode(mode) {
  currentMode = mode;
  var live = document.getElementById('mode-live');
  var vid  = document.getElementById('mode-video');
  var lbtn = document.getElementById('tab-live-btn');
  var vbtn = document.getElementById('tab-video-btn');
  if (live)  live.classList.toggle('hidden',  mode !== 'live');
  if (vid)   vid.classList.toggle('hidden',   mode !== 'video');
  if (lbtn)  lbtn.classList.toggle('active',  mode === 'live');
  if (vbtn)  vbtn.classList.toggle('active',  mode === 'video');
  if (mode === 'video' && ws) stopStream();
}

// ---- LIVE WEBCAM ----

async function startStream() {
  clearAlert('alert-zone');
  var startBtn    = document.getElementById('start-btn');
  var placeholder = document.getElementById('cam-placeholder');

  // Step 1: check browser support
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showAlert('alert-zone',
      'Camera not supported. Open this page at http://localhost:8000/demo in Chrome or Edge.',
      'danger');
    return;
  }

  // Step 2: show progress
  if (startBtn)    { startBtn.disabled = true; startBtn.textContent = 'Starting camera...'; }
  if (placeholder) { placeholder.textContent = 'Requesting camera permission...'; }

  // Step 3: get camera
  try {
    try {
      webcamStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false
      });
    } catch(hdErr) {
      webcamStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    }
  } catch(err) {
    if (startBtn) { startBtn.disabled = false; startBtn.textContent = 'Start Recognition'; }
    if (placeholder) { placeholder.innerHTML = '<div style="font-size:2rem">&#128247;</div>Camera not started'; }
    var msg = err.message;
    if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError')
      msg = 'Permission denied. Click the lock icon in your browser address bar, allow camera access, then try again.';
    else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError')
      msg = 'No camera found. Connect a webcam and try again.';
    else if (err.name === 'NotReadableError' || err.name === 'TrackStartError')
      msg = 'Camera is being used by another app (Teams, Zoom, etc.). Close it and try again.';
    showAlert('alert-zone', 'Camera error: ' + msg, 'danger');
    return;
  }

  // Step 4: attach stream to video element
  if (!_video) _video = document.getElementById('webcam-video');
  _video.srcObject = webcamStream;
  _video.style.display = 'block';
  if (placeholder) { placeholder.style.display = 'none'; }

  // Step 5: wait for video to be ready (handle race condition)
  if (_video.readyState < 1) {
    var metaOk = await new Promise(function(resolve) {
      var timer = setTimeout(function() { resolve(false); }, 10000);
      _video.addEventListener('loadedmetadata', function() {
        clearTimeout(timer); resolve(true);
      }, { once: true });
      _video.addEventListener('error', function() {
        clearTimeout(timer); resolve(false);
      }, { once: true });
    });
    if (!metaOk) {
      _doStopStream();
      if (startBtn) { startBtn.disabled = false; startBtn.textContent = 'Start Recognition'; }
      showAlert('alert-zone', 'Camera timed out. Please try again.', 'danger');
      return;
    }
  }

  // Start playback
  try { await _video.play(); } catch(x) { /* may already be playing */ }

  // Step 6: size canvases
  var vw = _video.videoWidth  || 640;
  var vh = _video.videoHeight || 480;
  if (!_canvas) _canvas = document.getElementById('stream-canvas');
  if (_canvas)  { _canvas.width = vw; _canvas.height = vh; }
  if (!_ctx && _canvas) _ctx = _canvas.getContext('2d');

  // Inference canvas: 640×480 — wide enough so each face in a group of 5–6
  // people is at least 60px wide (well above ArcFace's reliable detection floor).
  // 320×240 was too small: faces became ~30px in group shots → bad embeddings.
  _inferCanvas        = document.createElement('canvas');
  _inferCanvas.width  = 640;
  _inferCanvas.height = 480;

  // Scale factors so face boxes (from server, in original-frame coords) map to display canvas
  _scaleX = vw / 640;
  _scaleY = vh / 480;
  _lastFaces = [];

  // Start the smooth 60fps display loop (draws webcam + overlays boxes)
  _startDisplayLoop();

  if (startBtn) { startBtn.textContent = 'Verifying liveness...'; }

  // Step 7: run blink challenge — WebSocket opens only after user blinks
  runBlinkChallenge();
}


// ── Liveness Challenge Helpers ───────────────────────────────────────────────

function _showLivenessOverlay(timeoutSec) {
  var overlay = document.getElementById('liveness-overlay');
  if (!overlay) return;
  overlay.style.display = 'flex';
  // Reset UI elements
  var timerEl = document.getElementById('liveness-timer');
  var earEl   = document.getElementById('liveness-ear');
  var msgEl   = document.getElementById('liveness-msg');
  var barEl   = document.getElementById('liveness-progress-bar');
  var eyeEl   = document.getElementById('liveness-eye-icon');
  if (timerEl) timerEl.textContent = timeoutSec || 7;
  if (earEl)   earEl.textContent   = '—';
  if (msgEl)   msgEl.innerHTML     = 'Please <strong style="color:#fff">BLINK</strong> naturally to verify you are real.';
  if (barEl)   barEl.style.width   = '100%';
  if (eyeEl)   eyeEl.textContent   = '👁';
  // Pulsing animation on the eye icon
  if (eyeEl) {
    eyeEl.style.animation = 'none';
    setTimeout(function() {
      eyeEl.style.animation = 'liveness-pulse 2s ease-in-out infinite';
    }, 50);
  }
}

function _hideLivenessOverlay() {
  var overlay = document.getElementById('liveness-overlay');
  if (overlay) overlay.style.display = 'none';
  if (_livenessTimer) { clearInterval(_livenessTimer); _livenessTimer = null; }
}

function skipLivenessChallenge() {
  // Allow tester to skip the challenge (demo mode only)
  _livenessPassed = true;
  _hideLivenessOverlay();
  _openWebSocket();
}

/**
 * runBlinkChallenge()
 * Detects a blink entirely in the browser using pixel-level eye-region analysis.
 * NO server round-trips, NO external libraries needed.
 *
 * HOW: Samples a horizontal strip across the centre of the face frame (eye-level).
 * When eyes are open → bright horizontal strip (iris/sclera reflects light).
 * When eyes blink → strip darkens as eyelids close.
 * A blink = the strip brightness drops >15% below baseline, then recovers.
 * A static phone photo → no brightness change → blink never detected.
 */
async function runBlinkChallenge() {
  _livenessPassed  = false;
  _livenessStartedAt = Date.now();
  _livenessTimeoutSec = 10;  // 10 seconds to blink

  _showLivenessOverlay(_livenessTimeoutSec);

  // Ensure inference canvas is available
  if (!_inferCanvas) {
    _inferCanvas       = document.createElement('canvas');
    _inferCanvas.width  = 320;
    _inferCanvas.height = 240;
  }

  // Build blink dot
  var dotsEl = document.getElementById('blink-dots');
  if (dotsEl) {
    dotsEl.innerHTML = '';
    var dot = document.createElement('div');
    dot.id = 'blink-dot-0';
    dot.style.cssText = 'width:14px;height:14px;border-radius:50%;background:rgba(255,255,255,.2);border:2px solid rgba(255,255,255,.4);transition:all .3s';
    dotsEl.appendChild(dot);
  }

  var challengeDone   = false;
  var blinkDetected   = false;
  var baselineBright  = -1;    // rolling baseline brightness of eye strip
  var eyeHistory      = [];    // brightness samples
  var HISTORY_LEN     = 8;     // rolling window
  var belowBaseline   = false; // currently in a dip?
  var dipFrames       = 0;     // frames below threshold in current dip
  var MIN_DIP_FRAMES  = 2;     // min frames dark to count as blink
  var DIP_THRESHOLD   = 0.84;  // brightness drops to <84% of baseline → blink

  // We sample a horizontal strip at 35-55% height (eye level) and 20-80% width
  var offscreenCtx = _inferCanvas.getContext('2d');

  function _sampleEyeStrip() {
    if (!_video || !_video.readyState || _video.videoWidth === 0) return null;
    // Draw current frame to inference canvas
    offscreenCtx.drawImage(_video, 0, 0, 320, 240);
    // Sample eye-level strip (y=84..132 = 35-55% of 240px)
    var y0 = 84, y1 = 132;
    var x0 = 64, x1 = 256;  // 20-80% of 320px
    var imageData = offscreenCtx.getImageData(x0, y0, x1 - x0, y1 - y0);
    var data = imageData.data;
    var total = 0;
    var count = 0;
    for (var i = 0; i < data.length; i += 4) {
      // Luminance: 0.299R + 0.587G + 0.114B
      total += 0.299 * data[i] + 0.587 * data[i+1] + 0.114 * data[i+2];
      count++;
    }
    return count > 0 ? total / count : null;
  }

  _livenessTimer = setInterval(function() {
    if (challengeDone) return;

    // Update countdown
    var elapsed   = (Date.now() - _livenessStartedAt) / 1000;
    var remaining = Math.max(0, _livenessTimeoutSec - elapsed);
    var barEl   = document.getElementById('liveness-progress-bar');
    var timerEl = document.getElementById('liveness-timer');
    if (barEl)   barEl.style.width = ((remaining / _livenessTimeoutSec) * 100) + '%';
    if (timerEl) timerEl.textContent = remaining.toFixed(0);

    // Sample current frame brightness
    var bright = _sampleEyeStrip();
    if (bright === null) return;

    // Build baseline from first HISTORY_LEN samples
    eyeHistory.push(bright);
    if (eyeHistory.length > HISTORY_LEN * 3) eyeHistory.shift();

    if (baselineBright < 0) {
      if (eyeHistory.length >= HISTORY_LEN) {
        // Use top 70% average as baseline (ignore any outlier dips)
        var sorted = eyeHistory.slice().sort(function(a,b){return b-a;});
        var top    = sorted.slice(0, Math.ceil(sorted.length * 0.7));
        baselineBright = top.reduce(function(s,v){return s+v;}, 0) / top.length;
      }
      return;  // still building baseline
    }

    // Continuously update baseline (slow exponential moving average, upward only)
    if (bright > baselineBright) {
      baselineBright = baselineBright * 0.95 + bright * 0.05;
    }

    var ratio = bright / baselineBright;

    // Update EAR display with ratio (repurposing EAR label to show brightness ratio)
    var earEl = document.getElementById('liveness-ear');
    if (earEl) {
      earEl.textContent = ratio.toFixed(3);
      earEl.style.color = ratio < DIP_THRESHOLD ? '#f87171' : '#4ade80';
    }

    // Blink detection state machine
    if (ratio < DIP_THRESHOLD) {
      dipFrames++;
      belowBaseline = true;
    } else {
      if (belowBaseline && dipFrames >= MIN_DIP_FRAMES) {
        // Eye just re-opened after being closed long enough → blink!
        blinkDetected = true;
        var dotEl = document.getElementById('blink-dot-0');
        if (dotEl) {
          dotEl.style.background = '#4ade80';
          dotEl.style.border     = '2px solid #4ade80';
          dotEl.style.boxShadow  = '0 0 8px #4ade80';
        }
        console.log('[Liveness] Blink detected! ratio=' + ratio.toFixed(3) +
                    ' dipFrames=' + dipFrames + ' baseline=' + baselineBright.toFixed(1));
      }
      belowBaseline = false;
      dipFrames     = 0;
    }

    // Pass condition
    if (blinkDetected) {
      challengeDone   = true;
      _livenessPassed = true;
      if (_livenessTimer) { clearInterval(_livenessTimer); _livenessTimer = null; }

      var msgEl = document.getElementById('liveness-msg');
      var eyeEl = document.getElementById('liveness-eye-icon');
      if (msgEl) msgEl.innerHTML = '<strong style="color:#4ade80">✅ Liveness verified!</strong>';
      if (eyeEl) { eyeEl.textContent = '✅'; eyeEl.style.animation = 'none'; }
      if (barEl) barEl.style.background = '#4ade80';

      setTimeout(function() {
        _hideLivenessOverlay();
        _openWebSocket();
      }, 900);
      return;
    }

    // Timeout
    if (elapsed >= _livenessTimeoutSec) {
      challengeDone   = true;
      _livenessPassed = false;
      if (_livenessTimer) { clearInterval(_livenessTimer); _livenessTimer = null; }

      var msgEl2 = document.getElementById('liveness-msg');
      var eyeEl2 = document.getElementById('liveness-eye-icon');
      if (msgEl2) msgEl2.innerHTML = '<strong style="color:#f87171">❌ No blink detected.</strong><br><span style="font-size:.85rem">Please blink naturally and try again.</span>';
      if (eyeEl2) { eyeEl2.textContent = '❌'; eyeEl2.style.animation = 'none'; }
      if (barEl)  barEl.style.background = '#ef4444';

      setTimeout(function() {
        _hideLivenessOverlay();
        _doStopStream();
        showAlert('alert-zone', '🔒 Liveness check failed — no blink detected. Hold your phone photo in front and it cannot blink! Try with YOUR face and blink once.', 'danger');
      }, 1500);
    }
  }, 80);  // ~12fps sampling
}

/**
 * _openWebSocket()
 * Opens the recognition WebSocket AFTER liveness is passed.
 * Extracted from startStream() so we can call it post-challenge.
 */
function _openWebSocket() {
  var startBtn = document.getElementById('start-btn');
  if (startBtn) { startBtn.textContent = 'Connecting...'; }

  var wsProto = (location.protocol === 'https:') ? 'wss' : 'ws';
  var wsUrl   = wsProto + '://' + location.host + '/ws/stream';

  try { ws = new WebSocket(wsUrl); }
  catch(e) {
    _doStopStream();
    if (startBtn) { startBtn.disabled = false; startBtn.textContent = 'Start Recognition'; }
    showAlert('alert-zone', 'WebSocket failed: ' + e.message, 'danger');
    return;
  }

  ws.binaryType = 'arraybuffer';

  ws.onopen = function() {
    if (startBtn) {
      startBtn.disabled = false;
      startBtn.textContent = 'Start Recognition';
      startBtn.classList.add('hidden');
    }
    var badge   = document.getElementById('live-badge');
    var stopBtn = document.getElementById('stop-btn');
    if (badge)   badge.classList.add('active');
    if (stopBtn) stopBtn.classList.remove('hidden');
    _wsBusy     = false;
    _lastSentMs = 0;

    var fpsEl = document.getElementById('fps-select');
    var intervalMs = fpsEl ? parseInt(fpsEl.value, 10) : 333;
    _sendTimer = setInterval(_sendFrame, intervalMs);

    if (fpsEl && !fpsEl._demoListenerAdded) {
      fpsEl._demoListenerAdded = true;
      fpsEl.addEventListener('change', function() {
        if (_sendTimer) { clearInterval(_sendTimer); _sendTimer = null; }
        if (ws && ws.readyState === WebSocket.OPEN) {
          _sendTimer = setInterval(_sendFrame, parseInt(this.value, 10));
        }
      });
    }
  };

  ws.onmessage = function(event) {
    _wsBusy = false;
    try {
      var msg = JSON.parse(event.data);
      if (msg.error) {
        showAlert('alert-zone', 'Server error: ' + msg.error, 'warning');
        stopStream();
        return;
      }
      if (msg.type === 'frame') {
        _lastFaces = msg.faces || [];
        _updateFaceList(_lastFaces);
        totalFrames++;
        var fc = document.getElementById('frame-count');
        if (fc) fc.textContent = totalFrames;
      }
    } catch(x) {}
  };

  ws.onerror = function() {
    showAlert('alert-zone', 'WebSocket error. Check the server is running.', 'danger');
    stopStream();
  };

  ws.onclose = function(ev) {
    if (ev.code !== 1000 && ev.code !== 1005) {
      showAlert('alert-zone', 'Stream closed (code ' + ev.code + '). ' + (ev.reason || ''), 'warning');
    }
    _doStopStream();
  };
}


function _sendFrame() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  if (!_inferCanvas || !_video || !_video.videoWidth) return;

  // Drop frame if previous one is still being processed by the server.
  // This is the key backpressure mechanism that prevents lag build-up.
  if (_wsBusy) return;

  _wsBusy = true;
  var ictx = _inferCanvas.getContext('2d');
  ictx.drawImage(_video, 0, 0, _inferCanvas.width, _inferCanvas.height);
  _inferCanvas.toBlob(function(blob) {
    if (blob && ws && ws.readyState === WebSocket.OPEN) {
      blob.arrayBuffer().then(function(buf) {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(buf);
        } else {
          _wsBusy = false;
        }
      }).catch(function() { _wsBusy = false; });
    } else {
      _wsBusy = false;
    }
  }, 'image/jpeg', 0.70);   // 0.70 quality at 640×480 — sharp enough for multi-person group detection
}

// ── Display loop (runs at 60fps via rAF, completely independent of inference) ──
// Draws the raw webcam feed every frame, then overlays the last known face boxes.
// This means the video is ALWAYS smooth, even when inference takes 500ms.
function _startDisplayLoop() {
  if (_rafHandle) cancelAnimationFrame(_rafHandle);
  function _drawFrame() {
    if (!_video || !_canvas || !_ctx) return;
    if (_video.readyState >= 2 && _video.videoWidth) {
      // Draw live webcam
      _ctx.drawImage(_video, 0, 0, _canvas.width, _canvas.height);
      // Overlay face bounding boxes from last server response
      _drawFaceOverlay(_lastFaces);
    }
    _rafHandle = requestAnimationFrame(_drawFrame);
  }
  _rafHandle = requestAnimationFrame(_drawFrame);
}

// Draw bounding boxes + labels + auto-zoom PiP on the display canvas.
// Boxes are in server coords (320×240 inference frame); _scaleX/Y maps them to canvas.
function _drawFaceOverlay(faces) {
  if (!faces || !faces.length || !_ctx || !_canvas) return;

  // Find the most-confident face for the zoom panel
  var bestFace = faces.reduce(function(a, b) {
    return ((b.confidence || 0) > (a.confidence || 0)) ? b : a;
  }, faces[0]);

  _ctx.save();

  // ── Draw bounding boxes and labels for ALL detected faces ──────────────────
  for (var i = 0; i < faces.length; i++) {
    var f     = faces[i];
    var box   = f.box || {};
    var bx    = Math.round((box.x || 0) * _scaleX);
    var by    = Math.round((box.y || 0) * _scaleY);
    var bw    = Math.round((box.w || 0) * _scaleX);
    var bh    = Math.round((box.h || 0) * _scaleY);
    var isSpoof = f.is_spoof;
    var known   = f.is_known;
    var label   = isSpoof ? '⚠️ SPOOF DETECTED'
                          : (f.identity || 'UNKNOWN').replace(/_/g, ' ');

    var pct   = ((f.confidence || 0) * 100).toFixed(1) + '%';
    var text  = isSpoof ? label : (label + '  ' + pct);
    var color = isSpoof ? '#f97316' : (known ? '#00e676' : '#ff1744');

    var cLen  = Math.min(bw, bh) * 0.22;   // corner bracket length

    // Corner-bracket style box (more modern than solid rectangle)
    _ctx.strokeStyle = color;
    _ctx.lineWidth   = 3;
    _ctx.lineCap     = 'round';
    // Top-left
    _ctx.beginPath(); _ctx.moveTo(bx, by + cLen); _ctx.lineTo(bx, by); _ctx.lineTo(bx + cLen, by); _ctx.stroke();
    // Top-right
    _ctx.beginPath(); _ctx.moveTo(bx + bw - cLen, by); _ctx.lineTo(bx + bw, by); _ctx.lineTo(bx + bw, by + cLen); _ctx.stroke();
    // Bottom-left
    _ctx.beginPath(); _ctx.moveTo(bx, by + bh - cLen); _ctx.lineTo(bx, by + bh); _ctx.lineTo(bx + cLen, by + bh); _ctx.stroke();
    // Bottom-right
    _ctx.beginPath(); _ctx.moveTo(bx + bw - cLen, by + bh); _ctx.lineTo(bx + bw, by + bh); _ctx.lineTo(bx + bw, by + bh - cLen); _ctx.stroke();

    // Subtle semi-transparent fill so face region is highlighted
    _ctx.fillStyle = known ? 'rgba(0,230,118,0.07)' : 'rgba(255,23,68,0.07)';
    _ctx.fillRect(bx, by, bw, bh);

    // Label pill
    _ctx.font = 'bold 12px Inter, sans-serif';
    var tw = _ctx.measureText(text).width;
    var pillX = bx, pillY = by - 26, pillW = tw + 14, pillH = 22;
    // Clamp pill above top edge
    if (pillY < 2) pillY = by + bh + 4;
    _ctx.fillStyle = color;
    _ctx.beginPath();
    _ctx.roundRect(pillX, pillY, pillW, pillH, 5);
    _ctx.fill();
    _ctx.fillStyle = '#000';
    _ctx.fillText(text, pillX + 7, pillY + 15);
  }

  // ── Auto-zoom PiP: crop & magnify the best face into the corner ───────────
  var pb   = bestFace.box || {};
  var pbx  = Math.round((pb.x || 0) * _scaleX);
  var pby  = Math.round((pb.y || 0) * _scaleY);
  var pbw  = Math.round((pb.w || 0) * _scaleX);
  var pbh  = Math.round((pb.h || 0) * _scaleY);

  if (pbw > 8 && pbh > 8 && _video && _video.readyState >= 2) {
    // Add 15% padding around the face for context
    var padX = Math.round(pbw * 0.15);
    var padY = Math.round(pbh * 0.15);
    var cropX = Math.max(0, pbx - padX);
    var cropY = Math.max(0, pby - padY);
    var cropW = Math.min(_canvas.width  - cropX, pbw + padX * 2);
    var cropH = Math.min(_canvas.height - cropY, pbh + padY * 2);

    // PiP dimensions — square panel, 140px
    var PIP = 140;
    var margin = 12;
    var pipX = _canvas.width  - PIP - margin;
    var pipY = _canvas.height - PIP - margin - 28;  // 28px for label bar below

    // Panel shadow
    _ctx.shadowColor   = 'rgba(0,0,0,0.6)';
    _ctx.shadowBlur    = 12;
    _ctx.shadowOffsetX = 2;
    _ctx.shadowOffsetY = 2;

    // Panel border
    var pipColor = bestFace.is_known ? '#00e676' : '#ff1744';
    _ctx.strokeStyle = pipColor;
    _ctx.lineWidth   = 2.5;
    _ctx.strokeRect(pipX - 1, pipY - 1, PIP + 2, PIP + 2);

    // Draw zoomed face from the raw video (not canvas — avoids drawing boxes inside zoom)
    _ctx.shadowBlur = 0;
    _ctx.drawImage(_video, cropX, cropY, cropW, cropH, pipX, pipY, PIP, PIP);

    // Scanline grid overlay (optional aesthetic)
    _ctx.fillStyle = 'rgba(0,0,0,0.04)';
    for (var sy = pipY; sy < pipY + PIP; sy += 3) {
      _ctx.fillRect(pipX, sy, PIP, 1);
    }

    // Label bar below PiP
    var pipLabel = (bestFace.identity || 'UNKNOWN').replace(/_/g, ' ');
    var pipPct   = ((bestFace.confidence || 0) * 100).toFixed(1) + '%';
    _ctx.fillStyle = pipColor;
    _ctx.fillRect(pipX - 1, pipY + PIP + 1, PIP + 2, 26);
    _ctx.fillStyle = '#000';
    _ctx.font = 'bold 11px Inter, sans-serif';
    _ctx.textAlign = 'center';
    _ctx.fillText(pipLabel + '  ' + pipPct, pipX + PIP / 2, pipY + PIP + 17);
    _ctx.textAlign = 'left';

    // Corner "ZOOM" tag
    _ctx.fillStyle = 'rgba(0,0,0,0.55)';
    _ctx.fillRect(pipX, pipY, 40, 16);
    _ctx.fillStyle = pipColor;
    _ctx.font = 'bold 9px Inter, sans-serif';
    _ctx.fillText('\u{1F50D} ZOOM', pipX + 3, pipY + 11);
  }

  _ctx.restore();
}

function _doStopStream() {
  if (_sendTimer)   { clearInterval(_sendTimer); _sendTimer = null; }
  if (_rafHandle)   { cancelAnimationFrame(_rafHandle); _rafHandle = null; }
  if (ws)           { try { ws.close(); } catch(x){} ws = null; }
  if (webcamStream) {
    webcamStream.getTracks().forEach(function(t) { t.stop(); });
    webcamStream = null;
  }
  _wsBusy    = false;
  _lastFaces = [];
  _inferCanvas = null;

  // Clean up liveness challenge state
  _hideLivenessOverlay();
  _livenessPassed    = false;
  _livenessSessionId = null;

  var badge       = document.getElementById('live-badge');
  var placeholder = document.getElementById('cam-placeholder');
  var startBtn    = document.getElementById('start-btn');
  var stopBtn     = document.getElementById('stop-btn');
  if (badge)       badge.classList.remove('active');
  if (placeholder) placeholder.style.display = 'flex';
  if (startBtn)    startBtn.classList.remove('hidden');
  if (startBtn)    startBtn.disabled = false;
  if (startBtn)    startBtn.textContent = 'Start Recognition';
  if (stopBtn)     stopBtn.classList.add('hidden');
  if (_ctx && _canvas) _ctx.clearRect(0, 0, _canvas.width, _canvas.height);
}

function stopStream() { _doStopStream(); }

function updateThreshold() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    var el = document.getElementById('live-threshold');
    if (el) ws.send(JSON.stringify({ threshold: parseFloat(el.value) / 100 }));
  }
}

function _updateFaceList(faces) {
  var el = document.getElementById('live-faces');
  if (!el) return;
  if (!faces || !faces.length) {
    el.innerHTML = '<p class="text-sm text-muted">No faces detected in frame.</p>';
    return;
  }
  totalFaces += faces.length;
  var fc = document.getElementById('face-count');
  if (fc) fc.textContent = totalFaces;
  el.innerHTML = faces.map(function(f, i) {
    var isSpoof = f.is_spoof;
    var known   = isSpoof ? '⚠️ SPOOF' : (f.is_known ? 'Known' : 'Unknown');
    var cls     = isSpoof ? 'badge-orange' : (f.is_known ? 'badge-green' : 'badge-red');
    var name    = isSpoof ? '⚠️ PRESENTATION ATTACK' : (f.identity || 'UNKNOWN').replace(/_/g, ' ');
    var pct     = ((f.confidence || 0) * 100).toFixed(1) + '%';
    var rowStyle = isSpoof ? 'background:rgba(255,100,0,0.08);border-left:3px solid #f97316;' : '';
    return '<div class="candidate-row" style="' + rowStyle + '">' +
      '<div class="candidate-rank">#' + (i + 1) + '</div>' +
      '<div class="candidate-name" style="flex:1;' + (isSpoof ? 'color:#f97316;font-weight:700' : '') + '">' + name + '</div>' +
      '<div class="candidate-pct">' + pct + '</div>' +
      '<span class="badge ' + cls + '">' + known + '</span>' +
      '</div>';
  }).join('');
}


function takeSnapshot() {
  if (!_canvas || !webcamStream) return;
  _canvas.toBlob(function(blob) {
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'snapshot_' + Date.now() + '.jpg';
    a.click();
  }, 'image/jpeg', 0.9);
}

// ---- VIDEO UPLOAD ----

async function processVideo() {
  var videoInput = document.getElementById('video-input');
  var file = videoInput ? videoInput.files[0] : null;
  if (!file) { showAlert('alert-zone', 'Please select a video file first.', 'warning'); return; }

  setLoading('process-btn', true, 'Uploading…');
  var content = document.getElementById('video-result-content');
  if (content) content.innerHTML =
    '<div style="text-align:center;padding:20px">' +
    '<div class="spinner" style="margin:0 auto 12px;width:28px;height:28px;border-width:3px"></div>' +
    '<div style="font-size:.9rem;color:var(--dim)">Processing video… this may take a minute.</div>' +
    '<div id="upload-detail" style="font-size:.75rem;color:var(--muted);margin-top:6px">' +
      'Uploading ' + (file.size > 1048576 ? (file.size/1048576).toFixed(1) + ' MB' : Math.round(file.size/1024) + ' KB') +
    '</div></div>';

  var fd = new FormData();
  fd.append('file', file);
  var threshEl = document.getElementById('vid-threshold');
  var everyEl  = document.getElementById('every-n');
  fd.append('threshold', (parseFloat(threshEl ? threshEl.value : '35') / 100).toFixed(2));
  fd.append('process_every', everyEl ? everyEl.value : '3');

  // Update status message after upload completes (no streaming progress from server)
  var uploadedAt = Date.now();
  var progressMsg = setInterval(function() {
    var el = document.getElementById('upload-detail');
    if (el) {
      var secs = Math.round((Date.now() - uploadedAt) / 1000);
      el.textContent = 'Processing frames… ' + secs + 's elapsed';
    }
  }, 1000);

  try {
    var res   = await API.upload('/api/demo/process', fd);
    clearInterval(progressMsg);
    var stats = res.stats || {};
    var n     = stats.faces_detected || 0;
    var warn  = (n === 0)
      ? '<div class="alert alert-warning mb-3">⚠️ No faces detected. Try a clearer video or lower the confidence threshold.</div>'
      : '';
    if (content) content.innerHTML =
      '<div class="alert alert-success mb-3">✅ Video processed successfully.</div>' + warn +
      '<div class="table-wrap mb-3"><table><tbody>' +
      '<tr><td class="text-muted text-sm">Total frames</td><td class="mono">' + (stats.total_frames || 0) + '</td></tr>' +
      '<tr><td class="text-muted text-sm">Frames analysed</td><td class="mono">' + (stats.processed || 0) + '</td></tr>' +
      '<tr><td class="text-muted text-sm">Quality skipped</td><td class="mono">' + (stats.quality_skipped || 0) + '</td></tr>' +
      '<tr><td class="text-muted text-sm">Face detections</td><td class="mono">' + n + '</td></tr>' +
      '<tr><td class="text-muted text-sm">Confirmed identity</td><td class="mono">' + (stats.confirmed_identity || 'UNKNOWN') + '</td></tr>' +
      '</tbody></table></div>' +
      '<a href="' + res.output_url + '" class="btn btn-primary w-full" download="annotated.mp4" target="_blank">⬇️ Download Annotated Video</a>' +
      '<div class="mt-3"><video controls style="width:100%;border-radius:6px;margin-top:8px" preload="metadata">' +
      '<source src="' + res.output_url + '" type="video/mp4"></video></div>';
  } catch(e) {
    clearInterval(progressMsg);
    if (content) content.innerHTML =
      '<div class="alert alert-danger">' +
      '<strong>❌ Processing failed:</strong><br>' + (e.message || 'Unknown error') +
      '<br><small style="opacity:.7">Check the server logs for details.</small>' +
      '</div>';
  } finally {
    setLoading('process-btn', false);
  }
}

// ---- RETRAIN ----

var _retrainPollTimer = null;

async function triggerRetrain() {
  var btn     = document.getElementById('retrain-btn');
  var badge   = document.getElementById('retrain-badge');
  var logWrap = document.getElementById('retrain-log-wrap');
  var logEl   = document.getElementById('retrain-log');
  if (btn)     { btn.disabled = true; btn.textContent = 'Starting...'; }
  if (logWrap) logWrap.style.display = 'block';
  if (logEl)   logEl.textContent = '';
  _setBadge(badge, 'running');

  var jobId = null;
  try {
    var res = await API.post('/api/pipeline/retrain', {});
    jobId = res.job_id;
    if (logEl) logEl.textContent = res.already_running
      ? 'Retrain already running - attaching.\n'
      : 'Retrain started (job: ' + jobId + ')\n';
  } catch(err) {
    if (logEl) logEl.textContent = 'Failed to start: ' + err.message;
    if (btn)   { btn.disabled = false; btn.textContent = 'Retrain from Gallery'; }
    _setBadge(badge, 'error');
    return;
  }

  if (_retrainPollTimer) clearInterval(_retrainPollTimer);
  _retrainPollTimer = setInterval(async function() {
    try {
      var s = await API.get('/api/pipeline/retrain/status/' + jobId);
      if (logEl) { logEl.textContent = (s.log || []).join('\n'); logEl.scrollTop = logEl.scrollHeight; }
      if (s.status === 'done') {
        clearInterval(_retrainPollTimer); _retrainPollTimer = null;
        _setBadge(badge, 'done');
        if (btn) { btn.disabled = false; btn.textContent = 'Retrain from Gallery'; }
        showAlert('alert-zone', 'Retrain complete! Stream now updated.', 'success');
      } else if (s.status === 'error') {
        clearInterval(_retrainPollTimer); _retrainPollTimer = null;
        _setBadge(badge, 'error');
        if (btn) { btn.disabled = false; btn.textContent = 'Retrain from Gallery'; }
        showAlert('alert-zone', 'Retrain failed: ' + (s.error || 'Unknown error'), 'danger');
      }
    } catch(x) {}
  }, 2000);
}

function _setBadge(el, state) {
  if (!el) return;
  var map = {
    idle:    { text: 'Idle',     bg: 'var(--surface2,#2a2a3a)', color: 'var(--dim,#888)' },
    running: { text: 'Training', bg: '#1a3a5c',                 color: '#7ec8e3'         },
    done:    { text: 'Done',     bg: '#0f3020',                 color: '#4ade80'         },
    error:   { text: 'Error',    bg: '#3a1010',                 color: '#f87171'         }
  };
  var s = map[state] || map.idle;
  el.textContent = s.text;
  el.style.background = s.bg;
  el.style.color = s.color;
}
