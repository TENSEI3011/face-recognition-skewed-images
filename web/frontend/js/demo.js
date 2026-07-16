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

  // Inference canvas: fixed 320×240 — small enough for fast encode, large enough to detect faces
  _inferCanvas        = document.createElement('canvas');
  _inferCanvas.width  = 320;
  _inferCanvas.height = 240;

  // Scale factors so face boxes (from server, in original-frame coords) map to display canvas
  _scaleX = vw / 320;
  _scaleY = vh / 240;
  _lastFaces = [];

  // Start the smooth 60fps display loop (draws webcam + overlays boxes)
  _startDisplayLoop();

  if (startBtn) { startBtn.textContent = 'Connecting...'; }

  // Step 7: open WebSocket
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

    // Use setInterval instead of requestAnimationFrame to control send rate independently
    // of the browser render loop. This prevents frame queue build-up that causes lag.
    var fpsEl = document.getElementById('fps-select');
    var intervalMs = fpsEl ? parseInt(fpsEl.value, 10) : 333;
    _sendTimer = setInterval(_sendFrame, intervalMs);

    // When user changes FPS dropdown, restart the interval
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
    _wsBusy = false;   // unblock sender immediately — display is independent
    try {
      var msg = JSON.parse(event.data);
      if (msg.error) {
        showAlert('alert-zone', 'Server error: ' + msg.error, 'warning');
        stopStream();
        return;
      }
      if (msg.type === 'frame') {
        // Store face results for the display loop to draw as overlay.
        // We deliberately IGNORE msg.frame (server JPEG) — the display loop
        // draws the live webcam directly so it is always lag-free.
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
  }, 'image/jpeg', 0.65);   // 0.65 quality — good enough for face detection, smaller payload
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
    var known = f.is_known;
    var label = (f.identity || 'UNKNOWN').replace(/_/g, ' ');
    var pct   = ((f.confidence || 0) * 100).toFixed(1) + '%';
    var text  = label + '  ' + pct;
    var color = known ? '#00e676' : '#ff1744';
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

  var badge       = document.getElementById('live-badge');
  var placeholder = document.getElementById('cam-placeholder');
  var startBtn    = document.getElementById('start-btn');
  var stopBtn     = document.getElementById('stop-btn');
  if (badge)       badge.classList.remove('active');
  if (placeholder) placeholder.style.display = 'flex';
  if (startBtn)    startBtn.classList.remove('hidden');
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
    var known = f.is_known ? 'Known' : 'Unknown';
    var cls   = f.is_known ? 'badge-green' : 'badge-red';
    var name  = (f.identity || 'UNKNOWN').replace(/_/g, ' ');
    var pct   = ((f.confidence || 0) * 100).toFixed(1) + '%';
    return '<div class="candidate-row">' +
      '<div class="candidate-rank">#' + (i + 1) + '</div>' +
      '<div class="candidate-name" style="flex:1">' + name + '</div>' +
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
