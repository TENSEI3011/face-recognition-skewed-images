// demo.js — Live webcam stream + video upload processing

let ws             = null;
let streamInterval = null;   // setInterval handle (kept for compatibility)
let webcamStream   = null;
let totalFrames    = 0;
let totalFaces     = 0;
let currentMode    = 'live';

// Back-pressure flag: do NOT send a new frame until the server has replied
let _wsBusy        = false;
// Persistent offscreen canvas — created once, reused every frame
let _offscreenCanvas = null;
let _inferCanvas     = null;   // smaller canvas for inference
let _rafHandle       = null;   // requestAnimationFrame handle

// ── Mode switching ─────────────────────────────────────────────────────────────
function setMode(mode) {
  currentMode = mode;
  document.getElementById('mode-live').classList.toggle('hidden',  mode !== 'live');
  document.getElementById('mode-video').classList.toggle('hidden', mode !== 'video');
  document.getElementById('tab-live-btn').classList.toggle('active', mode === 'live');
  document.getElementById('tab-video-btn').classList.toggle('active', mode === 'video');
  if (mode === 'video' && ws) stopStream();
}

// ── LIVE WEBCAM STREAM ─────────────────────────────────────────────────────────

const video   = document.getElementById('webcam-video');
const canvas  = document.getElementById('stream-canvas');
const ctx     = canvas.getContext('2d');

async function startStream() {
  clearAlert('alert-zone');

  // Get webcam
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    video.srcObject = webcamStream;
    await video.play();
  } catch (e) {
    showAlert('alert-zone', `Camera error: ${e.message}. Check browser permissions.`, 'danger');
    return;
  }

  // Resize canvas to match video
  video.addEventListener('loadedmetadata', () => {
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
  });

  // Connect WebSocket
  const wsProto = location.protocol === 'https:' ? 'wss' : 'ws';
  const wsUrl   = `${wsProto}://${location.host}/ws/stream`;

  try {
    ws = new WebSocket(wsUrl);
  } catch (e) {
    showAlert('alert-zone', `WebSocket connection failed: ${e.message}`, 'danger');
    return;
  }

  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    document.getElementById('live-badge').classList.add('active');
    document.getElementById('cam-placeholder').style.display = 'none';
    document.getElementById('start-btn').classList.add('hidden');
    document.getElementById('stop-btn').classList.remove('hidden');

    _wsBusy = false;

    // Build persistent canvases once video dimensions are known
    const ensureCanvases = () => {
      if (!video.videoWidth) return;
      _offscreenCanvas = document.createElement('canvas');
      _offscreenCanvas.width  = video.videoWidth;
      _offscreenCanvas.height = video.videoHeight;
      // Inference canvas: half resolution for faster server processing
      _inferCanvas = document.createElement('canvas');
      _inferCanvas.width  = Math.round(video.videoWidth  / 2);
      _inferCanvas.height = Math.round(video.videoHeight / 2);
    };

    if (video.videoWidth) {
      ensureCanvases();
    } else {
      video.addEventListener('loadedmetadata', ensureCanvases, { once: true });
    }

    // Use rAF loop so we send as fast as the server can keep up
    const loop = () => {
      sendFrame();
      _rafHandle = requestAnimationFrame(loop);
    };
    _rafHandle = requestAnimationFrame(loop);
  };

  ws.onmessage = (event) => {
    // Release back-pressure as soon as the server responds
    _wsBusy = false;
    try {
      const msg = JSON.parse(event.data);
      if (msg.type === 'frame' && msg.frame) {
        // Draw annotated frame on canvas using createImageBitmap (non-blocking)
        const blob = _b64ToBlob(msg.frame);
        createImageBitmap(blob).then(bitmap => {
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
          bitmap.close();
        });
        updateFaceList(msg.faces || []);
        totalFrames++;
        document.getElementById('frame-count').textContent = totalFrames;
      }
    } catch {}
  };

  ws.onerror = () => {
    showAlert('alert-zone', 'WebSocket error. Is the server running?', 'danger');
    stopStream();
  };

  ws.onclose = () => {
    stopStream();
  };
}

function sendFrame() {
  // Skip if socket not ready, video not started, or server hasn't replied yet
  if (!ws || ws.readyState !== WebSocket.OPEN || !video.videoWidth) return;
  if (_wsBusy) return;   // ← back-pressure: don't pile up frames
  if (!_inferCanvas || !_offscreenCanvas) return;

  _wsBusy = true;

  // Draw at reduced resolution for inference (faster round-trip)
  const ictx = _inferCanvas.getContext('2d');
  ictx.drawImage(video, 0, 0, _inferCanvas.width, _inferCanvas.height);

  _inferCanvas.toBlob(blob => {
    if (blob && ws && ws.readyState === WebSocket.OPEN) {
      blob.arrayBuffer().then(buf => ws.send(buf));
    } else {
      _wsBusy = false;  // release if we couldn't send
    }
  }, 'image/jpeg', 0.70);
}

// Helper: convert base64 string → Blob without data URI overhead
function _b64ToBlob(b64) {
  const binary = atob(b64);
  const bytes  = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: 'image/jpeg' });
}

function stopStream() {
  if (_rafHandle)    { cancelAnimationFrame(_rafHandle); _rafHandle = null; }
  if (streamInterval){ clearInterval(streamInterval); streamInterval = null; }
  if (ws)            { ws.close(); ws = null; }
  if (webcamStream)  { webcamStream.getTracks().forEach(t => t.stop()); webcamStream = null; }

  _wsBusy = false;
  _offscreenCanvas = null;
  _inferCanvas     = null;

  document.getElementById('live-badge').classList.remove('active');
  document.getElementById('cam-placeholder').style.display = 'flex';
  document.getElementById('start-btn').classList.remove('hidden');
  document.getElementById('stop-btn').classList.add('hidden');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function updateThreshold() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    const thresh = parseFloat(document.getElementById('live-threshold').value) / 100;
    ws.send(JSON.stringify({ threshold: thresh }));
  }
}

function updateFaceList(faces) {
  const el = document.getElementById('live-faces');
  if (!faces.length) {
    el.innerHTML = '<p class="text-sm text-muted">No faces detected in frame.</p>';
    return;
  }
  totalFaces += faces.length;
  document.getElementById('face-count').textContent = totalFaces;

  el.innerHTML = faces.map((f, i) => `
    <div class="candidate-row">
      <div class="candidate-rank">#${i + 1}</div>
      <div class="candidate-name" style="flex:1">${f.identity.replace(/_/g, ' ')}</div>
      <div class="candidate-pct">${(f.confidence * 100).toFixed(1)}%</div>
      <span class="badge ${f.is_known ? 'badge-green' : 'badge-red'}">${f.is_known ? 'Known' : 'Unknown'}</span>
    </div>
  `).join('');
}

function takeSnapshot() {
  if (!webcamStream) return;
  const a = document.createElement('a');
  canvas.toBlob(blob => {
    a.href = URL.createObjectURL(blob);
    a.download = `snapshot_${Date.now()}.jpg`;
    a.click();
  }, 'image/jpeg', 0.9);
}

// ── VIDEO UPLOAD PROCESSING ────────────────────────────────────────────────────

const videoInput = document.getElementById('video-input');
const videoZone  = document.getElementById('video-upload-zone');

videoInput.addEventListener('change', () => {
  const f = videoInput.files[0];
  document.getElementById('video-file-name').textContent = f ? `Selected: ${f.name}` : '';
});

videoZone.addEventListener('dragover', e => { e.preventDefault(); videoZone.classList.add('drag-over'); });
videoZone.addEventListener('dragleave', () => videoZone.classList.remove('drag-over'));
videoZone.addEventListener('drop', e => {
  e.preventDefault();
  videoZone.classList.remove('drag-over');
  videoInput.files = e.dataTransfer.files;
  const f = videoInput.files[0];
  document.getElementById('video-file-name').textContent = f ? `Selected: ${f.name}` : '';
});

async function processVideo() {
  const file = videoInput.files[0];
  if (!file) { showAlert('alert-zone', 'Please select a video file.', 'warning'); return; }

  setLoading('process-btn', true, 'Processing…');
  const content = document.getElementById('video-result-content');
  content.innerHTML = `<div><span class="spinner"></span> Processing video — this may take a while…</div>`;

  const fd = new FormData();
  fd.append('file', file);
  fd.append('threshold', (parseFloat(document.getElementById('vid-threshold').value) / 100).toFixed(2));
  fd.append('process_every', document.getElementById('every-n').value);

  try {
    const res = await API.upload('/api/demo/process', fd);
    const stats = res.stats || {};
    content.innerHTML = `
      <div class="alert alert-success mb-3">✅ Video processed successfully.</div>
      <div class="table-wrap mb-3"><table><tbody>
        <tr><td class="text-muted text-sm">Total frames</td><td class="mono">${stats.total_frames ?? '—'}</td></tr>
        <tr><td class="text-muted text-sm">Frames analysed</td><td class="mono">${stats.processed ?? '—'}</td></tr>
        <tr><td class="text-muted text-sm">Face detections</td><td class="mono">${stats.faces_detected ?? '—'}</td></tr>
      </tbody></table></div>
      <a href="${res.output_url}" class="btn btn-primary w-full" download>⬇ Download Annotated Video</a>
      <div class="mt-3">
        <video controls style="width:100%;border-radius:6px;margin-top:8px" src="${res.output_url}"></video>
      </div>
    `;
  } catch (e) {
    content.innerHTML = `<div class="alert alert-danger">${e.message}</div>`;
  } finally {
    setLoading('process-btn', false);
  }
}

// ── RETRAIN & RELOAD ───────────────────────────────────────────────────────────

let _retrainPollTimer = null;

async function triggerRetrain() {
  const btn   = document.getElementById('retrain-btn');
  const badge = document.getElementById('retrain-badge');
  const logWrap = document.getElementById('retrain-log-wrap');
  const logEl   = document.getElementById('retrain-log');

  // Disable button, show log panel
  btn.disabled  = true;
  btn.textContent = '⏳ Starting…';
  logWrap.style.display = 'block';
  logEl.textContent     = '';
  _setBadge(badge, 'running');

  let jobId = null;
  try {
    const res = await API.post('/api/pipeline/retrain', {});
    jobId = res.job_id;

    if (res.already_running) {
      _appendLog(logEl, '⚠️ A retrain is already running — attaching to existing job.');
    } else {
      _appendLog(logEl, '🚀 Retrain job started (ID: ' + jobId + ')');
    }
  } catch (err) {
    _appendLog(logEl, '❌ Failed to start retrain: ' + err.message);
    btn.disabled    = false;
    btn.textContent = '🔄 Retrain from Gallery';
    _setBadge(badge, 'error');
    return;
  }

  // Poll for status every 2 seconds
  if (_retrainPollTimer) clearInterval(_retrainPollTimer);
  _retrainPollTimer = setInterval(async () => {
    try {
      const status = await API.get('/api/pipeline/retrain/status/' + jobId);

      // Append any new log lines
      const lines = status.log || [];
      logEl.textContent = lines.join('\n');
      logEl.scrollTop   = logEl.scrollHeight;

      if (status.status === 'done') {
        clearInterval(_retrainPollTimer);
        _retrainPollTimer = null;
        _setBadge(badge, 'done');
        btn.disabled    = false;
        btn.textContent = '🔄 Retrain from Gallery';
        showAlert('alert-zone',
          '✅ Retrain complete! The live stream now recognises newly added faces.',
          'success');
      } else if (status.status === 'error') {
        clearInterval(_retrainPollTimer);
        _retrainPollTimer = null;
        _setBadge(badge, 'error');
        btn.disabled    = false;
        btn.textContent = '🔄 Retrain from Gallery';
        showAlert('alert-zone',
          '❌ Retrain failed: ' + (status.error || 'Unknown error'),
          'danger');
      } else {
        btn.textContent = '⏳ Training…';
        _setBadge(badge, 'running');
      }
    } catch (err) {
      // Network blip — don't stop polling
    }
  }, 2000);
}

function _appendLog(el, line) {
  el.textContent += (el.textContent ? '\n' : '') + line;
  el.scrollTop    = el.scrollHeight;
}

function _setBadge(el, state) {
  const map = {
    idle:    { text: 'Idle',     bg: 'var(--surface2,#2a2a3a)', color: 'var(--dim,#888)' },
    running: { text: 'Training…', bg: '#1a3a5c',                color: '#7ec8e3'          },
    done:    { text: '✅ Done',   bg: '#0f3020',                color: '#4ade80'          },
    error:   { text: '❌ Error',  bg: '#3a1010',                color: '#f87171'          },
  };
  const s = map[state] || map.idle;
  el.textContent        = s.text;
  el.style.background   = s.bg;
  el.style.color        = s.color;
}
