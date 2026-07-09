// identify.js

const imgInput  = document.getElementById('img-input');
const imgPreview = document.getElementById('img-preview');
const uploadZone = document.getElementById('upload-zone');
const previewWrap = document.getElementById('preview-wrap');

imgInput.addEventListener('change', () => {
  if (imgInput.files[0]) showPreview(imgInput.files[0]);
});

uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f) { imgInput.files = e.dataTransfer.files; showPreview(f); }
});

function showPreview(file) {
  const reader = new FileReader();
  reader.onload = e => {
    imgPreview.src = e.target.result;
    imgPreview.style.display = 'block';
    previewWrap.style.display = 'none';
  };
  reader.readAsDataURL(file);
}

async function runIdentify() {
  const file = imgInput.files[0];
  if (!file) { showAlert('alert-zone', 'Please upload an image first.', 'warning'); return; }

  clearAlert('alert-zone');
  setLoading('identify-btn', true, 'Analysing…');
  document.getElementById('result-placeholder').style.display = 'none';
  document.getElementById('result-content').classList.add('hidden');

  const fd = new FormData();
  fd.append('file', file);
  fd.append('top_k', document.getElementById('top-k').value);
  fd.append('threshold', (parseFloat(document.getElementById('threshold').value) / 100).toFixed(2));
  fd.append('degradation', document.getElementById('degradation').value);
  fd.append('use_sr', document.getElementById('use-sr').checked ? 'true' : 'false');

  try {
    const res = await API.upload('/api/identify', fd);
    renderResult(res);
  } catch (e) {
    showAlert('alert-zone', `Identification failed: ${e.message}`, 'danger');
    document.getElementById('result-placeholder').style.display = 'block';
  } finally {
    setLoading('identify-btn', false);
  }
}

function renderResult(res) {
  document.getElementById('result-content').classList.remove('hidden');

  // Watchlist alert
  const wlAlert = document.getElementById('wl-alert');
  if (wlAlert) wlAlert.style.display = res.watchlist_hit ? 'block' : 'none';

  // Verdict banner
  const banner = document.getElementById('verdict-banner');
  const icon   = document.getElementById('verdict-icon');
  const name   = document.getElementById('verdict-name');
  const sub    = document.getElementById('verdict-sub');

  if (!res.detected) {
    banner.style.background = '#fef2f2';
    icon.textContent  = '❌';
    name.textContent  = 'No Face Detected';
    name.style.color  = 'var(--danger)';
    sub.textContent   = res.message || 'The pipeline could not find a face in the image.';
  } else if (res.is_known) {
    banner.style.background = '#f0fdf4';
    icon.textContent  = '✅';
    name.textContent  = res.identity.replace(/_/g, ' ');
    name.style.color  = 'var(--success)';
    sub.textContent   = `Confidence: ${(res.confidence * 100).toFixed(1)}%  ·  Threshold: ${(res.threshold * 100).toFixed(0)}%  ·  Profile: ${res.degradation}${res.use_sr ? '  ·  SR: On' : ''}`;
  } else {
    banner.style.background = '#fff7ed';
    icon.textContent  = '⚠️';
    name.textContent  = 'UNKNOWN';
    name.style.color  = 'var(--warning)';
    sub.textContent   = `Best match confidence ${(res.confidence * 100).toFixed(1)}% is below threshold ${(res.threshold * 100).toFixed(0)}%`;
  }

  // Annotated image
  document.getElementById('annotated-img').src = res.annotated_image;

  // Candidates
  const candidates = res.candidates || [];
  const list = document.getElementById('candidates-list');
  if (!candidates.length) {
    list.innerHTML = '<p class="text-sm text-muted">No candidates returned.</p>';
    return;
  }
  list.innerHTML = candidates.map((c, i) => `
    <div class="candidate-row">
      <div class="candidate-rank">#${c.rank}</div>
      <div class="candidate-name">${c.identity.replace(/_/g, ' ')}</div>
      <div class="candidate-bar-wrap">
        <div class="progress">
          <div class="progress-bar ${i === 0 && res.is_known ? 'green' : ''}"
               style="width:${(c.confidence * 100).toFixed(1)}%"></div>
        </div>
      </div>
      <div class="candidate-pct">${(c.confidence * 100).toFixed(1)}%</div>
    </div>
  `).join('');
}
