/* batch.js — Batch identification page */

let _selectedFiles = [];

// ── Drag & drop ────────────────────────────────────────────────────────────────
const dropZone = document.getElementById('batch-drop-zone');
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  previewFiles(e.dataTransfer.files);
});

function previewFiles(fileList) {
  _selectedFiles = Array.from(fileList);
  if (!_selectedFiles.length) return;

  document.getElementById('file-count-label').textContent =
    `${_selectedFiles.length} file(s) selected`;

  const thumbsEl = document.getElementById('file-thumbs');
  thumbsEl.innerHTML = '';
  _selectedFiles.slice(0, 20).forEach(file => {
    if (file.type.startsWith('image/')) {
      const img = document.createElement('img');
      img.className = 'thumb-img';
      img.alt = file.name;
      img.title = file.name;
      const url = URL.createObjectURL(file);
      img.src = url;
      img.onclick = () => openLightbox(url);
      thumbsEl.appendChild(img);
    } else {
      const box = document.createElement('div');
      box.style.cssText = `width:70px;height:70px;border-radius:6px;border:1px solid var(--border);
        display:flex;align-items:center;justify-content:center;font-size:1.4rem;background:var(--bg)`;
      box.textContent = file.name.endsWith('.zip') ? '🗜️' : '📄';
      box.title = file.name;
      thumbsEl.appendChild(box);
    }
  });
  if (_selectedFiles.length > 20) {
    const more = document.createElement('div');
    more.style.cssText = `display:flex;align-items:center;font-size:.78rem;color:var(--muted)`;
    more.textContent = `+${_selectedFiles.length - 20} more`;
    thumbsEl.appendChild(more);
  }

  document.getElementById('file-preview').style.display = 'block';
  document.getElementById('batch-run-btn').disabled = false;
}

function openLightbox(src) {
  const lb = document.getElementById('lightbox');
  document.getElementById('lightbox-img').src = src;
  lb.style.display = 'flex';
}

// ── Run batch ─────────────────────────────────────────────────────────────────
async function runBatch() {
  if (!_selectedFiles.length) return;
  setLoading('batch-run-btn', true, 'Processing…');
  clearAlert('alert-zone');
  document.getElementById('batch-summary-row').style.display = 'none';
  document.getElementById('batch-results-card').style.display = 'none';
  document.getElementById('export-csv-btn').style.display = 'none';

  const progress = document.getElementById('batch-progress');
  const progressBar = document.getElementById('batch-progress-bar');
  progress.style.display = 'block';
  progressBar.style.width = '0%';

  // Animate progress bar during upload
  let prog = 0;
  const progInterval = setInterval(() => {
    prog = Math.min(prog + 5, 85);
    progressBar.style.width = prog + '%';
  }, 200);

  const threshold = parseFloat(document.getElementById('batch-threshold').value);
  const topk      = parseInt(document.getElementById('batch-topk').value);

  const fd = new FormData();
  _selectedFiles.forEach(f => fd.append('files', f));
  fd.append('threshold', threshold);
  fd.append('top_k', topk);

  try {
    const data = await API.upload('/api/batch/identify', fd);
    clearInterval(progInterval);
    progressBar.style.width = '100%';
    setTimeout(() => { progress.style.display = 'none'; }, 500);

    renderSummary(data.summary || {});
    renderResults(data.results || []);
    document.getElementById('export-csv-btn').style.display = '';

    if ((data.summary || {}).watchlist_hits > 0) {
      showAlert('alert-zone', `🚨 ${data.summary.watchlist_hits} watchlist hit(s) detected!`, 'danger');
    } else {
      showAlert('alert-zone', `✅ Batch complete: ${(data.summary||{}).detected || 0} faces detected.`, 'success');
    }
  } catch (e) {
    clearInterval(progInterval);
    progress.style.display = 'none';
    showAlert('alert-zone', 'Error: ' + e.message, 'danger');
  } finally {
    setLoading('batch-run-btn', false);
  }
}

function renderSummary(s) {
  document.getElementById('b-total').textContent    = s.total ?? 0;
  document.getElementById('b-detected').textContent = s.detected ?? 0;
  document.getElementById('b-known').textContent    = s.known ?? 0;
  document.getElementById('b-alerts').textContent   = s.watchlist_hits ?? 0;
  document.getElementById('batch-summary-row').style.display = '';
}

function renderResults(results) {
  const tbody = document.getElementById('batch-tbody');
  if (!results.length) {
    tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;padding:20px" class="text-muted">No results.</td></tr>';
    document.getElementById('batch-results-card').style.display = '';
    return;
  }

  tbody.innerHTML = results.map((r, i) => {
    const rowClass = r.error ? 'result-error' : (r.is_known ? 'result-known' : (r.detected ? 'result-unknown' : ''));
    const thumb    = r.thumbnail
      ? `<img src="${r.thumbnail}" class="thumb-img" alt="thumb" onclick="openLightbox('${r.thumbnail}')">`
      : '<div style="width:70px;height:70px;background:var(--bg);border-radius:6px;display:flex;align-items:center;justify-content:center">🚫</div>';
    const candidates = (r.candidates || []).slice(0, 3)
      .map(c => `<div style="font-size:.72rem">${c.rank}. ${c.identity} <span class="text-muted">${(c.confidence*100).toFixed(1)}%</span></div>`)
      .join('');
    const conf = r.confidence != null ? (r.confidence * 100).toFixed(1) + '%' : '—';
    const known = r.detected
      ? (r.is_known
          ? '<span class="badge badge-green">✓ Known</span>'
          : '<span class="badge badge-red">✗ Unknown</span>')
      : '<span class="badge badge-gray">—</span>';
    const alert = r.watchlist_hit
      ? '<span class="badge badge-red">🚨</span>'
      : '<span class="badge badge-gray">—</span>';

    return `<tr class="${rowClass}">
      <td>${thumb}</td>
      <td class="mono" style="font-size:.78rem;max-width:160px;word-break:break-all">${r.filename || '—'}</td>
      <td>${r.error ? `<span class="badge badge-yellow">Error</span>` : (r.detected ? '<span class="badge badge-green">✓</span>' : '<span class="badge badge-gray">No face</span>')}</td>
      <td style="font-weight:${r.is_known ? '600' : '400'}">${r.identity || (r.error ? `<span class="text-muted">${r.error}</span>` : '—')}</td>
      <td class="mono">${conf}</td>
      <td>${known}</td>
      <td>${alert}</td>
      <td>${candidates || '<span class="text-muted" style="font-size:.75rem">—</span>'}</td>
    </tr>`;
  }).join('');

  document.getElementById('batch-results-card').style.display = '';
}

function exportCSV() {
  window.open('/api/batch/export/csv', '_blank');
}
