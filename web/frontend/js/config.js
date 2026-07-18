// config.js

async function loadConfig() {
  try {
    const cfg = await API.get('/api/config');
    applyConfig(cfg);
  } catch (e) {
    showAlert('alert-zone', `Failed to load config: ${e.message}`, 'danger');
  }
}

function applyConfig(cfg) {
  const mods = cfg.modalities || [];
  document.getElementById('mod-hog').checked      = mods.includes('hog');
  document.getElementById('mod-lbp').checked      = mods.includes('lbp');
  document.getElementById('mod-geometry').checked = mods.includes('geometry');
  document.getElementById('mod-arcface').checked  = mods.includes('arcface');

  if (cfg.pca_variance != null) {
    const pct = Math.round(cfg.pca_variance * 100);
    document.getElementById('pca-variance').value = pct;
    document.getElementById('pca-val').textContent = pct + '%';
  }

  if (cfg.threshold != null) {
    const pct = Math.round(cfg.threshold * 100);
    document.getElementById('conf-threshold').value = pct;
    document.getElementById('thresh-val').textContent = pct + '%';
  }

  if (cfg.top_k != null) {
    document.getElementById('top-k').value = cfg.top_k;
  }

  if (cfg.svm_kernel) {
    document.getElementById('svm-kernel').value = cfg.svm_kernel;
  }

  // ── Liveness / Anti-Spoofing ─────────────────────────────────────────────
  if (cfg.liveness_enabled != null) {
    document.getElementById('liveness-enabled').checked = cfg.liveness_enabled;
  }
  if (cfg.liveness_threshold != null) {
    const pct = Math.round(cfg.liveness_threshold * 100);
    document.getElementById('liveness-threshold').value = pct;
    document.getElementById('liveness-val').textContent = pct + '%';
  }

  renderSummary(cfg);
}

function renderSummary(cfg) {
  const mods = cfg.modalities || [];
  const livenessOn = cfg.liveness_enabled !== false;
  const livThresh  = cfg.liveness_threshold != null
    ? (cfg.liveness_threshold * 100).toFixed(0) + '%'
    : '50%';

  const el = document.getElementById('config-summary');
  el.innerHTML = `
    <div class="table-wrap"><table><tbody>
      <tr>
        <td class="text-muted text-sm" style="width:160px">Modalities</td>
        <td>${mods.map(m => `<span class="badge badge-blue" style="margin-right:4px">${m.toUpperCase()}</span>`).join('')}</td>
      </tr>
      <tr>
        <td class="text-muted text-sm">PCA Variance</td>
        <td class="mono">${cfg.pca_variance != null ? (cfg.pca_variance * 100).toFixed(0) + '%' : '—'}</td>
      </tr>
      <tr>
        <td class="text-muted text-sm">SVM Kernel</td>
        <td class="mono">${cfg.svm_kernel || '—'}</td>
      </tr>
      <tr>
        <td class="text-muted text-sm">Threshold</td>
        <td class="mono">${cfg.threshold != null ? (cfg.threshold * 100).toFixed(0) + '%' : '—'}</td>
      </tr>
      <tr>
        <td class="text-muted text-sm">Top-K</td>
        <td class="mono">${cfg.top_k ?? '—'}</td>
      </tr>
      <tr>
        <td class="text-muted text-sm">🛡️ Liveness</td>
        <td>
          <span class="badge ${livenessOn ? 'badge-blue' : ''}" style="margin-right:6px">
            ${livenessOn ? '✅ Enabled' : '⚠️ Disabled'}
          </span>
          <span class="text-muted text-sm">threshold: ${livThresh}</span>
        </td>
      </tr>
    </tbody></table></div>
  `;
}

async function saveConfig() {
  const mods = [];
  if (document.getElementById('mod-hog').checked)      mods.push('hog');
  if (document.getElementById('mod-lbp').checked)      mods.push('lbp');
  if (document.getElementById('mod-geometry').checked) mods.push('geometry');
  if (document.getElementById('mod-arcface').checked)  mods.push('arcface');

  if (!mods.length) {
    showAlert('alert-zone', 'Select at least one modality.', 'warning');
    return;
  }

  const payload = {
    modalities:          mods,
    pca_variance:        parseInt(document.getElementById('pca-variance').value, 10) / 100,
    svm_kernel:          document.getElementById('svm-kernel').value,
    threshold:           parseInt(document.getElementById('conf-threshold').value, 10) / 100,
    top_k:               parseInt(document.getElementById('top-k').value, 10),
    // ── Liveness ─────────────────────────────────────────────────────────
    liveness_enabled:    document.getElementById('liveness-enabled').checked,
    liveness_threshold:  parseInt(document.getElementById('liveness-threshold').value, 10) / 100,
  };

  setLoading('save-btn', true, 'Saving…');
  try {
    const res = await API.post('/api/config', payload);
    showAlert('alert-zone', res.message, 'success');
    renderSummary(res.config);
  } catch (e) {
    showAlert('alert-zone', `Save failed: ${e.message}`, 'danger');
  } finally {
    setLoading('save-btn', false);
  }
}

document.addEventListener('DOMContentLoaded', loadConfig);
