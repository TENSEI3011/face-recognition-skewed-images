// results.js

const EXP_LABELS = {
  baseline:    'Baseline',
  ablation:    'Ablation Study',
  pose_study:  'Pose Study',
  degradation: 'Degradation Sweep',
};

const EXP_DESCRIPTIONS = {
  baseline:    'Full pipeline evaluation — Rank-k IR, EER, TAR@FAR, AUC, d-prime.',
  ablation:    'Tests all 10 modality combinations (HOG, LBP, Geometry, ArcFace and subsets).',
  pose_study:  'Stratified evaluation by yaw/pitch angle and UAV altitude bins.',
  degradation: 'Accuracy vs. degradation level (CLEAN → EXTREME) and altitude sweep.',
};

let activeTab = 'baseline';
const loadedTabs = new Set();
let pollInterval = null;

function switchTab(exp) {
  document.querySelectorAll('.tab-btn').forEach((b, i) => {
    b.classList.toggle('active', b.textContent.trim().toLowerCase().includes(exp.split('_')[0]));
  });
  // Match by data-exp
  document.querySelectorAll('.tab-panel').forEach(p => {
    p.classList.toggle('active', p.dataset.exp === exp);
  });
  activeTab = exp;
  if (!loadedTabs.has(exp)) {
    loadedTabs.add(exp);
    loadExperiment(exp);
  }
}

// Fix tab button click mapping
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.onclick = function() {
    const map = { 'Baseline': 'baseline', 'Ablation Study': 'ablation', 'Pose Study': 'pose_study', 'Degradation': 'degradation' };
    const exp = map[btn.textContent.trim()];
    if (exp) switchTab(exp);
  };
});

async function loadExperiment(exp) {
  const panel = document.getElementById(`tab-${exp}`);
  panel.innerHTML = `<div style="padding:20px;color:var(--muted)"><span class="spinner"></span> Loading ${EXP_LABELS[exp]} results…</div>`;

  try {
    const data = await API.get(`/api/results/${exp}`);
    renderExperiment(exp, data);
  } catch (e) {
    panel.innerHTML = `<div class="alert alert-danger">Failed to load: ${e.message}</div>`;
  }
}

function renderExperiment(exp, data) {
  const panel = document.getElementById(`tab-${exp}`);
  let html = '';

  // Header + run button
  html += `
    <div class="flex justify-between items-center mb-4">
      <div>
        <div style="font-weight:600">${EXP_LABELS[exp]}</div>
        <div class="text-sm text-muted">${EXP_DESCRIPTIONS[exp]}</div>
      </div>
      <button class="btn btn-secondary btn-sm" id="run-btn-${exp}" onclick="runExperiment('${exp}')">
        ▶ Run Experiment
      </button>
    </div>
  `;

  // Job status placeholder
  html += `<div id="job-status-${exp}"></div>`;

  // Metrics table
  if (data.metrics) {
    html += renderMetricsTable(data.metrics);
  } else {
    html += `<div class="alert alert-info mb-4">No metrics yet. Click "Run Experiment" to generate results.</div>`;
  }

  // Plots
  if (data.plots && data.plots.length > 0) {
    html += `<div class="form-label mb-3 mt-2">${data.plot_count} Plot${data.plot_count !== 1 ? 's' : ''}</div>`;
    html += '<div class="plot-grid">';
    data.plots.forEach(plot => {
      html += `
        <div class="plot-card">
          <img src="${plot.data}" alt="${plot.title}" loading="lazy">
          <div class="plot-card-title">${plot.title}</div>
        </div>`;
    });
    html += '</div>';
  } else if (data.metrics) {
    html += `<div class="alert alert-warning mt-3">No plots found in results directory.</div>`;
  }

  panel.innerHTML = html;
}

function renderMetricsTable(m) {
  const METRIC_DEFS = [
    ['rank1',      'Rank-1 IR',        'Top-1 identification accuracy (CMC)',         true],
    ['rank5',      'Rank-5 IR',        'Top-5 identification accuracy',               true],
    ['eer',        'EER',              'Equal Error Rate (FAR = FRR)',                true],
    ['auc',        'AUC',              'Area under ROC Curve',                        false],
    ['tar_at_far', 'TAR @ FAR=0.1%',  'True Accept Rate at 0.1% False Accepts',      true],
    ['d_prime',    "d' (d-prime)",    'Signal detection discriminability index',      false],
  ];

  let html = '<div class="card mb-4"><div class="card-title"><span class="icon">📐</span> Metrics</div><div class="table-wrap"><table><thead><tr><th>Metric</th><th>Value</th><th>Description</th></tr></thead><tbody>';
  METRIC_DEFS.forEach(([key, label, desc, isPct]) => {
    const val = m[key];
    if (val == null) return;
    const disp = isPct ? `${(val * 100).toFixed(2)}%` : val.toFixed(4);
    html += `<tr><td style="font-weight:500">${label}</td><td class="mono" style="font-weight:700;color:var(--blue)">${disp}</td><td class="text-muted text-sm">${desc}</td></tr>`;
  });
  html += '</tbody></table></div></div>';
  return html;
}

async function runExperiment(exp) {
  const btn = document.getElementById(`run-btn-${exp}`);
  btn.disabled = true;
  btn.textContent = '⏳ Starting…';

  try {
    const res = await API.post(`/api/experiments/run/${exp}`, {});
    const jobId = res.job_id;
    showJobStatus(exp, jobId, 'running', '');
    pollJobStatus(exp, jobId);
  } catch (e) {
    showAlert('alert-zone', `Failed to start experiment: ${e.message}`, 'danger');
    btn.disabled = false;
    btn.textContent = '▶ Run Experiment';
  }
}

function showJobStatus(exp, jobId, status, output) {
  const el = document.getElementById(`job-status-${exp}`);
  if (!el) return;
  const colorMap = { running: 'info', done: 'success', error: 'danger', pending: 'warning' };
  const type = colorMap[status] || 'info';
  const statusIcon = { running: '⏳', done: '✅', error: '❌', pending: '⏸' }[status] || '·';
  el.innerHTML = `
    <div class="alert alert-${type} mb-3" style="display:flex;align-items:center;gap:10px;">
      <span style="font-size:1.1em">${statusIcon}</span>
      <div style="flex:1">
        <strong>Job ${jobId} — ${status.toUpperCase()}</strong>
        ${status === 'running' ? ' <span class="spinner" style="width:14px;height:14px"></span>' : ''}
        ${status === 'done' ? '<br><span class="text-sm">Results updated — <a href="" onclick="location.reload();return false">Reload page</a> to see new plots.</span>' : ''}
        ${status === 'error' ? '<br><span class="text-sm" style="color:inherit;opacity:.85">Script failed. Check the console output below for details.<br>Make sure data/gallery/ and data/probe/ have identity subfolders with face images.</span>' : ''}
      </div>
    </div>
    ${output ? `<div class="console-log mb-3" style="max-height:260px;overflow-y:auto;font-size:.78rem;white-space:pre-wrap;word-break:break-all;">${output.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').slice(-3000)}</div>` : ''}
  `;
}

function pollJobStatus(exp, jobId) {
  if (pollInterval) clearInterval(pollInterval);
  pollInterval = setInterval(async () => {
    try {
      const job = await API.get(`/api/experiments/status/${jobId}`);
      showJobStatus(exp, jobId, job.status, job.output);
      if (job.status === 'done' || job.status === 'error') {
        clearInterval(pollInterval);
        const btn = document.getElementById(`run-btn-${exp}`);
        if (btn) { btn.disabled = false; btn.textContent = '▶ Run Experiment'; }
        // Reload results
        loadedTabs.delete(exp);
        setTimeout(() => loadExperiment(exp), 1500);
      }
    } catch {}
  }, 2000);
}

// Init
document.addEventListener('DOMContentLoaded', () => {
  loadedTabs.add('baseline');
  loadExperiment('baseline');
});
