// dashboard.js

async function loadDashboard() {
  try {
    const [status, results] = await Promise.allSettled([
      API.get('/api/status'),
      API.get('/api/results/baseline'),
    ]);

    // Pipeline status
    if (status.status === 'fulfilled') {
      const d = status.value;
      const dot = document.getElementById('detail-dot');
      const dtxt = document.getElementById('detail-status-text');

      if (d.loaded) {
        dot.className = 'status-dot loaded';
        dtxt.textContent = 'Pipeline loaded and ready';
        document.getElementById('stat-pipeline').textContent = '✓';
        document.getElementById('stat-pipeline').style.color = 'var(--success)';
        document.getElementById('stat-pipeline-sub').textContent = 'Models loaded';
      } else if (d.error) {
        dot.className = 'status-dot';
        dtxt.textContent = 'Pipeline not loaded';
        document.getElementById('stat-pipeline').textContent = '✗';
        document.getElementById('stat-pipeline').style.color = 'var(--danger)';
        document.getElementById('stat-pipeline-sub').textContent = 'Run an experiment first';
        const errEl = document.getElementById('pipeline-error');
        errEl.textContent = d.error;
        errEl.classList.remove('hidden');
      }

      document.getElementById('stat-gallery').textContent = d.gallery_count ?? '—';
      document.getElementById('td-models').textContent = d.models_dir || '—';

      if (d.modalities) {
        const mods = d.modalities;
        document.getElementById('stat-modalities').textContent = mods.length;
        document.getElementById('stat-mod-names').textContent = mods.join(' + ').toUpperCase();
        document.getElementById('td-modalities').innerHTML = mods.map(m =>
          `<span class="badge badge-blue" style="margin-right:4px">${m.toUpperCase()}</span>`
        ).join('');
      }
    }

    // Metrics
    if (results.status === 'fulfilled' && results.value.metrics) {
      const m = results.value.metrics;
      let html = '<div class="table-wrap"><table><tbody>';
      const rows = [
        ['Rank-1 IR', m.rank1 != null ? formatPct(m.rank1) : '—'],
        ['Rank-5 IR', m.rank5 != null ? formatPct(m.rank5) : '—'],
        ['EER',       m.eer  != null ? formatPct(m.eer)  : '—'],
        ['AUC',       m.auc  != null ? formatNum(m.auc, 4) : '—'],
        ['TAR@FAR=0.1%', m.tar_at_far != null ? formatPct(m.tar_at_far) : '—'],
      ];
      rows.forEach(([k, v]) => {
        html += `<tr><td style="color:var(--muted);font-size:.78rem">${k}</td><td class="mono" style="font-weight:600">${v}</td></tr>`;
      });
      html += '</tbody></table></div>';

      // Show rank-1 in stat card
      if (m.rank1 != null) {
        document.getElementById('stat-rank1').textContent = formatPct(m.rank1);
      }

      document.getElementById('metrics-content').innerHTML = html;
    } else {
      document.getElementById('metrics-content').innerHTML =
        '<p class="text-sm text-muted">No baseline results yet. Run the baseline experiment to generate metrics.</p>' +
        '<a href="/results" class="btn btn-secondary btn-sm mt-3">Go to Experiments →</a>';
    }

    // Timestamp
    document.getElementById('last-updated').textContent =
      'Updated ' + new Date().toLocaleTimeString();

  } catch (e) {
    showAlert('alert-zone', `Failed to load dashboard: ${e.message}`, 'danger');
  }
}

document.addEventListener('DOMContentLoaded', loadDashboard);
