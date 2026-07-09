/* audit.js — Audit log page */

let _allEvents = [];

async function loadAudit() {
  const limit = document.getElementById('filter-limit').value || 200;
  const tbody = document.getElementById('audit-tbody');
  tbody.innerHTML = '<tr><td colspan="9" style="text-align:center;padding:20px"><span class="spinner"></span> Loading…</td></tr>';

  try {
    const [data, stats] = await Promise.all([
      API.get(`/api/audit?limit=${limit}`),
      API.get('/api/audit/stats'),
    ]);
    _allEvents = data.events || [];
    renderStats(stats);
    filterTable();
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="9" class="text-muted" style="padding:20px">Error: ${e.message}</td></tr>`;
  }
}

function renderStats(stats) {
  document.getElementById('stat-total').textContent   = stats.total_identifications ?? 0;
  document.getElementById('stat-known').textContent   = stats.known_count ?? 0;
  document.getElementById('stat-unknown').textContent = stats.unknown_count ?? 0;
  document.getElementById('stat-known-pct').textContent = `${stats.known_pct ?? 0}% of total`;
  const top = (stats.top_identities || [])[0];
  document.getElementById('stat-top').textContent       = top ? top.identity : '—';
  document.getElementById('stat-top-count').textContent = top ? `${top.count} hits` : '—';
}

function filterTable() {
  const identity = document.getElementById('filter-identity').value.trim().toLowerCase();
  const type     = document.getElementById('filter-type').value;

  const filtered = _allEvents.filter(e => {
    if (type && e.event !== type) return false;
    if (identity && !(e.identity || '').toLowerCase().includes(identity)) return false;
    return true;
  });

  const tbody = document.getElementById('audit-tbody');
  if (filtered.length === 0) {
    tbody.innerHTML = '<tr><td colspan="9" class="text-muted" style="text-align:center;padding:20px">No events match filters.</td></tr>';
    document.getElementById('audit-count').textContent = '';
    return;
  }

  tbody.innerHTML = filtered.map((e, i) => {
    const known   = e.is_known;
    const wlHit   = e.watchlist_hit;
    const conf    = e.confidence != null ? (e.confidence * 100).toFixed(1) + '%' : '—';
    const ts      = (e.timestamp || '').slice(0, 19).replace('T', ' ');
    return `<tr>
      <td class="mono" style="color:var(--muted)">${filtered.length - i}</td>
      <td class="mono">${ts}</td>
      <td><span class="badge badge-gray">${e.event || '—'}</span></td>
      <td style="font-weight:${known ? '600' : '400'}">${e.identity || '—'}</td>
      <td class="mono">${conf}</td>
      <td>${known ? '<span class="badge badge-green">✓ Known</span>' : '<span class="badge badge-red">✗ Unknown</span>'}</td>
      <td>${wlHit ? '<span class="badge badge-red">🚨 HIT</span>' : '<span class="badge badge-gray">—</span>'}</td>
      <td><span class="badge badge-blue">${e.source || 'identify'}</span></td>
      <td class="mono" style="font-size:.73rem;max-width:140px;overflow:hidden;text-overflow:ellipsis">${e.filename || '—'}</td>
    </tr>`;
  }).join('');

  document.getElementById('audit-count').textContent =
    `Showing ${filtered.length} of ${_allEvents.length} events`;
}

document.addEventListener('DOMContentLoaded', loadAudit);
