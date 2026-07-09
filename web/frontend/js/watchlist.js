/* watchlist.js — Watchlist & Alert management page */

async function loadWatchlist() {
  try {
    const data = await API.get('/api/watchlist');
    const container = document.getElementById('watchlist-content');
    const names = data.watchlist || [];
    if (names.length === 0) {
      container.innerHTML = '<p class="text-sm text-muted">No identities on watchlist.</p>';
      return;
    }
    container.innerHTML = names.map(name => `
      <div style="display:flex;align-items:center;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--border)">
        <span style="font-weight:500;font-size:.88rem">🎯 ${name}</span>
        <button class="btn btn-danger btn-sm" onclick="removeFromWatchlist('${name}')">Remove</button>
      </div>
    `).join('');
  } catch (e) {
    document.getElementById('watchlist-content').innerHTML =
      `<p class="text-sm text-muted">Error: ${e.message}</p>`;
  }
}

async function loadHits() {
  const tbody = document.getElementById('hits-tbody');
  try {
    const data = await API.get('/api/watchlist/hits?limit=50');
    const hits = data.hits || [];
    if (hits.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" class="text-muted" style="text-align:center;padding:20px">No watchlist hits recorded yet.</td></tr>';
      return;
    }
    tbody.innerHTML = hits.map(h => `
      <tr>
        <td class="mono">${(h.timestamp || '').slice(0,19).replace('T',' ')}</td>
        <td><span style="font-weight:600;color:#dc2626">${h.identity || '—'}</span></td>
        <td>${h.confidence != null ? (h.confidence * 100).toFixed(1) + '%' : '—'}</td>
        <td><span class="badge badge-blue">${h.source || 'identify'}</span></td>
        <td class="mono" style="font-size:.75rem">${h.filename || '—'}</td>
      </tr>
    `).join('');
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="5" class="text-muted" style="padding:20px">Error: ${e.message}</td></tr>`;
  }
}

async function loadGalleryNames() {
  try {
    const data = await API.get('/api/gallery');
    const dl = document.getElementById('gallery-names');
    (data.identities || []).forEach(id => {
      const opt = document.createElement('option');
      opt.value = id.name;
      dl.appendChild(opt);
    });
  } catch {}
}

async function addToWatchlist() {
  const name = document.getElementById('wl-name').value.trim();
  if (!name) { showAlert('alert-zone', 'Please enter an identity name.', 'warning'); return; }
  try {
    const data = await API.post('/api/watchlist', { name });
    showAlert('alert-zone', data.message, 'success');
    document.getElementById('wl-name').value = '';
    loadWatchlist();
  } catch (e) {
    showAlert('alert-zone', e.message, 'danger');
  }
}

async function removeFromWatchlist(name) {
  try {
    const data = await API.delete(`/api/watchlist/${encodeURIComponent(name)}`);
    showAlert('alert-zone', data.message, 'success');
    loadWatchlist();
  } catch (e) {
    showAlert('alert-zone', e.message, 'danger');
  }
}

// Init
document.addEventListener('DOMContentLoaded', () => {
  loadWatchlist();
  loadHits();
  loadGalleryNames();
});
