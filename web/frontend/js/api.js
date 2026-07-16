/* ============================================================
   api.js — Centralized fetch wrapper
   All API calls go through these helpers for consistent error handling.
   Enhanced with: auth headers, 401 redirect, watchlist alert polling.
   ============================================================ */

const API_BASE = '';  // Same origin — FastAPI serves both frontend and API

// ── Auth helpers ───────────────────────────────────────────────────────────────

function getToken() {
  return localStorage.getItem('auth_token') || '';
}

function getAuthHeaders() {
  const token = getToken();
  return token ? { 'Authorization': `Bearer ${token}` } : {};
}

function logout() {
  localStorage.removeItem('auth_token');
  localStorage.removeItem('auth_user');
  window.location.href = '/login';
}

// ── Fetch wrapper — with timeout ———————————————————————————————————————————————

function _fetchWithTimeout(url, options = {}, ms = 8000) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), ms);
  return fetch(url, { ...options, signal: controller.signal })
    .finally(() => clearTimeout(id));
}

const API = {
  async get(path) {
    const res = await _fetchWithTimeout(API_BASE + path, {
      headers: { ...getAuthHeaders() },
    });
    if (res.status === 401) { logout(); return; }
    if (!res.ok) throw new Error(`GET ${path} → ${res.status}: ${await res.text()}`);
    return res.json();
  },

  async post(path, body) {
    const isForm = body instanceof FormData;
    const res = await fetch(API_BASE + path, {
      method: 'POST',
      headers: isForm ? { ...getAuthHeaders() } : { 'Content-Type': 'application/json', ...getAuthHeaders() },
      body: isForm ? body : JSON.stringify(body),
    });
    if (res.status === 401) { logout(); return; }
    if (!res.ok) {
      const txt = await res.text();
      let detail = txt;
      try { detail = JSON.parse(txt).detail || txt; } catch {}
      throw new Error(detail);
    }
    return res.json();
  },

  async delete(path) {
    const res = await fetch(API_BASE + path, {
      method: 'DELETE',
      headers: { ...getAuthHeaders() },
    });
    if (res.status === 401) { logout(); return; }
    if (!res.ok) throw new Error(`DELETE ${path} → ${res.status}`);
    return res.json();
  },

  // Upload a file with FormData — NO timeout (video processing can take many minutes)
  async upload(path, formData) {
    const res = await fetch(API_BASE + path, {
      method: 'POST',
      headers: { ...getAuthHeaders() },
      body: formData,
      // No AbortController — let the server take as long as it needs
    });
    if (res.status === 401) { logout(); return; }
    if (!res.ok) {
      const txt = await res.text();
      let detail = txt;
      try { detail = JSON.parse(txt).detail || txt; } catch {}
      throw new Error(detail);
    }
    return res.json();
  }
};

// ── Shared UI helpers ──────────────────────────────────────────────────────────

function showAlert(container, msg, type = 'info') {
  const el = document.getElementById(container);
  if (!el) return;
  el.innerHTML = `<div class="alert alert-${type}">${msg}</div>`;
  el.classList.remove('hidden');
}

function clearAlert(container) {
  const el = document.getElementById(container);
  if (el) el.innerHTML = '';
}

function setLoading(btnId, loading, text = 'Processing…') {
  const btn = document.getElementById(btnId);
  if (!btn) return;
  btn.disabled = loading;
  btn.dataset.origText = btn.dataset.origText || btn.innerHTML;
  btn.innerHTML = loading
    ? `<span class="spinner"></span> ${text}`
    : btn.dataset.origText;
}

function formatPct(val) {
  return (val * 100).toFixed(1) + '%';
}

function formatNum(val, decimals = 4) {
  return typeof val === 'number' ? val.toFixed(decimals) : val;
}

// ── Pipeline status bar (shared across pages) ─────────────────────────────────

async function updateStatusBar() {
  try {
    const data = await API.get('/api/status');
    const dot  = document.getElementById('status-dot');
    const txt  = document.getElementById('status-text');
    if (!dot || !txt) return;

    if (data && data.loaded) {
      dot.className = 'status-dot loaded';
      txt.textContent = `Pipeline ready · ${data.gallery_count} identities`;
    } else if (data && data.error) {
      dot.className = 'status-dot';
      txt.textContent = 'Not loaded';
    } else {
      dot.className = 'status-dot loading';
      txt.textContent = 'Loading…';
    }
  } catch (e) {
    const dot = document.getElementById('status-dot');
    const txt = document.getElementById('status-text');
    if (dot) dot.className = 'status-dot';
    if (txt) txt.textContent = 'Server offline';
  }
}

// ── Watchlist alert banner ─────────────────────────────────────────────────────
// Shows a persistent banner if a watchlist hit was detected recently (last 60s)

let _lastWatchlistHitTs = null;

function _showWatchlistBanner(hit) {
  let banner = document.getElementById('watchlist-alert-banner');
  if (!banner) {
    banner = document.createElement('div');
    banner.id = 'watchlist-alert-banner';
    banner.style.cssText = `
      position:fixed;top:0;left:0;right:0;z-index:9999;
      background:#dc2626;color:white;
      padding:10px 20px;
      display:flex;align-items:center;justify-content:space-between;
      font-size:.88rem;font-weight:600;
      box-shadow:0 2px 8px rgba(0,0,0,.3);
    `;
    document.body.appendChild(banner);
  }
  banner.innerHTML = `
    <span>🚨 WATCHLIST ALERT: <strong>${hit.identity}</strong> detected at ${hit.timestamp ? hit.timestamp.slice(11,19) : 'unknown time'} (confidence: ${((hit.confidence||0)*100).toFixed(1)}%)</span>
    <button onclick="this.parentElement.remove()" style="background:none;border:1px solid rgba(255,255,255,.5);color:white;padding:2px 10px;border-radius:4px;cursor:pointer;font-size:.8rem">Dismiss</button>
  `;

  // Browser notification
  if ('Notification' in window && Notification.permission === 'granted') {
    new Notification('⚠️ Watchlist Alert', {
      body: `${hit.identity} detected (${((hit.confidence||0)*100).toFixed(1)}% confidence)`,
    });
  }
}

async function pollWatchlistHits() {
  try {
    const data = await API.get('/api/watchlist/hits?limit=1');
    if (!data || !data.hits || data.hits.length === 0) return;
    const latest = data.hits[0];
    const ts = latest.timestamp;
    if (!ts || ts === _lastWatchlistHitTs) return;
    // Only show if < 60 seconds old
    const age = (Date.now() - new Date(ts + 'Z').getTime()) / 1000;
    if (age < 60) {
      _lastWatchlistHitTs = ts;
      _showWatchlistBanner(latest);
    }
  } catch {}
}

// Request notification permission
if ('Notification' in window && Notification.permission === 'default') {
  Notification.requestPermission();
}

// ── Sidebar user display ───────────────────────────────────────────────────────

function _renderUserBadge() {
  const user = localStorage.getItem('auth_user');
  const footer = document.querySelector('.sidebar-footer');
  if (!footer || !user) return;
  footer.innerHTML = `
    HOG · LBP · Geometry · ArcFace<br>MTCNN → PCA → SVM<br>
    <span style="color:#93c5fd;font-size:.7rem">👤 ${user}</span>
    <button onclick="window.logout()" style="background:none;border:none;color:#f87171;font-size:.7rem;cursor:pointer;margin-left:6px">Sign out</button>
  `;
}

window.logout = function() {
  localStorage.removeItem('auth_token');
  localStorage.removeItem('auth_user');
  window.location.href = '/login';
};

// Mark active nav link
document.addEventListener('DOMContentLoaded', () => {
  const path = window.location.pathname.replace(/\/$/, '') || '/';
  document.querySelectorAll('.nav-link').forEach(a => {
    const href = a.getAttribute('href').replace(/\/$/, '') || '/';
    if (href === path) a.classList.add('active');
  });
  _renderUserBadge();

  // Delay status bar — don’t block first paint
  setTimeout(updateStatusBar, 200);
  setInterval(updateStatusBar, 10000);

  // Delay watchlist poll — not critical for initial page load
  setTimeout(() => {
    pollWatchlistHits();
    setInterval(pollWatchlistHits, 15000);
  }, 5000);
});
