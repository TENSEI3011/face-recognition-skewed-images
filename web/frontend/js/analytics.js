/* analytics.js — Analytics dashboard with HTML5 Canvas charts */

// ── Mini chart library (no dependencies) ──────────────────────────────────────

function drawBarChart(canvasId, labels, values, color = '#3b82f6', bgColor = '#f0f2f7') {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.offsetWidth || 600;
  const H = canvas.height || 200;
  canvas.width = W;

  ctx.clearRect(0, 0, W, H);
  const max = Math.max(...values, 1);
  const pad = { top: 16, right: 10, bottom: 36, left: 38 };
  const chartW = W - pad.left - pad.right;
  const chartH = H - pad.top - pad.bottom;
  const barW   = Math.max(2, chartW / labels.length - 3);

  // Grid lines
  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth   = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (chartH / 4) * i;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(W - pad.right, y); ctx.stroke();
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(Math.round(max - (max / 4) * i), pad.left - 4, y + 4);
  }

  // Bars
  values.forEach((v, i) => {
    const x = pad.left + i * (chartW / labels.length);
    const h = (v / max) * chartH;
    const y = pad.top + chartH - h;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect ? ctx.roundRect(x, y, barW, h, 3) : ctx.rect(x, y, barW, h);
    ctx.fill();
  });

  // X labels (show every Nth to avoid crowding)
  const step = Math.max(1, Math.floor(labels.length / 10));
  ctx.fillStyle = '#64748b';
  ctx.font = '9px Inter, sans-serif';
  ctx.textAlign = 'center';
  labels.forEach((lbl, i) => {
    if (i % step === 0) {
      const x = pad.left + i * (chartW / labels.length) + barW / 2;
      ctx.fillText(lbl.slice(5), x, H - 4); // show MM-DD
    }
  });
}

function drawHourlyChart(canvasId, hourly) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.offsetWidth || 800;
  const H = 90;
  canvas.width = W;
  canvas.height = H;

  const max = Math.max(...hourly.map(h => h.count), 1);
  const cellW = W / 24;
  const cellH = 40;
  const top   = 14;

  ctx.font = '10px Inter, sans-serif';
  ctx.textAlign = 'center';

  hourly.forEach(({ hour, count }) => {
    const x = hour * cellW;
    const alpha = count / max;
    ctx.fillStyle = `rgba(59,130,246,${0.08 + alpha * 0.85})`;
    ctx.fillRect(x + 2, top, cellW - 4, cellH);

    if (count > 0) {
      ctx.fillStyle = alpha > 0.5 ? 'white' : '#1e293b';
      ctx.font = '9px Inter, sans-serif';
      ctx.fillText(count, x + cellW / 2, top + cellH / 2 + 4);
    }

    // Hour label
    ctx.fillStyle = '#64748b';
    ctx.font = '9px Inter, sans-serif';
    ctx.fillText(`${hour}h`, x + cellW / 2, top + cellH + 14);
  });
}

function drawIdentityChart(canvasId, identities) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.offsetWidth || 400;
  const H = 200;
  canvas.width = W;
  const top5 = identities.slice(0, 8);
  const max  = Math.max(...top5.map(i => i.total), 1);
  const pad  = { top: 16, right: 10, bottom: 40, left: 90 };
  const chartW = W - pad.left - pad.right;
  const chartH = H - pad.top - pad.bottom;
  const barH   = Math.max(2, chartH / top5.length - 6);
  const colors = ['#3b82f6','#8b5cf6','#ec4899','#f59e0b','#10b981','#06b6d4','#ef4444','#84cc16'];

  ctx.clearRect(0, 0, W, H);

  top5.forEach((item, i) => {
    const y = pad.top + i * (chartH / top5.length);
    const w = (item.total / max) * chartW;
    ctx.fillStyle = colors[i % colors.length];
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(pad.left, y, w, barH, 3);
    else ctx.rect(pad.left, y, w, barH);
    ctx.fill();

    // Label
    ctx.fillStyle = '#1e293b';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(item.identity, pad.left - 4, y + barH / 2 + 4);

    // Count
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'left';
    ctx.fillText(item.total, pad.left + w + 4, y + barH / 2 + 4);
  });
}

// ── Data loading ───────────────────────────────────────────────────────────────

async function loadSummary() {
  const data = await API.get('/api/analytics/summary');
  document.getElementById('kpi-total').textContent     = data.total_identifications ?? 0;
  document.getElementById('kpi-known-pct').textContent = `${data.known_pct ?? 0}%`;
  document.getElementById('kpi-known-sub').textContent = `${data.known_count ?? 0} of ${data.total_identifications ?? 0} known`;
  document.getElementById('kpi-unknown').textContent   = data.unknown_count ?? 0;
  const top = (data.top_identities || [])[0];
  document.getElementById('kpi-top').textContent       = top ? top.identity : '—';
  document.getElementById('kpi-top-count').textContent = top ? `${top.count} detections` : '—';
}

async function loadDaily() {
  const data = await API.get('/api/analytics/daily');
  const daily = data.daily || [];
  drawBarChart('daily-chart', daily.map(d => d.date), daily.map(d => d.count));
}

async function loadIdentities() {
  const data = await API.get('/api/analytics/identities');
  const identities = data.identities || [];
  drawIdentityChart('identity-chart', identities);

  const tbody = document.getElementById('identity-tbody');
  if (!identities.length) {
    tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;padding:20px" class="text-muted">No data yet.</td></tr>';
    return;
  }
  const maxCount = Math.max(...identities.map(i => i.total), 1);
  tbody.innerHTML = identities.map((item, idx) => {
    const matchRate = item.total ? ((item.known / item.total) * 100).toFixed(0) : 0;
    const barW = Math.round((item.total / maxCount) * 100);
    return `<tr>
      <td style="font-weight:700;color:var(--muted)">#${idx+1}</td>
      <td style="font-weight:600">${item.identity}</td>
      <td>${item.total}</td>
      <td><span class="badge badge-green">${item.known}</span></td>
      <td>${matchRate}%</td>
      <td style="width:120px">
        <div class="progress"><div class="progress-bar" style="width:${barW}%"></div></div>
      </td>
    </tr>`;
  }).join('');
}

async function loadHourly() {
  const data = await API.get('/api/analytics/hourly');
  drawHourlyChart('hourly-chart', data.hourly || []);
}

async function loadAll() {
  try {
    await Promise.all([loadSummary(), loadDaily(), loadIdentities(), loadHourly()]);
  } catch (e) {
    showAlert('alert-zone', 'Error loading analytics: ' + e.message, 'warning');
  }
}

document.addEventListener('DOMContentLoaded', () => {
  loadAll();
  // Redraw on resize
  window.addEventListener('resize', () => { loadDaily(); loadIdentities(); loadHourly(); });
});
