// gallery.js

let currentIdentity = null;

async function loadGallery() {
  try {
    const data = await API.get('/api/gallery');
    const grid = document.getElementById('identity-grid');
    const count = document.getElementById('gallery-count');
    const identities = data.identities || [];

    count.textContent = `${identities.length} ${identities.length === 1 ? 'identity' : 'identities'}`;

    if (identities.length === 0) {
      grid.innerHTML = `
        <div style="grid-column:1/-1;text-align:center;padding:40px;color:var(--muted)">
          <div style="font-size:2rem;margin-bottom:12px">👤</div>
          <div style="font-weight:500;margin-bottom:6px">No identities enrolled</div>
          <div style="font-size:.8rem">Click "Enroll New Identity" to add faces to the gallery.</div>
        </div>`;
      return;
    }

    grid.innerHTML = identities.map(id => `
      <div class="identity-card" onclick="openIdentity('${id.name}')">
        <div class="identity-thumb-placeholder" id="thumb-${id.name}">👤</div>
        <div class="identity-info">
          <div class="identity-name">${id.name.replace(/_/g, ' ')}</div>
          <div class="identity-count">${id.image_count} image${id.image_count !== 1 ? 's' : ''}</div>
          <div class="identity-actions">
            <button class="btn btn-secondary btn-sm" onclick="event.stopPropagation();openIdentity('${id.name}')">View</button>
            <button class="btn btn-danger btn-sm" onclick="event.stopPropagation();deleteIdentity('${id.name}')">Delete</button>
          </div>
        </div>
      </div>
    `).join('');

    // Lazy-load first image thumbnails
    identities.forEach(id => loadThumb(id.name));

  } catch (e) {
    showAlert('alert-zone', `Failed to load gallery: ${e.message}`, 'danger');
  }
}

async function loadThumb(name) {
  try {
    const data = await API.get(`/api/gallery/${name}/images?max_images=1`);
    if (data.images && data.images.length > 0) {
      const el = document.getElementById(`thumb-${name}`);
      if (el) {
        el.outerHTML = `<img class="identity-thumb" src="${data.images[0].data}" alt="${name}" onclick="openIdentity('${name}')">`;
      }
    }
  } catch {}
}

async function openIdentity(name) {
  currentIdentity = name;
  document.getElementById('modal-title').textContent = name.replace(/_/g, ' ');
  document.getElementById('modal-images').innerHTML = '<div class="text-sm text-muted">Loading images…</div>';
  document.getElementById('identity-modal').classList.remove('hidden');
  document.getElementById('identity-modal').style.display = 'flex';

  document.getElementById('modal-delete-btn').onclick = () => deleteIdentity(name, true);

  try {
    const data = await API.get(`/api/gallery/${name}/images?max_images=12`);
    const imgs = data.images || [];
    if (imgs.length === 0) {
      document.getElementById('modal-images').innerHTML = '<p class="text-muted text-sm">No images found.</p>';
      return;
    }
    document.getElementById('modal-images').innerHTML = imgs.map(img => `
      <div style="border-radius:6px;overflow:hidden;border:1px solid var(--border)">
        <img src="${img.data}" style="width:100%;height:110px;object-fit:cover;display:block">
        <div style="padding:4px 6px;font-size:.65rem;color:var(--muted);background:var(--bg)">${img.filename}</div>
      </div>
    `).join('');
  } catch (e) {
    document.getElementById('modal-images').innerHTML = `<p class="text-danger text-sm">${e.message}</p>`;
  }
}

function closeModal() {
  document.getElementById('identity-modal').style.display = 'none';
  document.getElementById('identity-modal').classList.add('hidden');
}

async function deleteIdentity(name, fromModal = false) {
  if (!confirm(`Delete identity "${name.replace(/_/g, ' ')}" and all its images? This cannot be undone.`)) return;
  try {
    await API.delete(`/api/gallery/${name}`);
    if (fromModal) closeModal();
    showAlert('alert-zone', `Identity "${name}" deleted successfully.`, 'success');
    loadGallery();
  } catch (e) {
    showAlert('alert-zone', `Delete failed: ${e.message}`, 'danger');
  }
}

// Upload
const fileInput = document.getElementById('file-input');
const uploadZone = document.getElementById('upload-zone');

const VIDEO_EXTS = new Set(['.mp4', '.avi', '.mov', '.mkv', '.webm']);

function hasVideoFile(fileList) {
  return Array.from(fileList).some(f => VIDEO_EXTS.has(f.name.slice(f.name.lastIndexOf('.')).toLowerCase()));
}

fileInput.addEventListener('change', () => {
  const files = fileInput.files;
  const n = files.length;
  document.getElementById('selected-files').textContent =
    n > 0 ? `${n} file${n > 1 ? 's' : ''} selected` : '';

  const videoOpts = document.getElementById('video-options');
  const icon = document.getElementById('upload-icon');
  if (hasVideoFile(files)) {
    videoOpts.classList.remove('hidden');
    icon.textContent = '🎬';
  } else {
    videoOpts.classList.add('hidden');
    icon.textContent = '📁';
  }
});

uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  fileInput.files = e.dataTransfer.files;
  const files = fileInput.files;
  const n = files.length;
  document.getElementById('selected-files').textContent = n > 0 ? `${n} file${n > 1 ? 's' : ''} selected` : '';

  const videoOpts = document.getElementById('video-options');
  const icon = document.getElementById('upload-icon');
  if (hasVideoFile(files)) {
    videoOpts.classList.remove('hidden');
    icon.textContent = '🎬';
  } else {
    videoOpts.classList.add('hidden');
    icon.textContent = '📁';
  }
});

async function uploadIdentity() {
  const name  = document.getElementById('identity-name').value.trim();
  const files = fileInput.files;

  if (!name) { showAlert('alert-zone', 'Please enter an identity name.', 'warning'); return; }
  if (!files.length) { showAlert('alert-zone', 'Please select at least one image or video.', 'warning'); return; }

  setLoading('upload-btn', true, 'Uploading…');
  const fd = new FormData();
  fd.append('name', name);
  for (const f of files) fd.append('files', f);

  // Include frame interval if a video is present
  if (hasVideoFile(files)) {
    const interval = document.getElementById('frame-interval').value;
    fd.append('frame_interval', interval);
  }

  try {
    const res = await API.upload('/api/gallery/upload', fd);
    const errMsg = res.errors && res.errors.length ? ' Errors: ' + res.errors.join(', ') : '';
    showAlert('alert-zone', `${res.message}${errMsg}`, 'success');
    document.getElementById('upload-panel').classList.add('hidden');
    document.getElementById('identity-name').value = '';
    fileInput.value = '';
    document.getElementById('selected-files').textContent = '';
    document.getElementById('video-options').classList.add('hidden');
    document.getElementById('upload-icon').textContent = '📁';
    loadGallery();
  } catch (e) {
    showAlert('alert-zone', `Upload failed: ${e.message}`, 'danger');
  } finally {
    setLoading('upload-btn', false);
  }
}

document.addEventListener('DOMContentLoaded', loadGallery);
