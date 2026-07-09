/* login.js — Login page logic */

const form     = document.getElementById('login-form');
const errBox   = document.getElementById('login-error');
const loginBtn = document.getElementById('login-btn');

// If already logged in, skip to dashboard
if (localStorage.getItem('auth_token')) {
  window.location.href = '/';
}

function showError(msg) {
  errBox.textContent = msg;
  errBox.style.display = 'block';
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  errBox.style.display = 'none';
  loginBtn.disabled = true;
  loginBtn.textContent = 'Signing in…';

  const username = document.getElementById('username').value.trim();
  const password = document.getElementById('password').value;

  try {
    const res = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password }),
    });
    const data = await res.json();
    if (!res.ok) {
      showError(data.detail || 'Login failed.');
      return;
    }
    localStorage.setItem('auth_token', data.token);
    localStorage.setItem('auth_user', data.username);
    window.location.href = '/';
  } catch (err) {
    showError('Network error. Make sure the server is running.');
  } finally {
    loginBtn.disabled = false;
    loginBtn.textContent = 'Sign In';
  }
});
