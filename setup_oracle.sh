#!/bin/bash
# =============================================================================
# setup_oracle.sh — One-command server setup for Oracle Cloud ARM (Ubuntu 22.04)
#
# Run this on your Oracle Cloud VM after SSH-ing in:
#   bash setup_oracle.sh
#
# What this does:
#   1. Installs Python 3.11, pip, git
#   2. Opens firewall ports 80 and 8000
#   3. Clones your GitHub repo
#   4. Installs all Python packages
#   5. Creates a .env file with your MongoDB URI
#   6. Creates a systemd service (auto-starts on reboot, auto-restarts on crash)
#   7. Starts the server
# =============================================================================

set -e  # Exit immediately if any command fails

echo "================================================"
echo " Face Recognition UAV — Oracle Cloud Setup"
echo "================================================"

# ── 1. System update and Python install ──────────────────────────────────────
echo "[1/7] Updating system and installing Python 3.11..."
sudo apt-get update -y
sudo apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    nginx

echo "[1/7] Done."

# ── 2. Open firewall ports ────────────────────────────────────────────────────
echo "[2/7] Opening firewall ports 80 and 8000..."
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8000 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save 2>/dev/null || true
echo "[2/7] Done."

# ── 3. Clone the GitHub repo ──────────────────────────────────────────────────
echo "[3/7] Cloning GitHub repository..."
cd /home/ubuntu
if [ -d "face-recognition-skewed-images" ]; then
    echo "Repo already exists. Pulling latest..."
    cd face-recognition-skewed-images
    git pull origin main
else
    git clone https://github.com/TENSEI3011/face-recognition-skewed-images.git
    cd face-recognition-skewed-images
fi
echo "[3/7] Done."

# ── 4. Create Python virtual environment and install packages ─────────────────
echo "[4/7] Creating virtual environment and installing packages..."
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
echo "[4/7] Done."

# ── 5. Create .env file ───────────────────────────────────────────────────────
echo "[5/7] Setting up environment variables..."
cat > .env << 'EOF'
MONGO_URI=mongodb+srv://facerecog:FACERECOG%222026@facerecognition.phw8q0k.mongodb.net/?retryWrites=true&w=majority&appName=FACERECOGNITION
MONGO_DB_NAME=facerecog_db
EOF
echo "[5/7] .env file created."

# ── 6. Pre-download InsightFace models ────────────────────────────────────────
echo "[6/7] Pre-downloading InsightFace models (this takes 2-3 minutes)..."
source venv/bin/activate
python3 -c "
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))
print('InsightFace buffalo_l models downloaded OK')
"
echo "[6/7] Done."

# ── 7. Create systemd service ─────────────────────────────────────────────────
echo "[7/7] Creating systemd auto-start service..."
APP_DIR="/home/ubuntu/face-recognition-skewed-images"
VENV_PYTHON="$APP_DIR/venv/bin/python"
VENV_UVICORN="$APP_DIR/venv/bin/uvicorn"

sudo tee /etc/systemd/system/facerecog.service > /dev/null << EOF
[Unit]
Description=Face Recognition UAV System (FastAPI)
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
ExecStart=$VENV_UVICORN web.backend.main:app --host 0.0.0.0 --port 8000 --workers 1
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable facerecog
sudo systemctl start facerecog

echo "[7/7] Systemd service created and started."

# ── Setup Nginx reverse proxy (port 80 → 8000) ────────────────────────────────
echo "Setting up Nginx reverse proxy..."
sudo tee /etc/nginx/sites-available/facerecog > /dev/null << 'EOF'
server {
    listen 80;
    server_name _;

    client_max_body_size 50M;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/facerecog /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable nginx

# ── Summary ───────────────────────────────────────────────────────────────────
PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "unknown")
echo ""
echo "================================================"
echo " SETUP COMPLETE!"
echo "================================================"
echo ""
echo " Your app is running at:"
echo " http://$PUBLIC_IP"
echo ""
echo " Useful commands:"
echo "   sudo systemctl status facerecog   # check server status"
echo "   sudo journalctl -u facerecog -f   # view live logs"
echo "   sudo systemctl restart facerecog  # restart server"
echo "   cd face-recognition-skewed-images"
echo "   git pull && sudo systemctl restart facerecog  # update app"
echo ""
echo " The server auto-restarts on crash and on VM reboot."
echo "================================================"
