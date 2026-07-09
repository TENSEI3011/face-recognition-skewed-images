@echo off
title Face Recognition UAV System — Web Server

echo.
echo  ====================================================
echo   Face Recognition UAV System — Web Interface
echo  ====================================================
echo.

:: Change to project root (two levels up from web/)
cd /d "%~dp0.."

:: Install web requirements if needed
echo [1/2] Checking web dependencies...
pip install -q fastapi uvicorn[standard] python-multipart aiofiles websockets

echo [2/2] Starting server...
echo.
echo  Dashboard:  http://localhost:8000
echo  API Docs:   http://localhost:8000/docs
echo  Gallery:    http://localhost:8000/gallery
echo  Live Demo:  http://localhost:8000/demo
echo.
echo  Press Ctrl+C to stop.
echo.

:: Run from project root so `src` imports resolve correctly
python -m uvicorn web.backend.main:app --host 0.0.0.0 --port 8000 --reload

pause
