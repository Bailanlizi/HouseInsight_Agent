@echo off
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat (
  call .venv\Scripts\activate.bat
)
uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
