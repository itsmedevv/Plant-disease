@echo off
title Plant Disease Detection for Farmers
echo Starting Plant Disease Detection Web App...
cd /d "%~dp0"
call .\venv\Scripts\streamlit.exe run app.py
pause

