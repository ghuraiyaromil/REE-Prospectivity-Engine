@echo off
title GeoAI -- Auto Training Watcher
color 0A
cd /d "C:\GeoAI_repo"
echo.
echo  ============================================================
echo   GeoAI Auto-Training Watcher
echo  ============================================================
echo.
echo  Watching: D:\GeoAI-INDIA\deposits\
echo.
echo  HOW TO ADD A NEW DEPOSIT:
echo    1. Create a folder:  D:\GeoAI-INDIA\deposits\deposit_name\
echo    2. Drop any data files into that folder
echo       (CSV, TIF, SHP, ZIP -- any format)
echo    3. Pipeline trains automatically within 30 seconds
echo    4. Model bundle syncs to Google Drive
echo.
echo  Keep this window open while working.
echo  Press Ctrl+C to stop.
echo.

:: Create deposits folder if it doesn't exist
mkdir "D:\GeoAI-INDIA\deposits" 2>nul

python watch_and_train.py
pause
