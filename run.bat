@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion
cd /d "%~dp0"
title PG-Video-Transcriber

:: ============================================================
:: PG-Video-Transcriber — Windows: auto-install and launch
:: ============================================================

:: --- 1. Check Python ---
where python >nul 2>&1
if errorlevel 1 goto :no_python

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do set PYMAJOR=%%a& set PYMINOR=%%b

if !PYMAJOR! LSS 3 goto :old_python
if !PYMAJOR!==3 if !PYMINOR! LSS 10 goto :old_python

echo Python %PYVER% - OK

:: --- 2. Create venv if missing ---
if exist "venv\Scripts\activate.bat" goto :activate_venv

echo.
echo [1/4] Creating virtual environment...
python -m venv venv
if errorlevel 1 goto :venv_error
echo Virtual environment created.

:activate_venv
call venv\Scripts\activate.bat

:: --- 3. Check if packages need installation ---
set "MARKER=venv\.installed"

if not exist "%MARKER%" goto :install_packages

:: Compare requirements.txt hash
for /f "usebackq" %%h in (`certutil -hashfile requirements.txt MD5 ^| findstr /v "hash MD5"`) do set REQHASH=%%h
set /p OLDHASH=<"%MARKER%"
if not "!REQHASH!"=="!OLDHASH!" goto :install_packages

goto :launch

:: --- 4. Install packages ---
:install_packages
echo.
echo [2/4] Installing PyTorch with CUDA support...
echo This may take several minutes on first install.
echo.

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo WARNING: CUDA PyTorch failed. Installing CPU version...
    pip install torch torchaudio
)

echo.
echo [3/4] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 goto :deps_error

:: Safety check: speechbrain may overwrite torch with CPU version
python -c "import torch; assert torch.cuda.is_available()" >nul 2>&1
if errorlevel 1 (
    echo [3.5/4] Restoring CUDA PyTorch...
    pip install torch torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/cu124 >nul 2>&1
)

:: Save requirements.txt hash
for /f "usebackq" %%h in (`certutil -hashfile requirements.txt MD5 ^| findstr /v "hash MD5"`) do echo %%h> "%MARKER%"

echo.
echo Installation complete!

:: --- 5. Launch ---
:launch
echo.
echo [4/4] Starting PG-Video-Transcriber...
echo.
python main.py

if errorlevel 1 goto :app_error
goto :eof

:: ============================================================
:: Error handlers
:: ============================================================

:no_python
echo ERROR: Python not found.
echo Download Python 3.10+ from https://www.python.org/downloads/
echo IMPORTANT: check "Add Python to PATH" during installation.
pause
exit /b 1

:old_python
echo ERROR: Python 3.10+ required, found: %PYVER%
pause
exit /b 1

:venv_error
echo ERROR: Failed to create virtual environment.
pause
exit /b 1

:deps_error
echo ERROR: Failed to install dependencies.
pause
exit /b 1

:app_error
echo.
echo ERROR: Application exited with an error.
echo Logs: transcriber.log
pause
exit /b 1
