@echo off
rem ======================================================================
rem  PatientTriage.ai - double-click launcher for the browser console.
rem
rem  Double-click this file, or run it from a terminal with the same
rem  options the module takes:
rem
rem      PatientTriage.bat --profile urban --port 8010
rem
rem  First run creates .venv and installs dependencies (a few minutes).
rem  Every run after that starts in seconds.
rem ======================================================================

setlocal EnableExtensions
title PatientTriage.ai
cd /d "%~dp0"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"

echo.
echo   PatientTriage.ai - emergency department triage decision support
echo   ---------------------------------------------------------------
echo.

if exist "%VENV_PY%" goto :check_deps


rem --- no environment yet: find a Python 3.12 to build one with ---------
rem LightGBM has no wheel for 3.13, so 3.12 is the version that works.

echo   No environment found. Creating .venv - this happens once.
echo.

set "BOOT_PY="

py -3.12 -c "import sys" >nul 2>&1
if not errorlevel 1 set "BOOT_PY=py -3.12"
if defined BOOT_PY goto :make_venv

python -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,12) else 1)" >nul 2>&1
if not errorlevel 1 set "BOOT_PY=python"
if defined BOOT_PY goto :make_venv

goto :no_python

:make_venv
%BOOT_PY% -m venv ".venv"
if errorlevel 1 goto :venv_failed


rem --- dependencies ----------------------------------------------------

:check_deps
"%VENV_PY%" -c "import fastapi, uvicorn, patienttriage" >nul 2>&1
if not errorlevel 1 goto :launch

echo   Installing dependencies into .venv ...
echo.
"%VENV_PY%" -m pip install --upgrade pip >nul 2>&1
"%VENV_PY%" -m pip install -e ".[ui]"
if errorlevel 1 goto :install_failed
echo.


rem --- run -------------------------------------------------------------

:launch
echo   Starting the triage console. Your browser will open shortly.
echo   Close this window, or press Ctrl+C, to stop it.
echo.
"%VENV_PY%" -m patienttriage.ui.server %*
if errorlevel 1 goto :run_failed
goto :done


rem --- failures, each said plainly -------------------------------------

:no_python
echo   [!] Python 3.12 was not found on this machine.
echo.
echo       Install it from https://www.python.org/downloads/release/python-3129/
echo       and tick "Add python.exe to PATH" in the installer, then run this
echo       file again. 3.13 will not work: LightGBM has no wheel for it yet.
goto :halt

:venv_failed
echo.
echo   [!] Could not create the .venv virtual environment.
echo       Check that you have write permission in this folder.
goto :halt

:install_failed
echo.
echo   [!] Dependency installation failed. The pip output above says why -
echo       usually no internet connection, or a proxy blocking PyPI.
goto :halt

:run_failed
echo.
echo   [!] The console stopped with an error. The traceback above says why.
echo       A common one is the port already being in use - try:
echo           PatientTriage.bat --port 8010
goto :halt

:halt
echo.
pause
exit /b 1

:done
echo.
echo   Console stopped.
timeout /t 3 >nul
exit /b 0
