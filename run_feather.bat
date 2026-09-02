@echo off
REM ---------------------------------------------------------------------------
REM Launch the FEATHER app. Edit ENVNAME below if your conda env isn't "feather".
REM ---------------------------------------------------------------------------
set ENVNAME=feather

REM Run from this script's own folder so relative paths (stimuli, output) work.
cd /d "%~dp0"

REM Try to activate the conda environment, then start the app.
call conda activate %ENVNAME% 2>nul
if errorlevel 1 (
  echo Could not "conda activate %ENVNAME%".
  echo Make sure conda is on your PATH and the environment exists ^(see INSTALL.md^).
  pause
  exit /b 1
)

python "%~dp0playback_gui.py"
