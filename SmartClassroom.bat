@echo off
REM Navigate to the folder where this .bat file is located
cd /d "%~dp0"

echo ===================================================
echo       Smart Contactless Classroom System
echo ===================================================
echo Working directory: %cd%
echo.

REM Activate virtual environment if one exists
if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
) else if exist "smartclass\Scripts\activate.bat" (
    call "smartclass\Scripts\activate.bat"
)

REM Launch Application
echo Starting Application...
python main.py

echo.
echo Execution completed. Press any key to exit...
pause >nul
