@echo off
REM Go to the folder where this .bat file is located
cd /d "%~dp0"

REM (Optional) show where we are
echo Working directory: %cd%

REM Activate virtual environment (if it exists)
if exist "smartclass\Scripts\activate.bat" (
    call "smartclass\Scripts\activate.bat"
) else (
    echo [WARN] virtual environment 'smartclass' not found.
    echo Make sure Python and all required libraries are installed.
)

REM Run your main UI
python Ui.py

echo.
echo Finished. Press any key to exit...
pause >nul
