@echo off
REM ============================================================================
REM DIGITAL TWIN BATTERY PIPELINE - STARTUP SCRIPT
REM ============================================================================

echo.
echo ╔════════════════════════════════════════════════════════════════════════╗
echo ║         Digital Twin Battery Pipeline - Setup & Startup               ║
echo ╚════════════════════════════════════════════════════════════════════════╝
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Python is not installed or not in PATH
    echo    Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo ✅ Python founds

echo ✅ Dependencies installed

REM Show menu
echo.
echo ╔════════════════════════════════════════════════════════════════════════╗
echo ║                        CHOOSE AN OPTION                               ║
echo ╠════════════════════════════════════════════════════════════════════════╣
echo ║                                                                        ║
echo ║  1. Run Full Pipeline (Train all models)                              ║
echo ║  2. Run Interactive Demos                                             ║
echo ║  3. Deploy REST API Server                                            ║
echo ║  4. BMS Real-time Simulation                                          ║
echo ║  5. Open Documentation                                                ║
echo ║  0. Exit                                                              ║
echo ║                                                                        ║
echo ╚════════════════════════════════════════════════════════════════════════╝
echo.

:menu
set /p choice="Enter your choice (0-5): "

if "%choice%"=="1" (
    echo.
    echo 🚀 Running full pipeline...
    echo.
    python digital_twin_pipeline.py
    echo.
    pause
    goto menu
)

if "%choice%"=="2" (
    echo.
    echo 🎮 Running interactive demos...
    echo.
    python demo.py
    echo.
    pause
    goto menu
)

if "%choice%"=="3" (
    echo.
    echo 🌐 Starting REST API server on http://localhost:8000
    echo.
    echo Press Ctrl+C to stop the server
    echo API Documentation: http://localhost:8000/docs
    echo.
    python fastapi_deployment.py
    echo.
    pause
    goto menu
)

if "%choice%"=="4" (
    echo.
    echo ⏱️  Running BMS simulation...
    echo.
    python bms_inference.py bms
    echo.
    pause
    goto menu
)

if "%choice%"=="5" (
    echo.
    echo 📖 Opening README...
    if exist "README.md" (
        start notepad README.md
    ) else (
        echo ❌ README.md not found
    )
    goto menu
)

if "%choice%"=="0" (
    echo.
    echo 👋 Goodbye!
    echo.
    exit /b 0
)

echo ❌ Invalid choice. Please try again.
goto menu
