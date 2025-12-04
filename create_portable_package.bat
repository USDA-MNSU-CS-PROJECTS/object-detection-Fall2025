@echo off
echo ========================================
echo   Creating Portable Package
echo ========================================
echo.

echo This will create a folder your clients can run WITHOUT installing Python!
echo Package size will be approximately 1-1.5 GB
echo This process takes 10-15 minutes
echo.
pause

echo Step 1: Creating clean Python environment...
call conda create -y -n alfalfa_portable python=3.9
call conda activate alfalfa_portable

echo.
echo Step 2: Installing dependencies (CPU-only for smaller size)...
pip install gradio==3.50.2
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics pillow pandas numpy nd2reader nd2 shapely scikit-image tifffile

echo.
echo Step 3: Creating portable package folder structure...
if exist portable_package rmdir /s /q portable_package
mkdir portable_package
mkdir portable_package\app
mkdir portable_package\app\api
mkdir portable_package\models

echo.
echo Step 4: Copying application files...
copy src\app\main.py portable_package\app\
copy src\app\api\*.py portable_package\app\api\
if exist sample_trained_models\best.pt (
    copy sample_trained_models\best.pt portable_package\models\
) else (
    echo WARNING: best.pt not found in sample_trained_models\
    echo You'll need to copy it manually later!
    pause
)

echo.
echo Step 5: Copying Python environment (this takes 5-10 minutes)...
echo Copying approximately 1GB of files...
xcopy /E /I /H /Y "%CONDA_PREFIX%" portable_package\python_env

echo.
echo Step 6: Creating launcher script...
(
echo @echo off
echo title Alfalfa Stem Detection Tool
echo color 0A
echo cls
echo.
echo  ========================================
echo    ALFALFA STEM DETECTION TOOL
echo  ========================================
echo.
echo  Starting the application...
echo  Your web browser will open automatically in 10-15 seconds.
echo.
echo  IMPORTANT NOTES:
echo  - Keep this window OPEN while using the tool
echo  - First startup may take 15-20 seconds
echo  - The browser will open to: http://127.0.0.1:7860
echo  - Close this window when you're finished
echo.
echo  ========================================
echo.
echo.
echo cd /d "%%~dp0"
echo set PYTHONPATH=%%CD%%\app
echo.
echo REM Check if model exists
echo if not exist "models\best.pt" ^(
echo     echo ERROR: Model file not found!
echo     echo Expected location: models\best.pt
echo     echo.
echo     pause
echo     exit /b 1
echo ^)
echo.
echo REM Update the model path in main.py on-the-fly
echo set MODEL_PATH=%%CD%%\models\best.pt
echo.
echo REM Start the application
echo python_env\python.exe app\main.py
echo.
echo if errorlevel 1 ^(
echo     echo.
echo     echo ========================================
echo     echo ERROR: Application failed to start
echo     echo ========================================
echo     echo.
echo     echo Check the error messages above.
echo     echo.
echo     pause
echo     exit /b 1
echo ^)
echo.
echo echo.
echo echo  ========================================
echo echo  Application closed successfully
echo echo  ========================================
echo echo.
echo pause
) > portable_package\START_TOOL.bat

echo.
echo Step 7: Creating user README...
(
echo ========================================
echo   ALFALFA STEM DETECTION TOOL - v1.0
echo ========================================
echo.
echo QUICK START GUIDE ^(3 EASY STEPS^):
echo.
echo 1. Unzip this entire folder to your Desktop
echo    ^(or anywhere on your computer^)
echo.
echo 2. Double-click "START_TOOL.bat"
echo.
echo 3. Your web browser will open automatically
echo    - Upload your .nd2, .png, or .zip files
echo    - Click "Run Analysis" or "Convert"
echo    - Download your results!
echo.
echo ========================================
echo IMPORTANT NOTES
echo ========================================
echo.
echo - NO PYTHON INSTALLATION REQUIRED!
echo   Everything is included in this folder.
echo.
echo - Keep the black window OPEN while using the tool
echo   ^(This is normal - it shows the app is running^)
echo.
echo - First startup takes 15-20 seconds
echo   ^(Subsequent startups are faster^)
echo.
echo - The tool opens in your web browser at:
echo   http://127.0.0.1:7860
echo   ^(This is your local computer, not the internet^)
echo.
echo - You can process multiple files at once
echo   ^(Just select them all when uploading^)
echo.
echo ========================================
echo SYSTEM REQUIREMENTS
echo ========================================
echo.
echo - Windows 10 or later
echo - 8GB RAM minimum ^(16GB recommended^)
echo - 2GB free disk space
echo - No special graphics card needed
echo.
echo ========================================
echo TWO ANALYSIS MODES
echo ========================================
echo.
echo TAB 1 - Full Detection Pipeline:
echo   Analyzes images and detects alfalfa stem structures
echo   Results: CSV with measurements + annotated images
echo.
echo TAB 2 - Simple Image Converter:
echo   Converts ND2 files to PNG format only
echo   Results: ZIP file with converted images
echo.
echo ========================================
echo TROUBLESHOOTING
echo ========================================
echo.
echo Problem: Browser doesn't open automatically
echo Solution: Manually open your browser and go to:
echo           http://127.0.0.1:7860
echo.
echo Problem: "Model file not found" error
echo Solution: Make sure the "models" folder contains best.pt
echo.
echo Problem: Application won't start
echo Solution: 1. Make sure you unzipped the ENTIRE folder
echo           2. Try running START_TOOL.bat as Administrator
echo              ^(Right-click → Run as administrator^)
echo           3. Check your antivirus isn't blocking it
echo.
echo Problem: Port 7860 already in use
echo Solution: Close any other programs using that port,
echo           or restart your computer
echo.
echo ========================================
echo UPDATING THE MODEL
echo ========================================
echo.
echo To update to a newer model version:
echo 1. Replace the file: models\best.pt
echo 2. Restart the application
echo.
echo No reinstallation needed!
echo.
echo ========================================
echo NEED HELP?
echo ========================================
echo.
echo Contact: [YOUR EMAIL HERE]
echo.
echo ========================================
) > portable_package\README.txt

echo.
echo Step 8: Creating quick start batch file...
(
echo @echo off
echo echo This will create a desktop shortcut to the tool.
echo echo.
echo set "TARGET=%%CD%%\START_TOOL.bat"
echo set "SHORTCUT=%%USERPROFILE%%\Desktop\Alfalfa Stem Tool.lnk"
echo powershell "$s=(New-Object -COM WScript.Shell).CreateShortcut('%%SHORTCUT%%'); $s.TargetPath='%%TARGET%%'; $s.WorkingDirectory='%%CD%%'; $s.Save()"
echo echo.
echo echo Desktop shortcut created!
echo pause
) > portable_package\Create_Desktop_Shortcut.bat

echo.
echo ========================================
echo   Package Created Successfully!
echo ========================================
echo.
echo Location: C:\Users\Abi\Documents\GitHub\Dave-bot\portable_package\
echo.
echo Package contains:
echo   - START_TOOL.bat           (Main launcher - clients click this!)
echo   - README.txt               (User instructions)
echo   - Create_Desktop_Shortcut.bat  (Optional - creates desktop shortcut)
echo   - app\                     (Your application code)
echo   - models\best.pt           (AI model)
echo   - python_env\              (Complete Python 3.9 environment)
echo.
echo Approximate size: 1-1.5 GB
echo.
echo ========================================
echo NEXT STEPS
echo ========================================
echo.
echo 1. TEST IT: Run portable_package\START_TOOL.bat
echo    - Make sure the application starts
echo    - Try uploading a test file
echo    - Verify analysis works
echo.
echo 2. DISTRIBUTE: 
echo    - Zip the entire "portable_package" folder
echo    - Name it: AlfalfaStemTool_v1.0.zip
echo    - Send to your clients
echo.
echo 3. CLIENT INSTRUCTIONS:
echo    - Unzip the folder
echo    - Double-click START_TOOL.bat
echo    - That's it!
echo.
echo ========================================
echo.
echo Press any key to test the tool now...
pause

cd portable_package
call START_TOOL.bat
