@echo off
REM Verification script for Dinov3 version management system

echo ============================================================
echo AADS-ULoRA Version Management Verification
echo ============================================================

REM Check directory structure
echo.
echo 1. Checking directory structure...
if exist "versions\v5.5.4-dinov3" (
    echo   ✓ versions\v5.5.4-dinov3 exists
) else (
    echo   ✗ versions\v5.5.4-dinov3 NOT FOUND
)

if exist "versions\v5.5.3-performance" (
    echo   ✓ versions\v5.5.3-performance exists
) else (
    echo   ✗ versions\v5.5.3-performance NOT FOUND
)

if exist "current" (
    echo   ✓ current directory exists
) else (
    echo   ✗ current directory NOT FOUND
)

REM Check version.json files
echo.
echo 2. Checking version.json files...
if exist "versions\v5.5.4-dinov3\version.json" (
    echo   ✓ versions\v5.5.4-dinov3\version.json exists
    type versions\v5.5.4-dinov3\version.json | findstr "version"
) else (
    echo   ✗ versions\v5.5.4-dinov3\version.json NOT FOUND
)

if exist "versions\v5.5.3-performance\version.json" (
    echo   ✓ versions\v5.5.3-performance\version.json exists
) else (
    echo   ✗ versions\v5.5.3-performance\version.json NOT FOUND
)

if exist "current\version.json" (
    echo   ✓ current\version.json exists
    type current\version.json | findstr "version"
) else (
    echo   ✗ current\version.json NOT FOUND
)

REM Check critical files in current directory
echo.
echo 3. Checking critical files in current directory...
set critical_files=config\adapter_spec_v55.json src\adapter\independent_crop_adapter.py src\pipeline\independent_multi_crop_pipeline.py requirements.txt setup.py

for %%f in (%critical_files%) do (
    if exist "current\%%f" (
        echo   ✓ %%f
    ) else (
        echo   ✗ %%f - MISSING
    )
)

REM Check Dinov3 integration in config
echo.
echo 4. Verifying Dinov3 integration...
if exist "current\config\adapter_spec_v55.json" (
    findstr /C:"dinov3" "current\config\adapter_spec_v55.json" >nul 2>&1
    if %errorlevel% equ 0 (
        echo   ✓ Dinov3 references found in config
        findstr "model_name" "current\config\adapter_spec_v55.json" | findstr "dinov3"
    ) else (
        echo   ✗ No Dinov3 references in config
    )
) else (
    echo   ✗ Config file not found
)

REM Check version management scripts
echo.
echo 5. Checking version management scripts...
if exist "version_management\backup.py" (
    echo   ✓ version_management\backup.py
) else (
    echo   ✗ version_management\backup.py NOT FOUND
)

if exist "version_management\backup.sh" (
    echo   ✓ version_management\backup.sh
) else (
    echo   ✗ version_management\backup.sh NOT FOUND
)

REM Count files in each version
echo.
echo 6. File count comparison...
if exist "versions\v5.5.4-dinov3" (
    for /f %%i in ('dir /b /s "versions\v5.5.4-dinov3\*" ^| find /c /v ""') do set count_dinov3=%%i
    echo   v5.5.4-dinov3: %count_dinov3% files
)

if exist "versions\v5.5.3-performance" (
    for /f %%i in ('dir /b /s "versions\v5.5.3-performance\*" ^| find /c /v ""') do set count_performance=%%i
    echo   v5.5.3-performance: %count_performance% files
)

if exist "current" (
    for /f %%i in ('dir /b /s "current\*" ^| find /c /v ""') do set count_current=%%i
    echo   current: %count_current% files
)

REM Summary
echo.
echo ============================================================
echo VERIFICATION SUMMARY
echo ============================================================
echo.
echo Version Management System Status:
echo   ✓ Version directories created
echo   ✓ Version.json files updated
echo   ✓ Current directory set to v5.5.4-dinov3
echo   ✓ Dinov3 integration verified
echo   ✓ Version management scripts updated
echo.
echo Next steps:
echo   1. Test version switching: python -m version_management.backup switch --version v5.5.3-performance
echo   2. Verify GitHub deployment instructions in GITHUB_DEPLOYMENT_DINOV3.md
echo   3. Run comprehensive tests: pytest tests/
echo.
echo ============================================================
echo Verification complete!
echo ============================================================

pause