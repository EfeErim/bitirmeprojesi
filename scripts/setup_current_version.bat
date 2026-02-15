@echo off
REM Setup script to set v5.5.4-dinov3 as the active version

echo Setting up v5.5.4-dinov3 as the active version...

REM Create current directory if it doesn't exist
if not exist "current" mkdir current

REM Copy all files from v5.5.4-dinov3 to current directory
echo Copying files from versions\v5.5.4-dinov3 to current\...
robocopy versions\v5.5.4-dinov3 current /E /XD .git backups versions version_management /XF backup.log

REM Create a version.json in current directory to mark the active version
echo Creating version marker...
echo {> current\version.json
echo   "version": "v5.5.4-dinov3",>> current\version.json
echo   "active": true>> current\version.json
echo }>> current\version.json

echo.
echo Active version set to: v5.5.4-dinov3
echo Current directory now contains the active version files.
echo.
echo To verify, run: python -m version_management.backup current
pause