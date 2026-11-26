@echo off
setlocal

set "ROOT_DIR=%~dp0.."
set "VERSION=1.0.3"
if not "%~1"=="" set "VERSION=%~1"

rem Build universal multi-GPU CUDA binary (RTX 20 / 30 / 40 / 50 + PTX forward compatibility)
set "CUDA_ARCHS=75;80;86;89;120;120-virtual"

rem Clean build_cuda CMake cache so CUDA_ARCHS takes effect cleanly
if exist "%ROOT_DIR%\build_cuda\CMakeCache.txt" del /F /Q "%ROOT_DIR%\build_cuda\CMakeCache.txt"

call "%ROOT_DIR%\tools\build_all.bat"
if errorlevel 1 exit /b 1

set "STAGE_DIR=%ROOT_DIR%\dist\release_stage"
set "PLUGIN_DIR=%STAGE_DIR%\BepInEx\plugins\WeatherExpansionMod"
set "ZIP_PATH=%ROOT_DIR%\dist\WeatherExpansionMod-v%VERSION%-win64-cuda.zip"

if exist "%STAGE_DIR%" rmdir /S /Q "%STAGE_DIR%"
if exist "%ZIP_PATH%" del /F /Q "%ZIP_PATH%"
mkdir "%PLUGIN_DIR%"

copy /Y "%ROOT_DIR%\dist\WeatherExpansionMod\WeatherExpansionMod.dll" "%PLUGIN_DIR%\" >nul
copy /Y "%ROOT_DIR%\dist\WeatherExpansionMod\WeatherSimCuda.dll" "%PLUGIN_DIR%\" >nul
if exist "%ROOT_DIR%\README.md" copy /Y "%ROOT_DIR%\README.md" "%STAGE_DIR%\" >nul
if exist "%ROOT_DIR%\README_EN.md" copy /Y "%ROOT_DIR%\README_EN.md" "%STAGE_DIR%\" >nul
if exist "%ROOT_DIR%\LICENSE" copy /Y "%ROOT_DIR%\LICENSE" "%STAGE_DIR%\" >nul
if exist "%ROOT_DIR%\CHANGELOG.md" copy /Y "%ROOT_DIR%\CHANGELOG.md" "%STAGE_DIR%\" >nul

powershell -NoProfile -Command "Compress-Archive -Path '%STAGE_DIR%\*' -DestinationPath '%ZIP_PATH%' -Force"
if errorlevel 1 (
    echo [ERROR] Failed to create release zip archive.
    exit /b 1
)

rmdir /S /Q "%STAGE_DIR%"
echo [OK] Release package ready: %ZIP_PATH%
endlocal
