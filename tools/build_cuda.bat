@echo off
setlocal

set "ROOT_DIR=%~dp0.."
set "SRC_DIR=%ROOT_DIR%\WeatherSimCuda"
set "BUILD_DIR=%ROOT_DIR%\build_cuda"

call "D:\visual studio\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
    echo [ERROR] Failed to initialize MSVC environment.
    exit /b 1
)

set "CUDA_PATH=E:\CUDA\13.5"
set "PATH=E:\CUDA\13.5\bin;D:\geminispace\space1\cmake-4.5.1-windows-x86_64\bin;D:\visual studio\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"

echo [INFO] Configuring WeatherSimCuda with CMake + Ninja...
cmake -S "%SRC_DIR%" -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1

echo [INFO] Building WeatherSimCuda.dll...
cmake --build "%BUILD_DIR%"
if errorlevel 1 exit /b 1

echo [OK] Build succeeded: %BUILD_DIR%\bin\WeatherSimCuda.dll
endlocal
