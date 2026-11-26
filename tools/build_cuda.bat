@echo off
setlocal EnableDelayedExpansion

set "ROOT_DIR=%~dp0.."
set "SRC_DIR=%ROOT_DIR%\WeatherSimCuda"
set "BUILD_DIR=%ROOT_DIR%\build_cuda"

rem Optional user-local overrides (gitignored): set VS_VCVARS_BAT, CUDA_PATH, CMAKE_BIN_DIR, NINJA_BIN_DIR, CUDA_ARCHS
if exist "%ROOT_DIR%\tools\local_env.bat" (
    call "%ROOT_DIR%\tools\local_env.bat"
)

rem 1. Initialize MSVC x64 toolchain if cl.exe is not already in PATH
where cl.exe >nul 2>nul
if errorlevel 1 (
    if not defined VS_VCVARS_BAT (
        set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
        if exist "!VSWHERE!" (
            for /f "usebackq tokens=*" %%i in (`"!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
                if exist "%%i\VC\Auxiliary\Build\vcvars64.bat" set "VS_VCVARS_BAT=%%i\VC\Auxiliary\Build\vcvars64.bat"
            )
        )
    )
    if not defined VS_VCVARS_BAT (
        echo [ERROR] MSVC vcvars64.bat not found. Install Visual Studio Build Tools C++ workload, or set VS_VCVARS_BAT in tools\local_env.bat.
        exit /b 1
    )
    call "!VS_VCVARS_BAT!" >nul
    if errorlevel 1 (
        echo [ERROR] Failed to initialize MSVC environment from: !VS_VCVARS_BAT!
        exit /b 1
    )
)

rem 2. Add optional local tool paths (CMake / Ninja / CUDA)
if defined CUDA_PATH (
    if exist "%CUDA_PATH%\bin" set "PATH=%CUDA_PATH%\bin;%PATH%"
)
if defined CMAKE_BIN_DIR (
    if exist "%CMAKE_BIN_DIR%" set "PATH=%CMAKE_BIN_DIR%;%PATH%"
)
if defined NINJA_BIN_DIR (
    if exist "%NINJA_BIN_DIR%" set "PATH=%NINJA_BIN_DIR%;%PATH%"
)

where nvcc.exe >nul 2>nul
if errorlevel 1 (
    echo [ERROR] nvcc.exe not found. Install NVIDIA CUDA Toolkit 12.x/13.x and set CUDA_PATH.
    exit /b 1
)
where cmake.exe >nul 2>nul
if errorlevel 1 (
    echo [ERROR] cmake.exe not found. Install CMake 3.24+ or set CMAKE_BIN_DIR in tools\local_env.bat.
    exit /b 1
)

rem 3. Pick Ninja if available, otherwise let CMake use its default generator
set "GEN_ARGS="
where ninja.exe >nul 2>nul
if not errorlevel 1 (
    set "GEN_ARGS=-G Ninja"
)

set "ARCH_ARGS="
if defined CUDA_ARCHS (
    set "ARCH_ARGS=-DCMAKE_CUDA_ARCHITECTURES=%CUDA_ARCHS%"
)

echo [INFO] Configuring WeatherSimCuda...
cmake -S "%SRC_DIR%" -B "%BUILD_DIR%" %GEN_ARGS% -DCMAKE_BUILD_TYPE=Release %ARCH_ARGS%
if errorlevel 1 exit /b 1

echo [INFO] Building WeatherSimCuda.dll...
cmake --build "%BUILD_DIR%" --config Release
if errorlevel 1 exit /b 1

echo [OK] Build succeeded: %BUILD_DIR%\bin\WeatherSimCuda.dll
endlocal
