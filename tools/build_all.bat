@echo off
setlocal

set "ROOT_DIR=%~dp0.."

call "%ROOT_DIR%\tools\build_cuda.bat"
if errorlevel 1 exit /b 1

echo [INFO] Building WeatherExpansionMod (C# BepInEx 5)...
dotnet build "%ROOT_DIR%\WeatherExpansionMod\WeatherExpansionMod.csproj" -c Release
if errorlevel 1 exit /b 1

set "DIST_DIR=%ROOT_DIR%\dist\WeatherExpansionMod"
if not exist "%DIST_DIR%" mkdir "%DIST_DIR%"
copy /Y "%ROOT_DIR%\WeatherExpansionMod\bin\Release\netstandard2.1\WeatherExpansionMod.dll" "%DIST_DIR%\" >nul
copy /Y "%ROOT_DIR%\build_cuda\bin\WeatherSimCuda.dll" "%DIST_DIR%\" >nul

echo [OK] Ready-to-install Mod bundle created at:
echo      %DIST_DIR%
dir /B "%DIST_DIR%"
endlocal
