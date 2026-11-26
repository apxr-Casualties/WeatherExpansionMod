# Contributing to WeatherExpansionMod

Thank you for your interest in contributing to **WeatherExpansionMod (天气拓展)**!

## Development Setup

1. Clone this repository.
2. Copy `Directory.Build.props.example` to `Directory.Build.props` (gitignored) and set `<GameDir>` to your local *Casualties: Unknown* installation directory.
3. If your Visual Studio Build Tools, CUDA Toolkit, CMake, or Ninja are installed outside the system `PATH`, copy `tools/local_env.bat.example` to `tools/local_env.bat` (gitignored) and set the corresponding variables.
4. For faster local CUDA iteration, you may set `CUDA_ARCHS=<your_gpu_sm>` (e.g. `86` for RTX 30 series, `89` for RTX 40 series, `120` for RTX 50 series) in `tools/local_env.bat`.

## Testing & Verification

Before opening a Pull Request:
- Build cleanly with zero warnings and zero errors via `tools\build_all.bat`.
- Run the standalone CUDA physics & integrity test suite:
  ```bash
  python tools/verify_weather_sandbox.py --preset snow
  ```
- Keep `WeatherSimCuda.dll` self-contained (statically linked to `CUDA::cudart_static`) so players do not need external CUDA runtime DLLs.
- Keep `WeatherSimConfig` struct layout strictly identical between `WeatherSimCuda/include/weather_sim_api.h`, `WeatherExpansionMod/Native/WeatherSimNative.cs`, and `tools/weather_sim_binding.py`.

## Reporting Bugs

When filing a bug report, please include:
- Mod version & game version
- GPU model and NVIDIA driver version
- Relevant excerpts from `BepInEx/LogOutput.log`
- Clear reproduction steps or screenshots (`F7` radar / `F9` HUD are especially helpful)
