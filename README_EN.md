# WeatherExpansionMod (天气拓展)

**English** | [简体中文](README.md)

A **CUDA GPU-accelerated 2D physical weather simulation mod** for the 2D survival game **Casualties: Unknown** (built with BepInEx 5, Harmony, and a self-contained native CUDA solver).

Running asynchronously on a dedicated GPU worker thread, this mod replaces vanilla static fog/rain overlays with a real-time incompressible fluid & atmospheric thermodynamics engine driven by cavern terrain, water pools, magma, and player-built campfires.

---

## Key Features

### 1. Native CUDA 2D Atmospheric Solver (`WeatherSimCuda.dll`)
- **MAC Staggered Grid + Geometric Multigrid V-Cycle Pressure Solver**:
  - Solves 2D incompressible Navier-Stokes flow using an 8-level Geometric Multigrid solver with a single-block shared-memory coarsest solve, guaranteeing global mass conservation (`< 0.01%` horizontal-line vertical flux imbalance at `2048×2048`) so updrafts naturally drive compensating downdrafts.
  - Includes Vorticity Confinement and 2nd-order MacCormack scalar advection for crisp convective plumes and eddies.
- **Potential-Temperature Adiabatic Process & Phase-Change Latent Heat**:
  - Anchors mid-stratum background temperature directly to each layer's vanilla ambient temperature.
  - Rising air expands and cools adiabatically; water vapor condenses (`> 0°C`) or deposits into snow/ice (`≤ 0°C`) while releasing physical latent heat to energize moist convection.
- **Mass-Conserved Hydrology & Bidirectional ±2000-Tile Guardrails**:
  - **Dalton's Law Evaporation**: Airflow passing through game water tiles evaporates vapor proportionally to wind speed and saturation deficit, cooling the air and depleting water tiles; rainfall returns liquid water onto cavern floors at a strict 1:1 mass ratio.
  - **Bidirectional ±2000 Water-Tile Clamp**: Stops removing water tiles when net evaporation reaches `2000` tiles (while continuing to generate water vapor) and stops spawning new ground water tiles when net rainfall reaches `2000` tiles (while keeping clouds, rain particles, and wetness active).
- **Live Dynamic Heat Source Synchronization**:
  - Tracks both static terrain heat (hot blocks, magma) and dynamic player-built heat sources (campfires, lit torches, Heater crystals) in real time as they are placed, extinguished, or demolished.

### 2. Deep Gameplay Integration (`WeatherExpansionMod.dll`)
- **Physical Replacement for Vanilla Weather Modifiers**: Overrides flat vanilla overlays (`Foggy`, `Rainy`, `Dry`, `Flooded`, Layer 6 cold fog, Layer 7 fungal rain) with true thermodynamic boundary drivers.
- **Body Temperature & Aerodynamic Wind Forces**: Feeds local simulated air temperature, wind chill, and rain/snow wetness into `Body.HandleBodyTemperature`, and applies aerodynamic drag forces to nearby `Rigidbody2D` entities.
- **Player-Friendly Visuals**: Automatically clears fog around the player, halves cloud opacity inside solid rock, and renders balanced full-viewport wind-driven rain/snow particles.
- **Zero Main-Thread GPU Stall**: Runs all CUDA stepping and viewport readbacks on a background worker thread (`WeatherSimCudaWorker`) with double-buffered readbacks.

### 3. Interactive Full-Map Weather Radar & Diagnostics
- **`F7` — Full-Map Weather Radar UI**: Zoomable, pannable radar across 7 layers (**Relative Humidity / Cloud / Rain / Snow / Temperature with 0°C isotherm / Wind Speed & Arrows / Composite**) with live mouse-hover weather readout.
- **`F8` — Viewport Overlay Mode**: Natural Clouds/Mist → Temperature → Humidity → Wind Speed → Latent Heat → Off.
- **`F9` — Telemetry & Weather HUD**: Local temperature, wind chill, RH%, wind vector, hydrological balance counters, and GPU timing/VRAM.

---

## Performance (`1024×1024` tiles → `2048×2048` sim cells, 4 sim cells per game tile)

| GPU | Grid Size | Step Time | VRAM | Rate |
| :--- | :---: | :---: | :---: | :---: |
| NVIDIA RTX 5060 (8 GB) | `2048 × 2048` (4.19M cells) | **~6.3 ms** | **~532 MB** | 30 Hz Async |

---

## Requirements

- **Casualties: Unknown** (Windows x64)
- **BepInEx 5.x** (x64)
- **NVIDIA GPU (RTX 20 / 30 / 40 / 50 series, Compute Capability ≥ 7.5)** with up-to-date driver
  > Release builds of `WeatherSimCuda.dll` statically link `cudart_static` and bundle multi-architecture binaries (`sm_75 / 80 / 86 / 89 / 120 + PTX`) — **end users do NOT need to install the CUDA Toolkit**.

---

## Installation

1. Install **BepInEx 5 (x64)** into your *Casualties: Unknown* directory.
2. Download the latest `WeatherExpansionMod-vX.Y.Z-win64-cuda.zip` from [Releases](../../releases).
3. Extract into the game root folder so that both DLLs sit inside:
   ```text
   <GameRoot>/BepInEx/plugins/WeatherExpansionMod/
   ├── WeatherExpansionMod.dll
   └── WeatherSimCuda.dll
   ```
4. Launch the game and enter any layer. Configuration is generated automatically at `BepInEx/config/com.casualties.weather.expansion.cfg`.

---

## Building from Source

### Prerequisites
- **Visual Studio 2022 / Build Tools** (Desktop development with C++)
- **NVIDIA CUDA Toolkit 12.x / 13.x**
- **CMake 3.24+** (Ninja recommended)
- **.NET SDK 8.0+** (`netstandard2.1` target)

### Steps
1. Copy [Directory.Build.props.example](Directory.Build.props.example) to `Directory.Build.props` (gitignored) and point `<GameDir>` to your *Casualties: Unknown* install folder.
2. If MSVC, CUDA, or CMake are not in your system `PATH`, copy [tools/local_env.bat.example](tools/local_env.bat.example) to `tools/local_env.bat` and configure the paths.
3. Run:
   ```bat
   tools\build_all.bat
   ```
4. To package a universal multi-GPU GitHub Release zip:
   ```bat
   tools\package_release.bat 1.0.3
   ```

### Standalone Python Physics Sandbox
Preview or record 16:9 videos of the CUDA weather engine without launching Unity (requires `numpy`, `matplotlib`, and optionally `ffmpeg`):
```bash
python tools/verify_weather_sandbox.py --preset snow --live
python tools/verify_weather_sandbox.py --preset rainy --record output/weather_demo.mp4
```

---

## Changelog, Contributing & License

- Release history: [CHANGELOG.md](CHANGELOG.md)
- Contributing guidelines: [CONTRIBUTING.md](CONTRIBUTING.md)
- Licensed under the [MIT License](LICENSE).
