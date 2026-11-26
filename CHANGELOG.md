# Changelog

All notable changes to **WeatherExpansionMod (天气拓展)** are documented in this file.

## [v1.0.3] - 2026-11-26
### Added
- **Bidirectional ±2000-Tile Hydrological Guardrails**:
  - Added `MaxNetEvapWaterTiles` (default `2000`) and raised `MaxNetRainWaterTiles` default from `40` to `2000` (with automatic migration for unchanged configs).
  - When net evaporation exceeds rainfall by `2000` tiles, water tiles stop being removed while GPU water cells are immediately preserved so water vapor keeps generating uninterrupted.
  - When net rainfall exceeds evaporation by `2000` tiles, ground water tile spawning pauses while clouds, rain particles, and player wetness continue normally.
  - Added symmetric two-pass event batching and `[保水:停消水]` / `[防涝:停生水]` indicators to the `F9` HUD.

## [v1.0.2] - 2026-11-26
### Fixed
- **Horizontal-Line Updraft/Downdraft Flux Imbalance**: Replaced local Red-Black SOR with an 8-level **Geometric Multigrid V-Cycle** pressure solver (with a single-block shared-memory coarsest solve), slashing `2048×2048` horizontal-line vertical flux imbalance from `99.2%` down to `< 0.01%` while reducing GPU step time from `7.86 ms` to `6.28 ms`.
### Changed
- Slowed water-block cycle turnover to 1/5 by raising `EvapMassPerWaterTile` and `RainMassPerWaterTile` 5× (`0.14 → 0.70`, preserving 1:1 mass conservation and full cloud/rain visual density, with automatic config migration).
- Halved block-to-gas thermal exchange (`WallTempRelaxRate`: `0.02 → 0.01`, `ThermalDiffusivity`: `0.02 → 0.01`, hot terrain blocks `3.0 → 1.5`, magma `5.5 → 2.75`) while keeping dynamic player heaters (campfires, torches, Heater crystals) at full strength.

## [v1.0.1] - 2026-11-26
### Added
- **Interactive Full-Map Weather Radar UI (`F7`)**: Zoomable/pannable 7-layer radar (Relative Humidity, Cloud, Rain, Snow, Temperature with 0°C isotherm, Wind Speed + vector arrows, Composite) with mouse-hover meteorological telemetry.
- **Live Dynamic Heat Source Synchronization**: Real-time detection of placed, extinguished, or demolished campfires, torches, Heater crystals, and hot blocks via lightweight GPU stamping (`WeatherSim_UploadDynamicHeaters` / `WeatherSim_UploadStaticHeatPatch`).
### Changed
- Enforced strict 1:1 physical water mass conservation (`evapMassPerWaterTile == rainMassPerWaterTile`) and removed infinite artificial moisture injection on Rainy/Foggy modifiers.

## [v1.0.0] - 2026-11-26
### Added
- Initial release of the CUDA 2D atmospheric weather simulation mod (`WeatherSimCuda.dll` + `WeatherExpansionMod.dll`):
  - Dedicated background CUDA worker thread (`WeatherSimCudaWorker`) with double-buffered viewport & particle readbacks.
  - Potential-temperature adiabatic cooling/warming, Dalton in-water evaporation, condensation/deposition latent heat release, and snow formation.
  - Full-viewport coprime-strided rain/snow particle system, player-proximity fog attenuation, and halved wall-interior fog opacity.
  - Vanilla layer weather override, body temperature & wetness coupling, aerodynamic wind forces on `Rigidbody2D` entities, `F8` debug overlay modes, `F9` telemetry HUD, and standalone Python sandbox renderer (`tools/verify_weather_sandbox.py`).
