#pragma once

#include <stdint.h>

#ifdef _WIN32
  #ifdef WEATHER_SIM_CUDA_EXPORTS
    #define WEATHER_API __declspec(dllexport)
  #else
    #define WEATHER_API __declspec(dllimport)
  #endif
#else
  #define WEATHER_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Cell flags in cellType buffer
// 0 = Air
// 1 = Solid wall (impermeable to airflow, heat relaxation boundary)
// 2 = Water body (impermeable to airflow, moisture evaporation source)
enum WeatherCellType : uint8_t {
    WEATHER_CELL_AIR   = 0,
    WEATHER_CELL_SOLID = 1,
    WEATHER_CELL_WATER = 2
};

// All physical parameters in a flat POD struct (C# StructLayout Sequential compatible)
typedef struct WeatherSimConfig {
    int32_t width;                 // Simulation grid width (cells)
    int32_t height;                // Simulation grid height (cells)
    float   cellSizeTiles;         // 0.5f when 1 game tile = 2x2 (4) sim cells

    float   midLayerTempC;         // Vanilla layer ambient temperature applied at y = height / 2 (deg C)
    float   bgLapseRateCPerTile;   // Background vertical temp gradient dT/dy (deg C per game tile, positive = cooler above)
    float   adiabaticLapseCPerTile;// Effective dry adiabatic lapse rate (deg C per game tile)
    float   buoyancyScale;         // Effective gravity scaling for Boussinesq buoyancy (tiles/s^2)

    float   viscosity;             // Momentum diffusion coefficient (tiles^2/s)
    float   thermalDiffusivity;    // Temperature diffusion coefficient (tiles^2/s)
    float   moistureDiffusivity;   // Vapor/cloud diffusion coefficient (tiles^2/s)
    float   vorticityEps;          // Mild vorticity confinement strength (0 = off)
    float   velocityDamping;       // Mild linear drag per second (1/s)
    float   maxSpeedTilesPerSec;   // Velocity clamp for numerical & physics safety

    float   wallTempRelaxRate;     // Rate air adjacent to normal walls relaxes toward T_env(y) (1/s)
    float   evapRateCoeff;         // Surface evaporation base coefficient (1/s)
    float   evapWindFactor;        // Evaporation wind enhancement factor (s/tile)
    float   evapCoolingScale;      // Scale on evaporative cooling (1.0 = full physical Lv/cp)
    float   latentHeatScale;       // Scale on condensation/deposition latent heat (1.0 = full physical L/cp)

    float   autoconvertThresh;     // Cloud water qc threshold to form rain qr (kg/kg)
    float   autoconvertRate;       // qc -> qr rate (1/s)
    float   freezeRate;            // Supercooled qc -> qi rate below 0 C (1/s)
    float   meltRate;              // Snow qi -> qr melt rate above 0 C (1/(s*C))
    float   rainFallSpeed;         // Terminal rain fall speed (tiles/s)
    float   snowFallSpeed;         // Terminal snow fall speed (tiles/s)
    float   snowDepositRate;       // Fraction/rate of snow settling onto ground surface (1/s)

    int32_t sorIterations;         // Red-Black SOR iterations per step (e.g. 24-40)
    float   sorOmega;              // SOR over-relaxation factor (e.g. 1.75-1.85)
    float   pressureWarmStart;     // Reuse previous pressure field multiplier (e.g. 0.95)
    int32_t useMacCormack;         // 1 = MacCormack 2nd-order advection, 0 = Semi-Lagrangian
    float   initialRelHumidity;    // Initial RH [0..1] when initializing field

    // Layer weather modifier physical drivers (zero on normal layers; preserves mid-layer T_mid at y = H/2)
    float   bottomTempBoostC;      // Rainy/Layer7: extra warmth in lower stratum (deg C)
    float   topTempDropC;          // Rainy/Layer6/7: extra cooling in upper stratum (deg C, positive = colder top)
    float   bottomMoistureSource;  // Rainy/Foggy: steady lower-stratum vapor injection (kg/kg per s)
    float   ambientHumidityTarget; // Foggy/Layer6: gentle ambient RH floor (>0 enables slow relaxation)
    float   ambientHumidityRate;   // Relaxation rate toward ambientHumidityTarget (1/s)

    // Hydrological water-cycle thresholds (game-tile scale)
    int32_t subdivision;           // Sim cells per 1 game tile axis (1 or 2)
    float   evapMassPerWaterTile;  // Cumulative evap mass in a tile to remove 1 FluidManager water block
    float   rainMassPerWaterTile;  // Cumulative rain deposit in a tile to spawn 1 FluidManager water block
} WeatherSimConfig;

typedef struct WeatherFluidEvent {
    int32_t tileX;                 // Local tile X [0, width/subdivision)
    int32_t tileY;                 // Local tile Y [0, height/subdivision)
    int32_t delta;                 // -1 = evaporated 1 water tile, +1 = rainfall formed 1 water tile
} WeatherFluidEvent;

typedef struct WeatherPrecipCell {
    float   u;
    float   v;
    float   tempC;
    float   qc;
    float   qr;
    float   qi;
    float   snowAccum;
    int32_t isAir;
} WeatherPrecipCell;

typedef struct WeatherSimStats {
    float    lastStepGpuMs;        // Last Sim_Step CUDA event elapsed time (ms)
    uint64_t allocatedVramBytes;   // Exact VRAM allocated by this sim instance
    uint64_t freeVramBytes;        // Current device free VRAM
    uint64_t totalVramBytes;       // Current device total VRAM
    int32_t  stepCount;            // Total steps executed
    int32_t  lastCudaError;        // 0 = cudaSuccess
} WeatherSimStats;

typedef struct WeatherEntitySample {
    float u;           // Wind X (tiles/s)
    float v;           // Wind Y (tiles/s)
    float tempC;       // Air temperature (deg C)
    float relHumidity; // Relative humidity [0..1+]
    float cloud;       // Cloud/fog density (kg/kg)
    float rain;        // Rain density (kg/kg)
    float snow;        // Snow/ice density (kg/kg)
} WeatherEntitySample;

typedef struct WeatherSimHandle WeatherSimHandle;

// Fill config with sane, tested defaults for 1 game tile = 2x2 sim cells
WEATHER_API void WeatherSim_DefaultConfig(WeatherSimConfig* outConfig, int32_t width, int32_t height);

// Lifecycle
WEATHER_API WeatherSimHandle* WeatherSim_Create(const WeatherSimConfig* config);
WEATHER_API void WeatherSim_Destroy(WeatherSimHandle* handle);

// Update runtime parameters (e.g. vanilla ambientTemperature changing over time)
WEATHER_API void WeatherSim_UpdateConfig(WeatherSimHandle* handle, const WeatherSimConfig* config);
WEATHER_API void WeatherSim_SetMidLayerTemp(WeatherSimHandle* handle, float midLayerTempC);

// Reset fields to resting hydrostatic state at current midLayerTempC and initialRelHumidity
WEATHER_API int32_t WeatherSim_ResetFields(WeatherSimHandle* handle);

// Upload full cellType grid (size = width * height, row-major y * width + x)
WEATHER_API int32_t WeatherSim_UploadCellsFull(WeatherSimHandle* handle, const uint8_t* cellTypes);

// Upload incremental cellType updates (for SetBlock / water changes during play)
WEATHER_API int32_t WeatherSim_UploadCellsPatch(
    WeatherSimHandle* handle,
    const int32_t* xs,
    const int32_t* ys,
    const uint8_t* types,
    int32_t count);

// Upload full heat source field (deg C / s, size = width * height; positive = heater, negative = cooler)
WEATHER_API int32_t WeatherSim_UploadHeatSources(WeatherSimHandle* handle, const float* heatRateCPerSec);

// Upload optional wind forcing field (tiles/s^2, e.g. vents or ambient draft; can be null to clear)
WEATHER_API int32_t WeatherSim_UploadWindSources(
    WeatherSimHandle* handle,
    const float* forceX,
    const float* forceY);

// Advance simulation by dt seconds (internally substeps if needed for stability)
WEATHER_API int32_t WeatherSim_Step(WeatherSimHandle* handle, float dt);

// Sample wind/weather at arbitrary cell-space coordinates (cx in [0,W), cy in [0,H))
WEATHER_API int32_t WeatherSim_SamplePoints(
    WeatherSimHandle* handle,
    const float* cx,
    const float* cy,
    WeatherEntitySample* outSamples,
    int32_t count);

// Read individual scalar/velocity fields back to host (size = width * height, cell-centered)
// fieldId:
//   0 = u_center (tiles/s)
//   1 = v_center (tiles/s)
//   2 = tempC (deg C)
//   3 = qv (kg/kg)
//   4 = qc (kg/kg)
//   5 = qr (kg/kg)
//   6 = qi (kg/kg)
//   7 = snowAccum (kg/kg equivalent surface deposit)
//   8 = relHumidity (0..1+)
//   9 = divergence (1/s, for diagnostics)
//  10 = latentHeatRate (deg C/s, last step phase change heating/cooling)
//  11 = evapAccum (cumulative evaporation mass per cell)
WEATHER_API int32_t WeatherSim_CopyField(
    WeatherSimHandle* handle,
    int32_t fieldId,
    float* outHostBuffer);

// Pack a cropped viewport window [x0, x0+winW) x [y0, y0+winH) into RGBA8 for fast Unity Texture2D upload
// mode: 0 = Cloud/Rain/Snow composite, 1 = Temperature heatmap, 2 = Relative humidity,
//       3 = Wind speed, 4 = Latent heat release/absorption
// In mode 0:
//   - Fog/cloud alpha near (playerCx, playerCy) smoothly attenuates down to playerMinFogFactor
//   - Fog/cloud alpha inside solid walls (WEATHER_CELL_SOLID) is multiplied by wallFogMultiplier (e.g. 0.5)
WEATHER_API int32_t WeatherSim_ReadViewWindowRGBA8(
    WeatherSimHandle* handle,
    int32_t x0,
    int32_t y0,
    int32_t winW,
    int32_t winH,
    int32_t mode,
    float playerCx,
    float playerCy,
    float playerClearInnerCells,
    float playerClearOuterCells,
    float playerMinFogFactor,
    float wallFogMultiplier,
    uint8_t* outRgbaBytes);

// Sample a compact downsampled grid (outW x outH) across viewport [x0, x0+winW) x [y0, y0+winH)
// Used by Unity to spawn snowflakes/raindrops with exact local wind & phase densities (<0.05 ms)
WEATHER_API int32_t WeatherSim_ReadViewPrecipGrid(
    WeatherSimHandle* handle,
    int32_t x0,
    int32_t y0,
    int32_t winW,
    int32_t winH,
    int32_t outW,
    int32_t outH,
    WeatherPrecipCell* outCells);

// Drain discrete water-cycle events generated by evaporation (-1) and rainfall (+1).
// Returns number of events written to outEvents (up to maxEvents), or negative on error.
WEATHER_API int32_t WeatherSim_PopFluidEvents(
    WeatherSimHandle* handle,
    WeatherFluidEvent* outEvents,
    int32_t maxEvents);

// Diagnostics & VRAM stats
WEATHER_API int32_t WeatherSim_GetStats(WeatherSimHandle* handle, WeatherSimStats* outStats);

#ifdef __cplusplus
}
#endif
