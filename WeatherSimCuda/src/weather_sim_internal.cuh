#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include "weather_sim_api.h"

static constexpr int kMaxMgLevels = 10;

struct WeatherMgLevel {
    int32_t w = 0;
    int32_t h = 0;
    float*  p = nullptr;   // Level 0 aliases WeatherSimDeviceBuffers::p
    float*  rhs = nullptr; // Level 0 aliases WeatherSimDeviceBuffers::div
    float*  res = nullptr; // Residual buffer (levels 0..mgNumLevels-2)
};

struct WeatherSimDeviceBuffers {
    // Geometry & sources (W * H)
    uint8_t* cellType = nullptr;
    float*   heatSource = nullptr;     // Static terrain/layer heat & cooling (deg C / s)
    float*   moistureSource = nullptr; // Static moisture (>0 warm vapor kg/kg/s; <0 evaporative spray -rate 1/s)
    float*   dynHeat = nullptr;        // Dynamic scene heat (campfires, torches, Heater crystals)
    float*   windForceX = nullptr;
    float*   windForceY = nullptr;

    // Dynamic heat source upload scratch
    WeatherDynamicHeatSource* d_dynSources = nullptr;
    int32_t                   dynSourceCapacity = 0;

    int32_t* d_heatPatchX = nullptr;
    int32_t* d_heatPatchY = nullptr;
    float*   d_heatPatchVal = nullptr;
    int32_t  heatPatchCapacity = 0;

    // Full-map weather radar UI readback scratch
    uint8_t* d_fullMapRgba = nullptr;
    int32_t  fullMapRgbaCapacity = 0;
    float*   d_windGridU = nullptr;
    float*   d_windGridV = nullptr;
    int32_t  windGridCapacity = 0;

    // MAC staggered velocities
    // u: (W + 1) * H
    // v: W * (H + 1)
    float* u = nullptr;
    float* uTmp = nullptr;
    float* uHat = nullptr;
    float* v = nullptr;
    float* vTmp = nullptr;
    float* vHat = nullptr;

    // Pressure & diagnostics (W * H)
    float* p = nullptr;
    float* div = nullptr;
    float* vort = nullptr;

    // Geometric Multigrid V-Cycle pyramid (solves global low-frequency pressure modes)
    WeatherMgLevel mgLevels[kMaxMgLevels];
    int32_t        mgNumLevels = 0;

    // Thermodynamics & moisture (W * H)
    float* T = nullptr;
    float* TTmp = nullptr;
    float* THat = nullptr;

    float* qv = nullptr;
    float* qvTmp = nullptr;
    float* qvHat = nullptr;

    float* qc = nullptr;
    float* qcTmp = nullptr;
    float* qcHat = nullptr;

    float* qr = nullptr;
    float* qrTmp = nullptr;
    float* qi = nullptr;
    float* qiTmp = nullptr;

    // Surface & diagnostics (W * H)
    float* snowAccum = nullptr;
    float* evapAccum = nullptr;
    float* rainAccum = nullptr;
    float* latentRate = nullptr;

    // Discrete hydrological events (GPU ring buffer)
    WeatherFluidEvent* d_fluidEvents = nullptr;
    int32_t*           d_fluidEventCount = nullptr;
    int32_t            fluidEventCapacity = 0;

    // Reusable scratch for CopyField / SamplePoints / ViewWindow
    float*   scratchScalar = nullptr;
    float*   d_sampleX = nullptr;
    float*   d_sampleY = nullptr;
    WeatherEntitySample* d_sampleOut = nullptr;
    int32_t  sampleCapacity = 0;

    uint8_t* d_rgbaWindow = nullptr;
    int32_t  rgbaCapacity = 0;

    int32_t* d_patchX = nullptr;
    int32_t* d_patchY = nullptr;
    uint8_t* d_patchType = nullptr;
    int32_t  patchCapacity = 0;

    WeatherPrecipCell* d_precipGrid = nullptr;
    int32_t            precipCapacity = 0;
};

struct WeatherSimHandle {
    WeatherSimConfig cfg;
    WeatherSimDeviceBuffers d;
    cudaStream_t stream = nullptr;
    cudaEvent_t  evStart = nullptr;
    cudaEvent_t  evStop = nullptr;
    uint64_t     allocatedVramBytes = 0;
    float        lastStepGpuMs = 0.0f;
    int32_t      stepCount = 0;
    int32_t      lastCudaError = 0;
};

// Host-callable launcher declarations (implemented in weather_kernels.cu)
cudaError_t launchResetFields(WeatherSimHandle* h);
cudaError_t launchApplyInitialHumidityGradient(
    WeatherSimHandle* h,
    float bottomRh,
    float topRh,
    float yBottomFrac,
    float yTopFrac);
cudaError_t launchApplyInitialPerlinNoise(
    WeatherSimHandle* h,
    int32_t seed,
    float tempAmpC,
    float rhAmp,
    float wavelengthTiles,
    int32_t octaves);
cudaError_t launchPatchCells(WeatherSimHandle* h, int32_t count);
cudaError_t launchPatchStaticHeat(WeatherSimHandle* h, int32_t count);
cudaError_t launchStampDynamicHeaters(WeatherSimHandle* h, int32_t count);
cudaError_t launchStepSimulation(WeatherSimHandle* h, float dt);
cudaError_t launchSamplePoints(WeatherSimHandle* h, int32_t count);
cudaError_t launchPrepareFieldForCopy(WeatherSimHandle* h, int32_t fieldId, const float** outDevicePtr);
cudaError_t launchPackViewWindowRGBA8(
    WeatherSimHandle* h,
    int32_t x0, int32_t y0,
    int32_t winW, int32_t winH,
    int32_t mode,
    float playerCx, float playerCy,
    float playerClearInnerCells, float playerClearOuterCells,
    float playerMinFogFactor, float wallFogMultiplier);
cudaError_t launchSampleViewPrecipGrid(
    WeatherSimHandle* h,
    int32_t x0, int32_t y0,
    int32_t winW, int32_t winH,
    int32_t outW, int32_t outH);
cudaError_t launchRenderFullMapRGBA8(
    WeatherSimHandle* h,
    int32_t mapW, int32_t mapH,
    int32_t layerMode, int32_t showWalls);
cudaError_t launchSampleMapWindGrid(
    WeatherSimHandle* h,
    int32_t gridW, int32_t gridH);
cudaError_t launchResolveFluidEvents(WeatherSimHandle* h);
