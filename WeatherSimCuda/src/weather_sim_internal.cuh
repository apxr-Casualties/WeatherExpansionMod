#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include "weather_sim_api.h"

struct WeatherSimDeviceBuffers {
    // Geometry & sources (W * H)
    uint8_t* cellType = nullptr;
    float*   heatSource = nullptr;
    float*   windForceX = nullptr;
    float*   windForceY = nullptr;

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
cudaError_t launchPatchCells(WeatherSimHandle* h, int32_t count);
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
cudaError_t launchResolveFluidEvents(WeatherSimHandle* h);
