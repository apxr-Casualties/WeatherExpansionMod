#include "weather_sim_internal.cuh"
#include <string.h>
#include <algorithm>

namespace {

template <typename T>
cudaError_t allocBuf(T** ptr, size_t count, uint64_t& totalBytes) {
    size_t bytes = count * sizeof(T);
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(ptr), bytes);
    if (err == cudaSuccess) {
        totalBytes += bytes;
        cudaMemset(*ptr, 0, bytes);
    }
    return err;
}

template <typename T>
void freeBuf(T*& ptr) {
    if (ptr) {
        cudaFree(ptr);
        ptr = nullptr;
    }
}

void destroyAllBuffers(WeatherSimHandle* h) {
    if (!h) return;
    freeBuf(h->d.cellType);
    freeBuf(h->d.heatSource);
    freeBuf(h->d.moistureSource);
    freeBuf(h->d.dynHeat);
    freeBuf(h->d.windForceX);
    freeBuf(h->d.windForceY);
    freeBuf(h->d.d_dynSources);
    freeBuf(h->d.d_heatPatchX);
    freeBuf(h->d.d_heatPatchY);
    freeBuf(h->d.d_heatPatchVal);
    freeBuf(h->d.d_fullMapRgba);
    freeBuf(h->d.d_windGridU);
    freeBuf(h->d.d_windGridV);

    freeBuf(h->d.u);
    freeBuf(h->d.uTmp);
    freeBuf(h->d.uHat);
    freeBuf(h->d.v);
    freeBuf(h->d.vTmp);
    freeBuf(h->d.vHat);

    freeBuf(h->d.p);
    freeBuf(h->d.div);
    freeBuf(h->d.vort);

    for (int lvl = 0; lvl < h->d.mgNumLevels; ++lvl) {
        if (lvl > 0) {
            freeBuf(h->d.mgLevels[lvl].p);
            freeBuf(h->d.mgLevels[lvl].rhs);
        } else {
            h->d.mgLevels[lvl].p = nullptr;
            h->d.mgLevels[lvl].rhs = nullptr;
        }
        freeBuf(h->d.mgLevels[lvl].res);
        h->d.mgLevels[lvl].w = 0;
        h->d.mgLevels[lvl].h = 0;
    }
    h->d.mgNumLevels = 0;

    freeBuf(h->d.T);
    freeBuf(h->d.TTmp);
    freeBuf(h->d.THat);

    freeBuf(h->d.qv);
    freeBuf(h->d.qvTmp);
    freeBuf(h->d.qvHat);

    freeBuf(h->d.qc);
    freeBuf(h->d.qcTmp);
    freeBuf(h->d.qcHat);

    freeBuf(h->d.qr);
    freeBuf(h->d.qrTmp);
    freeBuf(h->d.qi);
    freeBuf(h->d.qiTmp);

    freeBuf(h->d.snowAccum);
    freeBuf(h->d.evapAccum);
    freeBuf(h->d.rainAccum);
    freeBuf(h->d.latentRate);
    freeBuf(h->d.scratchScalar);
    freeBuf(h->d.d_fluidEvents);
    freeBuf(h->d.d_fluidEventCount);

    freeBuf(h->d.d_sampleX);
    freeBuf(h->d.d_sampleY);
    freeBuf(h->d.d_sampleOut);
    freeBuf(h->d.d_rgbaWindow);
    freeBuf(h->d.d_patchX);
    freeBuf(h->d.d_patchY);
    freeBuf(h->d.d_patchType);
    freeBuf(h->d.d_precipGrid);

    if (h->evStart) { cudaEventDestroy(h->evStart); h->evStart = nullptr; }
    if (h->evStop)  { cudaEventDestroy(h->evStop);  h->evStop = nullptr; }
    if (h->stream)  { cudaStreamDestroy(h->stream); h->stream = nullptr; }
}

} // namespace

extern "C" {

WEATHER_API void WeatherSim_DefaultConfig(WeatherSimConfig* outConfig, int32_t width, int32_t height) {
    if (!outConfig) return;
    memset(outConfig, 0, sizeof(WeatherSimConfig));
    outConfig->width = width;
    outConfig->height = height;
    outConfig->cellSizeTiles = 0.5f;             // 1 game tile = 2x2 (4) simulation cells

    outConfig->midLayerTempC = 2.0f;             // Mid-layer vanilla ambient temperature (deg C)
    outConfig->bgLapseRateCPerTile = 0.04f;      // Background gradient: ~8C difference over 200 tiles
    outConfig->adiabaticLapseCPerTile = 0.065f;  // Scaled dry adiabatic lapse rate (rising air cools visibly!)
    outConfig->buoyancyScale = 60.0f;            // Doubled overall wind speed (~2x steady convective airflow)

    outConfig->viscosity = 0.025f;
    outConfig->thermalDiffusivity = 0.01f;       // Halved block/air conductive heat transfer (v1.0.2)
    outConfig->moistureDiffusivity = 0.02f;
    outConfig->vorticityEps = 0.90f;
    outConfig->velocityDamping = 0.008f;
    outConfig->maxSpeedTilesPerSec = 36.0f;

    outConfig->wallTempRelaxRate = 0.01f;        // Halved terrain block-to-gas thermal relaxation (v1.0.2)
    outConfig->evapRateCoeff = 0.18f;
    outConfig->evapWindFactor = 0.35f;
    outConfig->evapCoolingScale = 0.75f;
    outConfig->latentHeatScale = 1.0f;

    outConfig->autoconvertThresh = 0.0008f;      // 0.8 g/kg cloud water -> rain
    outConfig->autoconvertRate = 0.14f;
    outConfig->freezeRate = 0.35f;
    outConfig->meltRate = 0.25f;
    outConfig->rainFallSpeed = 8.0f;
    outConfig->snowFallSpeed = 1.5f;
    outConfig->snowDepositRate = 0.9f;

    outConfig->sorIterations = 18;
    outConfig->sorOmega = 1.80f;
    outConfig->pressureWarmStart = 0.90f;
    outConfig->useMacCormack = 1;
    outConfig->initialRelHumidity = 0.62f;

    outConfig->bottomTempBoostC = 0.0f;
    outConfig->topTempDropC = 0.0f;
    outConfig->bottomMoistureSource = 0.0f;
    outConfig->ambientHumidityTarget = 0.0f;
    outConfig->ambientHumidityRate = 0.0f;

    outConfig->subdivision = 2;
    outConfig->evapMassPerWaterTile = 0.70f;     // 5x tile mass threshold -> 1/5 water-block cycle rate (v1.0.2)
    outConfig->rainMassPerWaterTile = 0.70f;     // Strict 1:1 physical water mass conservation!
}

WEATHER_API WeatherSimHandle* WeatherSim_Create(const WeatherSimConfig* config) {
    if (!config || config->width < 8 || config->height < 8) return nullptr;

    WeatherSimHandle* h = new WeatherSimHandle();
    h->cfg = *config;

    size_t W = (size_t)config->width;
    size_t H = (size_t)config->height;
    size_t N = W * H;
    size_t Nu = (W + 1) * H;
    size_t Nv = W * (H + 1);

    cudaError_t err = cudaStreamCreate(&h->stream);
    if (err == cudaSuccess) err = cudaEventCreate(&h->evStart);
    if (err == cudaSuccess) err = cudaEventCreate(&h->evStop);

    auto check = [&](cudaError_t e) {
        if (err == cudaSuccess && e != cudaSuccess) err = e;
    };

    uint64_t& bytes = h->allocatedVramBytes;
    check(allocBuf(&h->d.cellType, N, bytes));
    check(allocBuf(&h->d.heatSource, N, bytes));
    check(allocBuf(&h->d.moistureSource, N, bytes));
    check(allocBuf(&h->d.dynHeat, N, bytes));
    check(allocBuf(&h->d.windForceX, N, bytes));
    check(allocBuf(&h->d.windForceY, N, bytes));

    check(allocBuf(&h->d.u, Nu, bytes));
    check(allocBuf(&h->d.uTmp, Nu, bytes));
    check(allocBuf(&h->d.uHat, Nu, bytes));
    check(allocBuf(&h->d.v, Nv, bytes));
    check(allocBuf(&h->d.vTmp, Nv, bytes));
    check(allocBuf(&h->d.vHat, Nv, bytes));

    check(allocBuf(&h->d.p, N, bytes));
    check(allocBuf(&h->d.div, N, bytes));
    check(allocBuf(&h->d.vort, N, bytes));

    // Build Geometric Multigrid pyramid down to coarsest grid <= 16-32 cells
    {
        int32_t curW = config->width;
        int32_t curH = config->height;
        h->d.mgNumLevels = 0;

        while (h->d.mgNumLevels < kMaxMgLevels) {
            int lvl = h->d.mgNumLevels;
            WeatherMgLevel& L = h->d.mgLevels[lvl];
            L.w = curW;
            L.h = curH;
            size_t lvlN = (size_t)curW * (size_t)curH;
            if (lvl == 0) {
                L.p = h->d.p;
                L.rhs = h->d.div;
            } else {
                check(allocBuf(&L.p, lvlN, bytes));
                check(allocBuf(&L.rhs, lvlN, bytes));
            }
            h->d.mgNumLevels++;

            int32_t nextW = (curW + 1) / 2;
            int32_t nextH = (curH + 1) / 2;
            if (h->d.mgNumLevels >= kMaxMgLevels || nextW < 8 || nextH < 8 || std::max(curW, curH) <= 16) {
                break;
            }
            check(allocBuf(&L.res, lvlN, bytes));
            curW = nextW;
            curH = nextH;
        }
    }

    check(allocBuf(&h->d.T, N, bytes));
    check(allocBuf(&h->d.TTmp, N, bytes));
    check(allocBuf(&h->d.THat, N, bytes));

    check(allocBuf(&h->d.qv, N, bytes));
    check(allocBuf(&h->d.qvTmp, N, bytes));
    check(allocBuf(&h->d.qvHat, N, bytes));

    check(allocBuf(&h->d.qc, N, bytes));
    check(allocBuf(&h->d.qcTmp, N, bytes));
    check(allocBuf(&h->d.qcHat, N, bytes));

    check(allocBuf(&h->d.qr, N, bytes));
    check(allocBuf(&h->d.qrTmp, N, bytes));
    check(allocBuf(&h->d.qi, N, bytes));
    check(allocBuf(&h->d.qiTmp, N, bytes));

    check(allocBuf(&h->d.snowAccum, N, bytes));
    check(allocBuf(&h->d.evapAccum, N, bytes));
    check(allocBuf(&h->d.rainAccum, N, bytes));
    check(allocBuf(&h->d.latentRate, N, bytes));
    check(allocBuf(&h->d.scratchScalar, N, bytes));

    h->d.fluidEventCapacity = 4096;
    check(allocBuf(&h->d.d_fluidEvents, (size_t)h->d.fluidEventCapacity, bytes));
    check(allocBuf(&h->d.d_fluidEventCount, 1, bytes));

    if (err != cudaSuccess) {
        destroyAllBuffers(h);
        delete h;
        return nullptr;
    }

    h->lastCudaError = (int32_t)launchResetFields(h);
    cudaStreamSynchronize(h->stream);
    return h;
}

WEATHER_API void WeatherSim_Destroy(WeatherSimHandle* handle) {
    if (!handle) return;
    if (handle->stream) cudaStreamSynchronize(handle->stream);
    destroyAllBuffers(handle);
    delete handle;
}

WEATHER_API void WeatherSim_UpdateConfig(WeatherSimHandle* handle, const WeatherSimConfig* config) {
    if (!handle || !config) return;
    int32_t oldW = handle->cfg.width;
    int32_t oldH = handle->cfg.height;
    handle->cfg = *config;
    handle->cfg.width = oldW;
    handle->cfg.height = oldH;
}

WEATHER_API void WeatherSim_SetMidLayerTemp(WeatherSimHandle* handle, float midLayerTempC) {
    if (!handle) return;
    handle->cfg.midLayerTempC = midLayerTempC;
}

WEATHER_API int32_t WeatherSim_ResetFields(WeatherSimHandle* handle) {
    if (!handle) return -1;
    cudaError_t err = launchResetFields(handle);
    if (err == cudaSuccess) err = cudaStreamSynchronize(handle->stream);
    handle->stepCount = 0;
    handle->lastCudaError = (int32_t)err;
    return (int32_t)err;
}

WEATHER_API int32_t WeatherSim_ApplyInitialHumidityGradient(
    WeatherSimHandle* handle,
    float bottomRh,
    float topRh,
    float yBottomFrac,
    float yTopFrac)
{
    if (!handle) return -1;
    cudaError_t err = launchApplyInitialHumidityGradient(handle, bottomRh, topRh, yBottomFrac, yTopFrac);
    if (err == cudaSuccess) err = cudaStreamSynchronize(handle->stream);
    handle->lastCudaError = (int32_t)err;
    return (int32_t)err;
}

WEATHER_API int32_t WeatherSim_ApplyInitialPerlinNoise(
    WeatherSimHandle* handle,
    int32_t seed,
    float tempAmpC,
    float rhAmp,
    float wavelengthTiles,
    int32_t octaves)
{
    if (!handle) return -1;
    cudaError_t err = launchApplyInitialPerlinNoise(handle, seed, tempAmpC, rhAmp, wavelengthTiles, octaves);
    if (err == cudaSuccess) err = cudaStreamSynchronize(handle->stream);
    handle->lastCudaError = (int32_t)err;
    return (int32_t)err;
}

WEATHER_API int32_t WeatherSim_UploadMoistureSources(WeatherSimHandle* handle, const float* moistureRates) {
    if (!handle) return -1;
    size_t N = (size_t)handle->cfg.width * (size_t)handle->cfg.height;
    cudaError_t err;
    if (moistureRates) {
        err = cudaMemcpyAsync(handle->d.moistureSource, moistureRates, N * sizeof(float), cudaMemcpyHostToDevice, handle->stream);
    } else {
        err = cudaMemsetAsync(handle->d.moistureSource, 0, N * sizeof(float), handle->stream);
    }
    if (err == cudaSuccess) err = cudaStreamSynchronize(handle->stream);
    handle->lastCudaError = (int32_t)err;
    return (int32_t)err;
}

WEATHER_API int32_t WeatherSim_UploadCellsFull(WeatherSimHandle* handle, const uint8_t* cellTypes) {
    if (!handle || !cellTypes) return -1;
    size_t N = (size_t)handle->cfg.width * (size_t)handle->cfg.height;
    cudaError_t err = cudaMemcpyAsync(handle->d.cellType, cellTypes, N * sizeof(uint8_t), cudaMemcpyHostToDevice, handle->stream);
    if (err == cudaSuccess) err = cudaStreamSynchronize(handle->stream);
    handle->lastCudaError = (int32_t)err;
    return (int32_t)err;
}

WEATHER_API int32_t WeatherSim_UploadCellsPatch(
    WeatherSimHandle* handle,
    const int32_t* xs,
    const int32_t* ys,
    const uint8_t* types,
    int32_t count)
{
    if (!handle || count <= 0) return 0;
    if (!xs || !ys || !types) return -1;

    if (count > handle->d.patchCapacity) {
        freeBuf(handle->d.d_patchX);
        freeBuf(handle->d.d_patchY);
        freeBuf(handle->d.d_patchType);
        uint64_t dummy = 0;
        cudaError_t e1 = allocBuf(&handle->d.d_patchX, count, dummy);
        cudaError_t e2 = allocBuf(&handle->d.d_patchY, count, dummy);
        cudaError_t e3 = allocBuf(&handle->d.d_patchType, count, dummy);
        if (e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess) return -2;
        handle->d.patchCapacity = count;
    }

    cudaError_t err = cudaMemcpyAsync(handle->d.d_patchX, xs, count * sizeof(int32_t), cudaMemcpyHostToDevice, handle->stream);
    if (err == cudaSuccess) err = cudaMemcpyAsync(handle->d.d_patchY, ys, count * sizeof(int32_t), cudaMemcpyHostToDevice, handle->stream);
    if (err == cudaSuccess) err = cudaMemcpyAsync(handle->d.d_patchType, types, count * sizeof(uint8_t), cudaMemcpyHostToDevice, handle->stream);
    if (err == cudaSuccess) err = launchPatchCells(handle, count);
    handle->lastCudaError = (int32_t)err;
    return (int32_t)err;
}

WEATHER_API int32_t WeatherSim_UploadHeatSources(WeatherSimHandle* handle, const float* heatRateCPerSec) {
    if (!handle) return -1;
    size_t N = (size_t)handle->cfg.width * (size_t)handle->cfg.height;
    cudaError_t err;
    if (heatRateCPerSec) {
        err = cudaMemcpyAsync(handle->d.heatSource, heatRateCPerSec, N * sizeof(float), cudaMemcpyHostToDevice, handle->stream);
    } else {
        err = cudaMemsetAsync(handle->d.heatSource, 0, N * sizeof(float), handle->stream);
    }
    if (err == cudaSuccess) err = cudaStreamSynchronize(handle->stream);
    handle->lastCudaError = (int32_t)err;
    return (int32_t)err;
}

WEATHER_API int32_t WeatherSim_UploadStaticHeatPatch(
    WeatherSimHandle* handle,
    const int32_t* xs,
    const int32_t* ys,
    const float* rates,
    int32_t count)
{
    if (!handle || count <= 0) return 0;
    if (!xs || !ys || !rates) return -1;

    if (count > handle->d.heatPatchCapacity) {
        freeBuf(handle->d.d_heatPatchX);
        freeBuf(handle->d.d_heatPatchY);
        freeBuf(handle->d.d_heatPatchVal);
        uint64_t dummy = 0;
        cudaError_t e1 = allocBuf(&handle->d.d_heatPatchX, count, dummy);
        cudaError_t e2 = allocBuf(&handle->d.d_heatPatchY, count, dummy);
        cudaError_t e3 = allocBuf(&handle->d.d_heatPatchVal, count, dummy);
        if (e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess) return -2;
        handle->d.heatPatchCapacity = count;
    }

    cudaError_t err = cudaMemcpyAsync(handle->d.d_heatPatchX, xs, count * sizeof(int32_t), cudaMemcpyHostToDevice, handle->stream);
    if (err == cudaSuccess) err = cudaMemcpyAsync(handle->d.d_heatPatchY, ys, count * sizeof(int32_t), cudaMemcpyHostToDevice, handle->stream);
    if (err == cudaSuccess) err = cudaMemcpyAsync(handle->d.d_heatPatchVal, rates, count * sizeof(float), cudaMemcpyHostToDevice, handle->stream);
    if (err == cudaSuccess) err = launchPatchStaticHeat(handle, count);
    handle->lastCudaError = (int32_t)err;
    return (int32_t)err;
}

WEATHER_API int32_t WeatherSim_UploadDynamicHeaters(
    WeatherSimHandle* handle,
    const WeatherDynamicHeatSource* sources,
    int32_t count)
{
    if (!handle) return -1;
    if (count < 0) count = 0;
    if (count > 256) count = 256;

    if (count > 0 && !sources) return -1;

    if (count > handle->d.dynSourceCapacity) {
        freeBuf(handle->d.d_dynSources);
        uint64_t dummy = 0;
        cudaError_t e = allocBuf(&handle->d.d_dynSources, (size_t)count, dummy);
        if (e != cudaSuccess) return -2;
        handle->d.dynSourceCapacity = count;
    }

    cudaError_t err = cudaSuccess;
    if (count > 0) {
        err = cudaMemcpyAsync(
            handle->d.d_dynSources, sources,
            (size_t)count * sizeof(WeatherDynamicHeatSource),
            cudaMemcpyHostToDevice, handle->stream);
    }
    if (err == cudaSuccess) {
        err = launchStampDynamicHeaters(handle, count);
    }
    handle->lastCudaError = (int32_t)err;
    return (int32_t)err;
}

WEATHER_API int32_t WeatherSim_UploadWindSources(
    WeatherSimHandle* handle,
    const float* forceX,
    const float* forceY)
{
    if (!handle) return -1;
    size_t N = (size_t)handle->cfg.width * (size_t)handle->cfg.height;
    cudaError_t err = cudaSuccess;
    if (forceX) err = cudaMemcpyAsync(handle->d.windForceX, forceX, N * sizeof(float), cudaMemcpyHostToDevice, handle->stream);
    else        err = cudaMemsetAsync(handle->d.windForceX, 0, N * sizeof(float), handle->stream);

    if (err == cudaSuccess) {
        if (forceY) err = cudaMemcpyAsync(handle->d.windForceY, forceY, N * sizeof(float), cudaMemcpyHostToDevice, handle->stream);
        else        err = cudaMemsetAsync(handle->d.windForceY, 0, N * sizeof(float), handle->stream);
    }
    if (err == cudaSuccess) err = cudaStreamSynchronize(handle->stream);
    handle->lastCudaError = (int32_t)err;
    return (int32_t)err;
}

WEATHER_API int32_t WeatherSim_Step(WeatherSimHandle* handle, float dt) {
    if (!handle || dt <= 0.0f) return -1;
    dt = std::min(dt, 0.1f);

    cudaEventRecord(handle->evStart, handle->stream);
    cudaError_t err = launchStepSimulation(handle, dt);
    cudaEventRecord(handle->evStop, handle->stream);
    if (err == cudaSuccess) {
        err = cudaEventSynchronize(handle->evStop);
        if (err == cudaSuccess) {
            cudaEventElapsedTime(&handle->lastStepGpuMs, handle->evStart, handle->evStop);
        }
    }
    handle->stepCount++;
    handle->lastCudaError = (int32_t)err;
    return (int32_t)err;
}

WEATHER_API int32_t WeatherSim_SamplePoints(
    WeatherSimHandle* handle,
    const float* cx,
    const float* cy,
    WeatherEntitySample* outSamples,
    int32_t count)
{
    if (!handle || count <= 0) return 0;
    if (!cx || !cy || !outSamples) return -1;

    if (count > handle->d.sampleCapacity) {
        freeBuf(handle->d.d_sampleX);
        freeBuf(handle->d.d_sampleY);
        freeBuf(handle->d.d_sampleOut);
        uint64_t dummy = 0;
        cudaError_t e1 = allocBuf(&handle->d.d_sampleX, count, dummy);
        cudaError_t e2 = allocBuf(&handle->d.d_sampleY, count, dummy);
        cudaError_t e3 = allocBuf(&handle->d.d_sampleOut, count, dummy);
        if (e1 != cudaSuccess || e2 != cudaSuccess || e3 != cudaSuccess) return -2;
        handle->d.sampleCapacity = count;
    }

    cudaError_t err = cudaMemcpyAsync(handle->d.d_sampleX, cx, count * sizeof(float), cudaMemcpyHostToDevice, handle->stream);
    if (err == cudaSuccess) err = cudaMemcpyAsync(handle->d.d_sampleY, cy, count * sizeof(float), cudaMemcpyHostToDevice, handle->stream);
    if (err == cudaSuccess) err = launchSamplePoints(handle, count);
    if (err == cudaSuccess) err = cudaMemcpyAsync(outSamples, handle->d.d_sampleOut, count * sizeof(WeatherEntitySample), cudaMemcpyDeviceToHost, handle->stream);
    if (err == cudaSuccess) err = cudaStreamSynchronize(handle->stream);
    handle->lastCudaError = (int32_t)err;
    return (int32_t)err;
}

WEATHER_API int32_t WeatherSim_CopyField(
    WeatherSimHandle* handle,
    int32_t fieldId,
    float* outHostBuffer)
{
    if (!handle || !outHostBuffer) return -1;
    const float* dPtr = nullptr;
    cudaError_t err = launchPrepareFieldForCopy(handle, fieldId, &dPtr);
    if (err == cudaSuccess && dPtr) {
        size_t bytes = (size_t)handle->cfg.width * (size_t)handle->cfg.height * sizeof(float);
        err = cudaMemcpyAsync(outHostBuffer, dPtr, bytes, cudaMemcpyDeviceToHost, handle->stream);
        if (err == cudaSuccess) err = cudaStreamSynchronize(handle->stream);
    }
    handle->lastCudaError = (int32_t)err;
    return (int32_t)err;
}

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
    uint8_t* outRgbaBytes)
{
    if (!handle || !outRgbaBytes || winW <= 0 || winH <= 0) return -1;
    int32_t bytesNeeded = winW * winH * 4;
    if (bytesNeeded > handle->d.rgbaCapacity) {
        freeBuf(handle->d.d_rgbaWindow);
        uint64_t dummy = 0;
        cudaError_t e = allocBuf(&handle->d.d_rgbaWindow, (size_t)bytesNeeded, dummy);
        if (e != cudaSuccess) return -2;
        handle->d.rgbaCapacity = bytesNeeded;
    }

    cudaError_t err = launchPackViewWindowRGBA8(
        handle, x0, y0, winW, winH, mode,
        playerCx, playerCy,
        playerClearInnerCells, playerClearOuterCells,
        playerMinFogFactor, wallFogMultiplier);
    if (err == cudaSuccess) {
        err = cudaMemcpyAsync(outRgbaBytes, handle->d.d_rgbaWindow, (size_t)bytesNeeded, cudaMemcpyDeviceToHost, handle->stream);
        if (err == cudaSuccess) err = cudaStreamSynchronize(handle->stream);
    }
    handle->lastCudaError = (int32_t)err;
    return (int32_t)err;
}

WEATHER_API int32_t WeatherSim_ReadViewPrecipGrid(
    WeatherSimHandle* handle,
    int32_t x0,
    int32_t y0,
    int32_t winW,
    int32_t winH,
    int32_t outW,
    int32_t outH,
    WeatherPrecipCell* outCells)
{
    if (!handle || !outCells || winW <= 0 || winH <= 0 || outW <= 0 || outH <= 0) return -1;
    int32_t count = outW * outH;
    if (count > handle->d.precipCapacity) {
        freeBuf(handle->d.d_precipGrid);
        uint64_t dummy = 0;
        cudaError_t e = allocBuf(&handle->d.d_precipGrid, (size_t)count, dummy);
        if (e != cudaSuccess) return -2;
        handle->d.precipCapacity = count;
    }

    cudaError_t err = launchSampleViewPrecipGrid(handle, x0, y0, winW, winH, outW, outH);
    if (err == cudaSuccess) {
        err = cudaMemcpyAsync(outCells, handle->d.d_precipGrid, (size_t)count * sizeof(WeatherPrecipCell), cudaMemcpyDeviceToHost, handle->stream);
        if (err == cudaSuccess) err = cudaStreamSynchronize(handle->stream);
    }
    handle->lastCudaError = (int32_t)err;
    return (int32_t)err;
}

WEATHER_API int32_t WeatherSim_RenderFullMapRGBA8(
    WeatherSimHandle* handle,
    int32_t mapW,
    int32_t mapH,
    int32_t layerMode,
    int32_t showWalls,
    uint8_t* outRgbaBytes)
{
    if (!handle || !outRgbaBytes || mapW <= 0 || mapH <= 0) return -1;
    int32_t bytesNeeded = mapW * mapH * 4;
    if (bytesNeeded > handle->d.fullMapRgbaCapacity) {
        freeBuf(handle->d.d_fullMapRgba);
        uint64_t dummy = 0;
        cudaError_t e = allocBuf(&handle->d.d_fullMapRgba, (size_t)bytesNeeded, dummy);
        if (e != cudaSuccess) return -2;
        handle->d.fullMapRgbaCapacity = bytesNeeded;
    }

    cudaError_t err = launchRenderFullMapRGBA8(handle, mapW, mapH, layerMode, showWalls);
    if (err == cudaSuccess) {
        err = cudaMemcpyAsync(outRgbaBytes, handle->d.d_fullMapRgba, (size_t)bytesNeeded, cudaMemcpyDeviceToHost, handle->stream);
        if (err == cudaSuccess) err = cudaStreamSynchronize(handle->stream);
    }
    handle->lastCudaError = (int32_t)err;
    return (int32_t)err;
}

WEATHER_API int32_t WeatherSim_SampleMapWindGrid(
    WeatherSimHandle* handle,
    int32_t gridW,
    int32_t gridH,
    float* outU,
    float* outV)
{
    if (!handle || !outU || !outV || gridW <= 0 || gridH <= 0) return -1;
    int32_t count = gridW * gridH;
    if (count > handle->d.windGridCapacity) {
        freeBuf(handle->d.d_windGridU);
        freeBuf(handle->d.d_windGridV);
        uint64_t dummy = 0;
        cudaError_t e1 = allocBuf(&handle->d.d_windGridU, (size_t)count, dummy);
        cudaError_t e2 = allocBuf(&handle->d.d_windGridV, (size_t)count, dummy);
        if (e1 != cudaSuccess || e2 != cudaSuccess) return -2;
        handle->d.windGridCapacity = count;
    }

    cudaError_t err = launchSampleMapWindGrid(handle, gridW, gridH);
    if (err == cudaSuccess) {
        err = cudaMemcpyAsync(outU, handle->d.d_windGridU, (size_t)count * sizeof(float), cudaMemcpyDeviceToHost, handle->stream);
    }
    if (err == cudaSuccess) {
        err = cudaMemcpyAsync(outV, handle->d.d_windGridV, (size_t)count * sizeof(float), cudaMemcpyDeviceToHost, handle->stream);
    }
    if (err == cudaSuccess) err = cudaStreamSynchronize(handle->stream);
    handle->lastCudaError = (int32_t)err;
    return (int32_t)err;
}

WEATHER_API int32_t WeatherSim_PopFluidEvents(
    WeatherSimHandle* handle,
    WeatherFluidEvent* outEvents,
    int32_t maxEvents)
{
    if (!handle || !outEvents || maxEvents <= 0) return 0;
    if (!handle->d.d_fluidEvents || !handle->d.d_fluidEventCount) return 0;

    int32_t gpuCount = 0;
    cudaError_t err = cudaMemcpyAsync(&gpuCount, handle->d.d_fluidEventCount, sizeof(int32_t), cudaMemcpyDeviceToHost, handle->stream);
    if (err == cudaSuccess) err = cudaStreamSynchronize(handle->stream);
    if (err != cudaSuccess) {
        handle->lastCudaError = (int32_t)err;
        return -1;
    }

    int32_t toCopy = std::min(gpuCount, std::min(maxEvents, handle->d.fluidEventCapacity));
    if (toCopy > 0) {
        err = cudaMemcpyAsync(outEvents, handle->d.d_fluidEvents, (size_t)toCopy * sizeof(WeatherFluidEvent), cudaMemcpyDeviceToHost, handle->stream);
    }
    if (gpuCount > 0 && err == cudaSuccess) {
        err = cudaMemsetAsync(handle->d.d_fluidEventCount, 0, sizeof(int32_t), handle->stream);
    }
    if (err == cudaSuccess) err = cudaStreamSynchronize(handle->stream);
    handle->lastCudaError = (int32_t)err;
    return (err == cudaSuccess) ? toCopy : -1;
}

WEATHER_API int32_t WeatherSim_GetStats(WeatherSimHandle* handle, WeatherSimStats* outStats) {
    if (!handle || !outStats) return -1;
    size_t freeB = 0, totalB = 0;
    cudaMemGetInfo(&freeB, &totalB);
    outStats->lastStepGpuMs = handle->lastStepGpuMs;
    outStats->allocatedVramBytes = handle->allocatedVramBytes;
    outStats->freeVramBytes = (uint64_t)freeB;
    outStats->totalVramBytes = (uint64_t)totalB;
    outStats->stepCount = handle->stepCount;
    outStats->lastCudaError = handle->lastCudaError;
    return 0;
}

} // extern "C"
