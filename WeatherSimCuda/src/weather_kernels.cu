#include "weather_sim_internal.cuh"
#include <math.h>

namespace {

constexpr float kLvOverCp = 2491.0f; // K per (kg/kg) vaporization
constexpr float kLsOverCp = 2824.0f; // K per (kg/kg) sublimation/deposition
constexpr float kLfOverCp = 333.0f;  // K per (kg/kg) fusion
constexpr float kRefPressureHPa = 1013.25f;

__device__ __forceinline__ int cellIdx(int i, int j, int W) {
    return j * W + i;
}

__device__ __forceinline__ int uIdx(int i, int j, int W) {
    return j * (W + 1) + i;
}

__device__ __forceinline__ int vIdx(int i, int j, int W) {
    return j * W + i;
}

__device__ __forceinline__ float smoothstep01(float x) {
    x = fminf(fmaxf(x, 0.0f), 1.0f);
    return x * x * (3.0f - 2.0f * x);
}

__device__ __forceinline__ float lowerZoneWeight(float yCell, int height) {
    float yn = yCell / fmaxf((float)(height - 1), 1.0f);
    return smoothstep01((0.45f - yn) / 0.45f);
}

__device__ __forceinline__ float upperZoneWeight(float yCell, int height) {
    float yn = yCell / fmaxf((float)(height - 1), 1.0f);
    return smoothstep01((yn - 0.55f) / 0.45f);
}

__device__ __forceinline__ float envTempAtY(float yCell, const WeatherSimConfig& cfg) {
    float yMid = 0.5f * (float)(cfg.height - 1);
    float baseT = cfg.midLayerTempC - cfg.bgLapseRateCPerTile * (yCell - yMid) * cfg.cellSizeTiles;
    float wBot = lowerZoneWeight(yCell, cfg.height);
    float wTop = upperZoneWeight(yCell, cfg.height);
    return baseT + cfg.bottomTempBoostC * wBot - cfg.topTempDropC * wTop;
}

__device__ __forceinline__ void satWater(float tempC, float& qsw, float& dqsw_dT) {
    float t = fminf(fmaxf(tempC, -50.0f), 60.0f);
    float denom = t + 237.3f;
    float es = 6.1078f * expf(17.27f * t / denom);
    float des_dT = es * (17.27f * 237.3f) / (denom * denom);
    float pMinusE = fmaxf(kRefPressureHPa - es, 200.0f);
    qsw = 0.622f * es / pMinusE;
    dqsw_dT = 0.622f * kRefPressureHPa * des_dT / (pMinusE * pMinusE);
}

__device__ __forceinline__ void satIce(float tempC, float& qsi, float& dqsi_dT) {
    float t = fminf(fmaxf(tempC, -60.0f), 20.0f);
    float denom = t + 265.5f;
    float es = 6.1078f * expf(21.875f * t / denom);
    float des_dT = es * (21.875f * 265.5f) / (denom * denom);
    float pMinusE = fmaxf(kRefPressureHPa - es, 200.0f);
    qsi = 0.622f * es / pMinusE;
    dqsi_dT = 0.622f * kRefPressureHPa * des_dT / (pMinusE * pMinusE);
}

__device__ __forceinline__ float sampleU(const float* __restrict__ u, float x, float y, int W, int H) {
    float gx = fminf(fmaxf(x, 0.0f), (float)W);
    float gy = fminf(fmaxf(y - 0.5f, 0.0f), (float)(H - 1));
    int i0 = (int)floorf(gx);
    int j0 = (int)floorf(gy);
    int i1 = min(i0 + 1, W);
    int j1 = min(j0 + 1, H - 1);
    float fx = gx - (float)i0;
    float fy = gy - (float)j0;
    float a = u[uIdx(i0, j0, W)] * (1.0f - fx) + u[uIdx(i1, j0, W)] * fx;
    float b = u[uIdx(i0, j1, W)] * (1.0f - fx) + u[uIdx(i1, j1, W)] * fx;
    return a * (1.0f - fy) + b * fy;
}

__device__ __forceinline__ float sampleV(const float* __restrict__ v, float x, float y, int W, int H) {
    float gx = fminf(fmaxf(x - 0.5f, 0.0f), (float)(W - 1));
    float gy = fminf(fmaxf(y, 0.0f), (float)H);
    int i0 = (int)floorf(gx);
    int j0 = (int)floorf(gy);
    int i1 = min(i0 + 1, W - 1);
    int j1 = min(j0 + 1, H);
    float fx = gx - (float)i0;
    float fy = gy - (float)j0;
    float a = v[vIdx(i0, j0, W)] * (1.0f - fx) + v[vIdx(i1, j0, W)] * fx;
    float b = v[vIdx(i0, j1, W)] * (1.0f - fx) + v[vIdx(i1, j1, W)] * fx;
    return a * (1.0f - fy) + b * fy;
}

__device__ __forceinline__ float sampleScalar(
    const float* __restrict__ field,
    float x, float y,
    int W, int H,
    float& outMin,
    float& outMax)
{
    float gx = fminf(fmaxf(x - 0.5f, 0.0f), (float)(W - 1));
    float gy = fminf(fmaxf(y - 0.5f, 0.0f), (float)(H - 1));
    int i0 = (int)floorf(gx);
    int j0 = (int)floorf(gy);
    int i1 = min(i0 + 1, W - 1);
    int j1 = min(j0 + 1, H - 1);
    float fx = gx - (float)i0;
    float fy = gy - (float)j0;

    float v00 = field[cellIdx(i0, j0, W)];
    float v10 = field[cellIdx(i1, j0, W)];
    float v01 = field[cellIdx(i0, j1, W)];
    float v11 = field[cellIdx(i1, j1, W)];

    outMin = fminf(fminf(v00, v10), fminf(v01, v11));
    outMax = fmaxf(fmaxf(v00, v10), fmaxf(v01, v11));

    float a = v00 * (1.0f - fx) + v10 * fx;
    float b = v01 * (1.0f - fx) + v11 * fx;
    return a * (1.0f - fy) + b * fy;
}

// RK2 trace clamped strictly to outer simulation domain edges (internal walls do not block airflow)
__device__ __forceinline__ void tracePoint(
    const float* __restrict__ u,
    const float* __restrict__ v,
    float x, float y,
    float dtOverDx,
    float extraVyCells,
    int W, int H,
    float& outX, float& outY)
{
    float u1 = sampleU(u, x, y, W, H);
    float v1 = sampleV(v, x, y, W, H) + extraVyCells;
    float xm = fminf(fmaxf(x - 0.5f * dtOverDx * u1, 0.5f), (float)W - 0.5f);
    float ym = fminf(fmaxf(y - 0.5f * dtOverDx * v1, 0.5f), (float)H - 0.5f);

    float u2 = sampleU(u, xm, ym, W, H);
    float v2 = sampleV(v, xm, ym, W, H) + extraVyCells;
    outX = fminf(fmaxf(x - dtOverDx * u2, 0.5f), (float)W - 0.5f);
    outY = fminf(fmaxf(y - dtOverDx * v2, 0.5f), (float)H - 0.5f);
}

// ============================================================================
// Kernels
// ============================================================================

__global__ void kResetFields(
    WeatherSimConfig cfg,
    float* __restrict__ T,
    float* __restrict__ qv,
    float* __restrict__ qc,
    float* __restrict__ qr,
    float* __restrict__ qi,
    float* __restrict__ p,
    float* __restrict__ div,
    float* __restrict__ snowAccum,
    float* __restrict__ evapAccum,
    float* __restrict__ rainAccum,
    float* __restrict__ latentRate)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= cfg.width || j >= cfg.height) return;
    int id = cellIdx(i, j, cfg.width);
    float temp = envTempAtY((float)j, cfg);
    T[id] = temp;
    float qsw, dq, qsi, dqi;
    satWater(temp, qsw, dq);
    satIce(temp, qsi, dqi);
    float qs = (temp <= 0.0f) ? qsi : qsw;
    qv[id] = qs * cfg.initialRelHumidity;
    qc[id] = 0.0f;
    qr[id] = 0.0f;
    qi[id] = 0.0f;
    p[id] = 0.0f;
    div[id] = 0.0f;
    snowAccum[id] = 0.0f;
    evapAccum[id] = 0.0f;
    rainAccum[id] = 0.0f;
    latentRate[id] = 0.0f;
}

__device__ __forceinline__ float perlinFade(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

__device__ __forceinline__ uint32_t perlinHash(int32_t x, int32_t y, uint32_t seed) {
    uint32_t h = (uint32_t)x * 374761393u + (uint32_t)y * 668265263u + seed * 1442695041u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return h ^ (h >> 16);
}

__device__ __forceinline__ float perlinGrad(int32_t ix, int32_t iy, float dx, float dy, uint32_t seed) {
    uint32_t h = perlinHash(ix, iy, seed) & 7u;
    // 8 unit-ish gradient directions
    const float invSqrt2 = 0.70710678f;
    switch (h) {
        case 0u: return  dx;
        case 1u: return -dx;
        case 2u: return  dy;
        case 3u: return -dy;
        case 4u: return ( dx + dy) * invSqrt2;
        case 5u: return (-dx + dy) * invSqrt2;
        case 6u: return ( dx - dy) * invSqrt2;
        default: return (-dx - dy) * invSqrt2;
    }
}

__device__ __forceinline__ float perlin2D(float x, float y, uint32_t seed) {
    int32_t x0 = (int32_t)floorf(x);
    int32_t y0 = (int32_t)floorf(y);
    float fx = x - (float)x0;
    float fy = y - (float)y0;

    float u = perlinFade(fx);
    float v = perlinFade(fy);

    float n00 = perlinGrad(x0,     y0,     fx,        fy,        seed);
    float n10 = perlinGrad(x0 + 1, y0,     fx - 1.0f, fy,        seed);
    float n01 = perlinGrad(x0,     y0 + 1, fx,        fy - 1.0f, seed);
    float n11 = perlinGrad(x0 + 1, y0 + 1, fx - 1.0f, fy - 1.0f, seed);

    float nx0 = n00 + u * (n10 - n00);
    float nx1 = n01 + u * (n11 - n01);
    return nx0 + v * (nx1 - nx0);
}

__device__ __forceinline__ float fbmPerlin2D(float x, float y, uint32_t seed, int32_t octaves) {
    float sum = 0.0f;
    float amp = 1.0f;
    float norm = 0.0f;
    float freq = 1.0f;
    for (int32_t o = 0; o < octaves; ++o) {
        sum += amp * perlin2D(x * freq, y * freq, seed + (uint32_t)o * 1013904223u);
        norm += amp;
        amp *= 0.5f;
        freq *= 2.0f;
    }
    // Scale up slightly because 2D Perlin peaks near ~0.707
    float val = (norm > 1e-6f) ? (sum / norm) * 1.414f : 0.0f;
    return fminf(fmaxf(val, -1.0f), 1.0f);
}

__global__ void kApplyInitialHumidityGradient(
    WeatherSimConfig cfg,
    float bottomRh,
    float topRh,
    float yBottomFrac,
    float yTopFrac,
    const float* __restrict__ T,
    float* __restrict__ qv)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= cfg.width || j >= cfg.height) return;

    int id = cellIdx(i, j, cfg.width);
    float yn = (float)j / fmaxf((float)(cfg.height - 1), 1.0f);
    float span = fmaxf(yTopFrac - yBottomFrac, 1e-4f);
    float t = smoothstep01((yn - yBottomFrac) / span);
    float rh = fminf(fmaxf(bottomRh + (topRh - bottomRh) * t, 0.05f), 1.02f);

    float temp = T[id];
    float qsw, dq, qsi, dqi;
    satWater(temp, qsw, dq);
    satIce(temp, qsi, dqi);
    float qs = (temp <= 0.0f) ? qsi : qsw;
    qv[id] = qs * rh;
}

__global__ void kApplyInitialPerlinNoise(
    WeatherSimConfig cfg,
    uint32_t seed,
    float tempAmpC,
    float rhAmp,
    float invWavelength,
    int32_t octaves,
    float* __restrict__ T,
    float* __restrict__ qv)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= cfg.width || j >= cfg.height) return;

    int id = cellIdx(i, j, cfg.width);
    float oldT = T[id];
    float oldQsw, oldDq, oldQsi, oldDqi;
    satWater(oldT, oldQsw, oldDq);
    satIce(oldT, oldQsi, oldDqi);
    float oldQs = (oldT <= 0.0f) ? oldQsi : oldQsw;
    float baseRh = (oldQs > 1e-7f) ? (qv[id] / oldQs) : cfg.initialRelHumidity;

    float tx = (float)i * cfg.cellSizeTiles * invWavelength;
    float ty = (float)j * cfg.cellSizeTiles * invWavelength;

    float nT = (tempAmpC != 0.0f) ? fbmPerlin2D(tx, ty, seed, octaves) : 0.0f;
    float nRH = (rhAmp != 0.0f) ? fbmPerlin2D(tx + 137.3f, ty + 291.7f, seed ^ 0x517cc1b7u, octaves) : 0.0f;

    float newT = fminf(fmaxf(oldT + nT * tempAmpC, -45.0f), 55.0f);
    T[id] = newT;

    float qsw, dq, qsi, dqi;
    satWater(newT, qsw, dq);
    satIce(newT, qsi, dqi);
    float qs = (newT <= 0.0f) ? qsi : qsw;

    float targetRh = (rhAmp != 0.0f)
        ? fminf(fmaxf(baseRh + nRH * rhAmp, 0.08f), 1.02f)
        : fminf(fmaxf(baseRh, 0.05f), 1.02f);
    qv[id] = qs * targetRh;
}

__global__ void kPatchCells(
    int W, int H,
    const int32_t* __restrict__ xs,
    const int32_t* __restrict__ ys,
    const uint8_t* __restrict__ types,
    int32_t count,
    uint8_t* __restrict__ cellType,
    float* __restrict__ snowAccum)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) return;
    int x = xs[k];
    int y = ys[k];
    if (x >= 0 && x < W && y >= 0 && y < H) {
        int id = cellIdx(x, y, W);
        cellType[id] = types[k];
        if (types[k] != WEATHER_CELL_AIR) {
            snowAccum[id] = 0.0f;
        }
    }
}

__global__ void kPatchStaticHeat(
    int W, int H,
    const int32_t* __restrict__ xs,
    const int32_t* __restrict__ ys,
    const float* __restrict__ rates,
    int32_t count,
    float* __restrict__ heatSource)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) return;
    int x = xs[k];
    int y = ys[k];
    if (x >= 0 && x < W && y >= 0 && y < H) {
        heatSource[cellIdx(x, y, W)] = rates[k];
    }
}

// One 16x16 block per dynamic heat source (campfire, lit torch, Heater crystal)
__global__ void kStampDynamicHeaters(
    int W, int H,
    const WeatherDynamicHeatSource* __restrict__ sources,
    int32_t count,
    float* __restrict__ dynHeat)
{
    int sIdx = blockIdx.x;
    if (sIdx >= count) return;
    WeatherDynamicHeatSource src = sources[sIdx];
    float r = fminf(fmaxf(src.radiusCells, 1.0f), 40.0f);
    if (fabsf(src.rateCPerSec) < 1e-4f) return;

    int xMin = max(0, (int)floorf(src.cx - r));
    int xMax = min(W - 1, (int)ceilf(src.cx + r));
    int yMin = max(0, (int)floorf(src.cy - r));
    int yMax = min(H - 1, (int)ceilf(src.cy + r));
    float invR2 = 1.0f / (r * r);

    for (int y = yMin + threadIdx.y; y <= yMax; y += blockDim.y) {
        float dy = ((float)y + 0.5f) - src.cy;
        int row = y * W;
        for (int x = xMin + threadIdx.x; x <= xMax; x += blockDim.x) {
            float dx = ((float)x + 0.5f) - src.cx;
            float d2 = (dx * dx + dy * dy) * invR2;
            if (d2 < 1.0f) {
                float w = (1.0f - d2) * (1.0f - d2);
                atomicAdd(&dynHeat[row + x], src.rateCPerSec * w);
            }
        }
    }
}

// Enforce zero normal velocity ONLY on outer domain borders
__global__ void kEnforceBoundaryU(int W, int H, float* __restrict__ u) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > W || j >= H) return;
    if (i == 0 || i == W) {
        u[uIdx(i, j, W)] = 0.0f;
    }
}

__global__ void kEnforceBoundaryV(int W, int H, float* __restrict__ v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= W || j > H) return;
    if (j == 0 || j == H) {
        v[vIdx(i, j, W)] = 0.0f;
    }
}

__global__ void kComputeVorticity(
    int W, int H, float invDx,
    const float* __restrict__ u,
    const float* __restrict__ v,
    float* __restrict__ vort)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= W || j >= H) return;
    int id = cellIdx(i, j, W);
    if (i == 0 || i == W - 1 || j == 0 || j == H - 1) {
        vort[id] = 0.0f;
        return;
    }
    float vR = 0.5f * (v[vIdx(i + 1, j, W)] + v[vIdx(i + 1, j + 1, W)]);
    float vL = 0.5f * (v[vIdx(i - 1, j, W)] + v[vIdx(i - 1, j + 1, W)]);
    float uU = 0.5f * (u[uIdx(i, j + 1, W)] + u[uIdx(i + 1, j + 1, W)]);
    float uD = 0.5f * (u[uIdx(i, j - 1, W)] + u[uIdx(i + 1, j - 1, W)]);
    vort[id] = 0.5f * invDx * ((vR - vL) - (uU - uD));
}

__global__ void kApplyForcesU(
    WeatherSimConfig cfg, float dt,
    const float* __restrict__ windForceX,
    const float* __restrict__ vort,
    const float* __restrict__ uIn,
    float* __restrict__ uOut)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int W = cfg.width, H = cfg.height;
    if (i > W || j >= H) return;
    int id = uIdx(i, j, W);
    if (i == 0 || i == W) {
        uOut[id] = 0.0f;
        return;
    }

    float val = uIn[id];
    float fx = 0.5f * (windForceX[cellIdx(i - 1, j, W)] + windForceX[cellIdx(i, j, W)]);

    if (cfg.vorticityEps > 0.0f && j > 0 && j < H - 1 && i > 1 && i < W - 1) {
        float wC = 0.5f * (vort[cellIdx(i - 1, j, W)] + vort[cellIdx(i, j, W)]);
        float dwdx = (fabsf(vort[cellIdx(i, j, W)]) - fabsf(vort[cellIdx(i - 1, j, W)]));
        float dwdy = 0.25f * (
            (fabsf(vort[cellIdx(i - 1, j + 1, W)]) + fabsf(vort[cellIdx(i, j + 1, W)])) -
            (fabsf(vort[cellIdx(i - 1, j - 1, W)]) + fabsf(vort[cellIdx(i, j - 1, W)])));
        float mag = sqrtf(dwdx * dwdx + dwdy * dwdy) + 1e-5f;
        float ny = dwdy / mag;
        fx += cfg.vorticityEps * ny * wC;
    }

    float dx = cfg.cellSizeTiles;
    float alpha = fminf(cfg.viscosity * dt / (dx * dx), 0.15f);
    if (alpha > 0.0f) {
        float uL = uIn[uIdx(i - 1, j, W)];
        float uR = uIn[uIdx(i + 1, j, W)];
        float uD = (j > 0) ? uIn[uIdx(i, j - 1, W)] : 0.0f;
        float uU = (j < H - 1) ? uIn[uIdx(i, j + 1, W)] : 0.0f;
        val += alpha * (uL + uR + uD + uU - 4.0f * val);
    }

    val = (val + dt * fx) * expf(-cfg.velocityDamping * dt);
    uOut[id] = fminf(fmaxf(val, -cfg.maxSpeedTilesPerSec), cfg.maxSpeedTilesPerSec);
}

__global__ void kApplyForcesV(
    WeatherSimConfig cfg, float dt,
    const float* __restrict__ windForceY,
    const float* __restrict__ T,
    const float* __restrict__ qv,
    const float* __restrict__ qc,
    const float* __restrict__ qr,
    const float* __restrict__ qi,
    const float* __restrict__ vort,
    const float* __restrict__ vIn,
    float* __restrict__ vOut)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int W = cfg.width, H = cfg.height;
    if (i >= W || j > H) return;
    int id = vIdx(i, j, W);
    if (j == 0 || j == H) {
        vOut[id] = 0.0f;
        return;
    }

    int cD = cellIdx(i, j - 1, W);
    int cU = cellIdx(i, j, W);

    float tFace = 0.5f * (T[cD] + T[cU]);
    float tEnv = envTempAtY((float)j - 0.5f, cfg);
    float tRefK = fmaxf(cfg.midLayerTempC + 273.15f, 200.0f);

    float qvFace = 0.5f * (qv[cD] + qv[cU]);
    float qswEnv, dqEnv, qsiEnv, dqiEnv;
    satWater(tEnv, qswEnv, dqEnv);
    satIce(tEnv, qsiEnv, dqiEnv);
    float qvEnv = ((tEnv <= 0.0f) ? qsiEnv : qswEnv) * cfg.initialRelHumidity;
    float condLoad = 0.5f * ((qc[cD] + qr[cD] + qi[cD]) + (qc[cU] + qr[cU] + qi[cU]));

    // Boussinesq buoyancy relative to resting environmental profile (T_env, qv_env)
    float buoy = cfg.buoyancyScale * ((tFace - tEnv) / tRefK + 0.61f * (qvFace - qvEnv) - 1.5f * condLoad);
    float fy = buoy + 0.5f * (windForceY[cD] + windForceY[cU]);

    if (cfg.vorticityEps > 0.0f && i > 0 && i < W - 1 && j > 1 && j < H - 1) {
        float wC = 0.5f * (vort[cD] + vort[cU]);
        float dwdx = 0.25f * (
            (fabsf(vort[cellIdx(i + 1, j - 1, W)]) + fabsf(vort[cellIdx(i + 1, j, W)])) -
            (fabsf(vort[cellIdx(i - 1, j - 1, W)]) + fabsf(vort[cellIdx(i - 1, j, W)])));
        float dwdy = (fabsf(vort[cU]) - fabsf(vort[cD]));
        float mag = sqrtf(dwdx * dwdx + dwdy * dwdy) + 1e-5f;
        float nx = dwdx / mag;
        fy -= cfg.vorticityEps * nx * wC;
    }

    float val = vIn[id];
    float dx = cfg.cellSizeTiles;
    float alpha = fminf(cfg.viscosity * dt / (dx * dx), 0.15f);
    if (alpha > 0.0f) {
        float vD = vIn[vIdx(i, j - 1, W)];
        float vU = vIn[vIdx(i, j + 1, W)];
        float vL = (i > 0) ? vIn[vIdx(i - 1, j, W)] : 0.0f;
        float vR = (i < W - 1) ? vIn[vIdx(i + 1, j, W)] : 0.0f;
        val += alpha * (vL + vR + vD + vU - 4.0f * val);
    }

    val = (val + dt * fy) * expf(-cfg.velocityDamping * dt);
    vOut[id] = fminf(fmaxf(val, -cfg.maxSpeedTilesPerSec), cfg.maxSpeedTilesPerSec);
}

__global__ void kAdvectU(
    int W, int H, float dtOverDx, float maxSpeed,
    const float* __restrict__ uVel,
    const float* __restrict__ vVel,
    const float* __restrict__ uSrc,
    float* __restrict__ uDst)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > W || j >= H) return;
    int id = uIdx(i, j, W);
    if (i == 0 || i == W) {
        uDst[id] = 0.0f;
        return;
    }
    float x = (float)i;
    float y = (float)j + 0.5f;
    float sx, sy;
    tracePoint(uVel, vVel, x, y, dtOverDx, 0.0f, W, H, sx, sy);
    float val = sampleU(uSrc, sx, sy, W, H);
    uDst[id] = fminf(fmaxf(val, -maxSpeed), maxSpeed);
}

__global__ void kAdvectV(
    int W, int H, float dtOverDx, float maxSpeed,
    const float* __restrict__ uVel,
    const float* __restrict__ vVel,
    const float* __restrict__ vSrc,
    float* __restrict__ vDst)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= W || j > H) return;
    int id = vIdx(i, j, W);
    if (j == 0 || j == H) {
        vDst[id] = 0.0f;
        return;
    }
    float x = (float)i + 0.5f;
    float y = (float)j;
    float sx, sy;
    tracePoint(uVel, vVel, x, y, dtOverDx, 0.0f, W, H, sx, sy);
    float val = sampleV(vSrc, sx, sy, W, H);
    vDst[id] = fminf(fmaxf(val, -maxSpeed), maxSpeed);
}

__global__ void kComputeDivergence(
    int W, int H, float warmStart,
    const float* __restrict__ u,
    const float* __restrict__ v,
    float* __restrict__ div,
    float* __restrict__ p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= W || j >= H) return;
    int id = cellIdx(i, j, W);
    float d = (u[uIdx(i + 1, j, W)] - u[uIdx(i, j, W)]) +
              (v[vIdx(i, j + 1, W)] - v[vIdx(i, j, W)]);
    div[id] = d;
    p[id] *= warmStart;
}

// Half-width packed Red-Black Gauss-Seidel SOR (100% warp active, 2x faster than masked launch)
__global__ void kPressureRedBlackSOR(
    int W, int H, int parity, float omega,
    const float* __restrict__ div,
    float* __restrict__ p)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= H) return;
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2 + ((j + parity) & 1);
    if (i >= W) return;
    int id = cellIdx(i, j, W);

    float sumP = 0.0f;
    int openCount = 0;

    if (i > 0)     { sumP += p[id - 1]; openCount++; }
    if (i < W - 1) { sumP += p[id + 1]; openCount++; }
    if (j > 0)     { sumP += p[id - W]; openCount++; }
    if (j < H - 1) { sumP += p[id + W]; openCount++; }

    float target = (sumP - div[id]) / (float)openCount;
    p[id] = (1.0f - omega) * p[id] + omega * target;
}

// Residual of discrete Neumann Poisson eq: (sumP - openCount * p) = rhs  =>  res = rhs - (sumP - openCount * p)
__global__ void kMgComputeResidual(
    int W, int H,
    const float* __restrict__ rhs,
    const float* __restrict__ p,
    float* __restrict__ res)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= W || j >= H) return;
    int id = cellIdx(i, j, W);

    float sumP = 0.0f;
    int openCount = 0;
    if (i > 0)     { sumP += p[id - 1]; openCount++; }
    if (i < W - 1) { sumP += p[id + 1]; openCount++; }
    if (j > 0)     { sumP += p[id - W]; openCount++; }
    if (j < H - 1) { sumP += p[id + W]; openCount++; }

    res[id] = rhs[id] - (sumP - (float)openCount * p[id]);
}

// Cell-centered finite-volume 2x2 sum restriction (exact 4x average scaling for unit-stencil discrete Laplacian)
__global__ void kMgRestrictSum(
    int fW, int fH, int cW, int cH,
    const float* __restrict__ resFine,
    float* __restrict__ rhsCoarse)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    int J = blockIdx.y * blockDim.y + threadIdx.y;
    if (I >= cW || J >= cH) return;

    int fx0 = I * 2;
    int fy0 = J * 2;
    float sum = resFine[fy0 * fW + fx0];
    if (fx0 + 1 < fW) {
        sum += resFine[fy0 * fW + (fx0 + 1)];
    }
    if (fy0 + 1 < fH) {
        sum += resFine[(fy0 + 1) * fW + fx0];
        if (fx0 + 1 < fW) {
            sum += resFine[(fy0 + 1) * fW + (fx0 + 1)];
        }
    }
    rhsCoarse[J * cW + I] = sum;
}

// Bilinear cell-centered prolongation with Neumann boundary clamping: pFine += P(eCoarse)
__global__ void kMgProlongateBilinearAdd(
    int fW, int fH, int cW, int cH,
    const float* __restrict__ eCoarse,
    float* __restrict__ pFine)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= fW || j >= fH) return;

    float cx = fminf(fmaxf(0.5f * (float)i - 0.25f, 0.0f), (float)(cW - 1));
    float cy = fminf(fmaxf(0.5f * (float)j - 0.25f, 0.0f), (float)(cH - 1));
    int i0 = (int)floorf(cx);
    int j0 = (int)floorf(cy);
    int i1 = min(i0 + 1, cW - 1);
    int j1 = min(j0 + 1, cH - 1);
    float fx = cx - (float)i0;
    float fy = cy - (float)j0;

    float v00 = eCoarse[j0 * cW + i0];
    float v10 = eCoarse[j0 * cW + i1];
    float v01 = eCoarse[j1 * cW + i0];
    float v11 = eCoarse[j1 * cW + i1];

    float a = v00 * (1.0f - fx) + v10 * fx;
    float b = v01 * (1.0f - fx) + v11 * fx;
    pFine[j * fW + i] += a * (1.0f - fy) + b * fy;
}

// Single-block shared-memory Red-Black SOR solver for coarsest Multigrid level (W * H <= 1024)
// Eliminates 70+ tiny kernel launches per V-cycle on Windows WDDM!
__global__ void kMgCoarsestSolveShared(
    int W, int H, int numIters, float omega,
    const float* __restrict__ rhs,
    float* __restrict__ p)
{
    __shared__ float sP[1024];
    __shared__ float sRhs[1024];

    int N = W * H;
    int tid = threadIdx.x;

    for (int idx = tid; idx < N; idx += blockDim.x) {
        sP[idx] = 0.0f;
        sRhs[idx] = rhs[idx];
    }
    __syncthreads();

    for (int it = 0; it < numIters; ++it) {
        for (int parity = 0; parity < 2; ++parity) {
            for (int idx = tid; idx < N; idx += blockDim.x) {
                int j = idx / W;
                int i = idx - j * W;
                if (((i + j) & 1) == parity) {
                    float sumP = 0.0f;
                    int cnt = 0;
                    if (i > 0)     { sumP += sP[idx - 1]; cnt++; }
                    if (i < W - 1) { sumP += sP[idx + 1]; cnt++; }
                    if (j > 0)     { sumP += sP[idx - W]; cnt++; }
                    if (j < H - 1) { sumP += sP[idx + W]; cnt++; }
                    float target = (sumP - sRhs[idx]) / (float)cnt;
                    sP[idx] = (1.0f - omega) * sP[idx] + omega * target;
                }
            }
            __syncthreads();
        }
    }

    for (int idx = tid; idx < N; idx += blockDim.x) {
        p[idx] = sP[idx];
    }
}

__global__ void kProjectU(int W, int H, const float* __restrict__ p, float* __restrict__ u) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > W || j >= H) return;
    int id = uIdx(i, j, W);
    if (i == 0 || i == W) {
        u[id] = 0.0f;
        return;
    }
    int cL = cellIdx(i - 1, j, W);
    int cR = cellIdx(i, j, W);
    u[id] -= (p[cR] - p[cL]);
}

__global__ void kProjectV(int W, int H, const float* __restrict__ p, float* __restrict__ v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= W || j > H) return;
    int id = vIdx(i, j, W);
    if (j == 0 || j == H) {
        v[id] = 0.0f;
        return;
    }
    int cD = cellIdx(i, j - 1, W);
    int cU = cellIdx(i, j, W);
    v[id] -= (p[cU] - p[cD]);
}

// Convert T <-> potential temperature theta = T + Gamma_d * (y - y_mid) * dx
__global__ void kTempToTheta(
    WeatherSimConfig cfg,
    float sign,
    const float* __restrict__ src,
    float* __restrict__ dst)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= cfg.width || j >= cfg.height) return;
    int id = cellIdx(i, j, cfg.width);
    float yMid = 0.5f * (float)(cfg.height - 1);
    float shift = cfg.adiabaticLapseCPerTile * ((float)j - yMid) * cfg.cellSizeTiles;
    dst[id] = src[id] + sign * shift;
}

__global__ void kAdvectScalarSL(
    int W, int H, float dtOverDx, float extraVyCells,
    int nonNegative,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ src,
    float* __restrict__ dst)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= W || j >= H) return;
    int id = cellIdx(i, j, W);
    float x = (float)i + 0.5f;
    float y = (float)j + 0.5f;
    float sx, sy;
    tracePoint(u, v, x, y, dtOverDx, extraVyCells, W, H, sx, sy);
    float mn, mx;
    float val = sampleScalar(src, sx, sy, W, H, mn, mx);
    if (nonNegative) val = fmaxf(val, 0.0f);
    dst[id] = val;
}

__global__ void kAdvectScalarMacCormackCorrect(
    int W, int H, float dtOverDx, float extraVyCells,
    int nonNegative,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ srcOrig,
    const float* __restrict__ phiHat,
    float* __restrict__ dst)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= W || j >= H) return;
    int id = cellIdx(i, j, W);
    float x = (float)i + 0.5f;
    float y = (float)j + 0.5f;

    float sx, sy;
    tracePoint(u, v, x, y, dtOverDx, extraVyCells, W, H, sx, sy);
    float mn, mx;
    sampleScalar(srcOrig, sx, sy, W, H, mn, mx);

    float bx, by;
    tracePoint(u, v, x, y, -dtOverDx, extraVyCells, W, H, bx, by);
    float dummyMn, dummyMx;
    float phiBwd = sampleScalar(phiHat, bx, by, W, H, dummyMn, dummyMx);

    float corr = phiHat[id] + 0.5f * (srcOrig[id] - phiBwd);
    corr = fminf(fmaxf(corr, mn), mx);
    if (nonNegative) corr = fmaxf(corr, 0.0f);
    dst[id] = corr;
}

__global__ void kDiffuseScalar(
    int W, int H, float alpha,
    const float* __restrict__ src,
    float* __restrict__ dst)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= W || j >= H) return;
    int id = cellIdx(i, j, W);
    float c = src[id];
    float sum = 0.0f;
    int cnt = 0;
    if (i > 0)     { sum += src[id - 1]; cnt++; }
    if (i < W - 1) { sum += src[id + 1]; cnt++; }
    if (j > 0)     { sum += src[id - W]; cnt++; }
    if (j < H - 1) { sum += src[id + W]; cnt++; }
    dst[id] = c + alpha * (sum - (float)cnt * c);
}

// Thermodynamics + In-Water Humidification + Condensation/Deposition Latent Heat + Freezing/Melting + Snow
__global__ void kMicrophysicsStep(
    WeatherSimConfig cfg, float dt,
    const uint8_t* __restrict__ cellType,
    const float* __restrict__ heatSource,
    const float* __restrict__ moistureSource,
    const float* __restrict__ dynHeat,
    const float* __restrict__ u,
    const float* __restrict__ v,
    float* __restrict__ T,
    float* __restrict__ qv,
    float* __restrict__ qc,
    float* __restrict__ qr,
    float* __restrict__ qi,
    float* __restrict__ snowAccum,
    float* __restrict__ evapAccum,
    float* __restrict__ rainAccum,
    float* __restrict__ latentRate)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int W = cfg.width, H = cfg.height;
    if (i >= W || j >= H) return;
    int id = cellIdx(i, j, W);

    uint8_t ctype = cellType[id];
    float temp = T[id];
    float v_qv = fmaxf(qv[id], 0.0f);
    float v_qc = fmaxf(qc[id], 0.0f);
    float v_qr = fmaxf(qr[id], 0.0f);
    float v_qi = fmaxf(qi[id], 0.0f);
    float sAcc = fmaxf(snowAccum[id], 0.0f);

    float uc = 0.5f * (u[uIdx(i, j, W)] + u[uIdx(i + 1, j, W)]);
    float vc = 0.5f * (v[vIdx(i, j, W)] + v[vIdx(i, j + 1, W)]);
    float speed = sqrtf(uc * uc + vc * vc);

    // 1. Static terrain heat (hot blocks/magma) + dynamic heat (campfires/torches/Heaters)
    temp += (heatSource[id] + dynHeat[id]) * dt;

    {
        float wBot = lowerZoneWeight((float)j, H);
        float wTop = upperZoneWeight((float)j, H);
        if (wBot > 0.0f && cfg.bottomTempBoostC != 0.0f) {
            float tEnv = envTempAtY((float)j, cfg);
            temp += (tEnv - temp) * (1.0f - expf(-0.08f * wBot * dt));
        }
        if (wTop > 0.0f && cfg.topTempDropC != 0.0f) {
            float tEnv = envTempAtY((float)j, cfg);
            temp += (tEnv - temp) * (1.0f - expf(-0.08f * wTop * dt));
        }
        if (wBot > 0.0f && cfg.bottomMoistureSource > 0.0f) {
            float qswMod, dqMod;
            satWater(temp, qswMod, dqMod);
            float targetCap = 1.02f * qswMod;
            if (v_qv < targetCap) {
                v_qv = fminf(v_qv + cfg.bottomMoistureSource * wBot * dt, targetCap);
            }
        }
        if (cfg.ambientHumidityTarget != 0.0f && cfg.ambientHumidityRate > 0.0f) {
            float effRate = (cfg.ambientHumidityTarget > 0.0f)
                ? cfg.ambientHumidityRate
                : (cfg.ambientHumidityRate * wBot);
            if (effRate > 0.0f) {
                float targetRh = fabsf(cfg.ambientHumidityTarget);
                float qswMod, dqMod, qsiMod, dqiMod;
                satWater(temp, qswMod, dqMod);
                satIce(temp, qsiMod, dqiMod);
                float qsMod = (temp <= 0.0f) ? qsiMod : qswMod;
                float qvFloor = qsMod * targetRh;
                if (v_qv < qvFloor) {
                    v_qv += (qvFloor - v_qv) * (1.0f - expf(-effRate * dt));
                }
            }
        }
    }

    // Gentle thermal relaxation inside solid rock / domain edges toward stratum background profile T_env(y)
    if (cfg.wallTempRelaxRate > 0.0f &&
        (ctype == WEATHER_CELL_SOLID || i == 0 || i == W - 1 || j == 0 || j == H - 1)) {
        float tEnv = envTempAtY((float)j, cfg);
        float wFactor = 1.0f - expf(-cfg.wallTempRelaxRate * dt);
        temp += (tEnv - temp) * wFactor;
    }

    float LvCp = kLvOverCp * cfg.latentHeatScale;
    float LsCp = kLsOverCp * cfg.latentHeatScale;
    float LfCp = kLfOverCp * cfg.latentHeatScale;
    float LvCpEvap = kLvOverCp * cfg.evapCoolingScale;

    float dTLatent = 0.0f;

    // 1b. Static per-cell moisture sources in open air:
    //     ms > 0: warm/neutral convective vapor injection (kg/kg/s, no evaporative cooling)
    //     ms < 0: non-depleting waterfall/spray evaporation (-ms in 1/s, Dalton law + evaporative cooling)
    if (ctype == WEATHER_CELL_AIR) {
        float ms = moistureSource[id];
        if (ms > 0.0f) {
            float qswSrc, dqSrc;
            satWater(temp, qswSrc, dqSrc);
            float cap = 1.025f * qswSrc;
            if (v_qv < cap) {
                v_qv = fminf(v_qv + ms * dt, cap);
            }
        } else if (ms < 0.0f) {
            float qswSrc, dqSrc;
            satWater(temp, qswSrc, dqSrc);
            float deficit = 1.015f * qswSrc - v_qv;
            if (deficit > 0.0f) {
                float rate = (-ms) * (1.0f + cfg.evapWindFactor * speed);
                float frac = 1.0f - expf(-rate * dt);
                float maxEq = deficit / (1.0f + LvCpEvap * dqSrc);
                float evap = fminf(deficit * frac, maxEq);
                v_qv += evap;
                float cool = LvCpEvap * evap;
                temp -= cool;
                dTLatent -= cool;
            }
        }
    }

    // 2. Real Dalton's Law water evaporation inside game water cells (proportional to wind & saturation deficit)
    if (ctype == WEATHER_CELL_WATER && cfg.evapRateCoeff > 0.0f) {
        float qsw, dqsw_dT;
        satWater(temp, qsw, dqsw_dT);
        float deficit = qsw - v_qv;
        if (deficit > 0.0f) {
            float rate = cfg.evapRateCoeff * (1.0f + cfg.evapWindFactor * speed);
            float frac = 1.0f - expf(-rate * dt);
            float maxEq = deficit / (1.0f + LvCpEvap * dqsw_dT);
            float evap = fminf(deficit * frac, maxEq);
            v_qv += evap;
            float cool = LvCpEvap * evap;
            temp -= cool;
            dTLatent -= cool;
            evapAccum[id] += evap;
        }
    }

    // 3. Saturation adjustment: Condensation (>0C) & Deposition (<=0C) + Latent Heat
    if (temp > 0.0f) {
        float qsw, dqsw_dT;
        satWater(temp, qsw, dqsw_dT);
        float excess = v_qv - qsw;
        if (excess > 0.0f) {
            float dq = excess / (1.0f + LvCp * dqsw_dT);
            dq = fminf(dq, v_qv);
            v_qv -= dq;
            v_qc += dq;
            float heat = LvCp * dq;
            temp += heat;
            dTLatent += heat;
        } else if (excess < 0.0f) {
            if (v_qc > 0.0f) {
                float dqMax = (-excess) / (1.0f + LvCp * dqsw_dT);
                float dq = fminf(v_qc, dqMax * (1.0f - expf(-4.0f * dt)));
                v_qc -= dq;
                v_qv += dq;
                float cool = LvCp * dq;
                temp -= cool;
                dTLatent -= cool;
            }
            if (v_qr > 0.0f) {
                satWater(temp, qsw, dqsw_dT);
                float def = fmaxf(qsw - v_qv, 0.0f);
                float dqMax = def / (1.0f + LvCp * dqsw_dT);
                // Raindrops are much larger than cloud droplets -> slow mid-air evaporation so rain reaches the ground!
                float dq = fminf(v_qr, dqMax * (1.0f - expf(-0.045f * dt)));
                v_qr -= dq;
                v_qv += dq;
                float cool = LvCp * dq;
                temp -= cool;
                dTLatent -= cool;
            }
        }
    } else {
        float qsi, dqsi_dT;
        satIce(temp, qsi, dqsi_dT);
        float excessIce = v_qv - qsi;
        if (excessIce > 0.0f) {
            float dq = excessIce / (1.0f + LsCp * dqsi_dT);
            dq = fminf(dq, v_qv);
            if (LsCp > 1e-4f && temp + LsCp * dq > 0.0f) {
                float dqToZero = fmaxf((0.0f - temp) / LsCp, 0.0f);
                dq = fminf(dq, dqToZero);
            }
            v_qv -= dq;
            v_qi += dq;
            float heat = LsCp * dq;
            temp += heat;
            dTLatent += heat;
        } else if (excessIce < 0.0f && v_qi > 0.0f) {
            float dqMax = (-excessIce) / (1.0f + LsCp * dqsi_dT);
            float dq = fminf(v_qi, dqMax * (1.0f - expf(-0.35f * dt)));
            v_qi -= dq;
            v_qv += dq;
            float cool = LsCp * dq;
            temp -= cool;
            dTLatent -= cool;
        }

        if (v_qc > 0.0f && temp < -0.1f) {
            float coldBoost = 1.0f + 0.25f * fminf(-temp, 20.0f);
            float frzFrac = 1.0f - expf(-cfg.freezeRate * coldBoost * dt);
            float dq = v_qc * frzFrac;
            if (LfCp > 1e-4f && temp + LfCp * dq > 0.0f) {
                dq = fminf(dq, fmaxf((0.0f - temp) / LfCp, 0.0f));
            }
            v_qc -= dq;
            v_qi += dq;
            float heat = LfCp * dq;
            temp += heat;
            dTLatent += heat;
        }
    }

    // 4. Autoconversion (qc -> qr) & Accretion
    if (v_qc > cfg.autoconvertThresh) {
        float dq = (v_qc - cfg.autoconvertThresh) * (1.0f - expf(-cfg.autoconvertRate * dt));
        v_qc -= dq;
        v_qr += dq;
    }
    if (v_qr > 1e-5f && v_qc > 1e-5f) {
        float accr = v_qc * (1.0f - expf(-25.0f * v_qr * dt));
        v_qc -= accr;
        v_qr += accr;
    }

    // 5. Melting when T > 0C
    if (temp > 0.05f) {
        if (v_qi > 0.0f) {
            float meltFrac = 1.0f - expf(-cfg.meltRate * temp * dt);
            float dm = v_qi * meltFrac;
            if (LfCp > 1e-4f && temp - LfCp * dm < 0.0f) {
                dm = fminf(dm, temp / LfCp);
            }
            v_qi -= dm;
            v_qr += dm;
            float cool = LfCp * dm;
            temp -= cool;
            dTLatent -= cool;
        }
        if (sAcc > 0.0f && temp > 0.1f) {
            float meltFrac = 1.0f - expf(-0.25f * cfg.meltRate * temp * dt);
            float dm = sAcc * meltFrac;
            if (LfCp > 1e-4f && temp - LfCp * dm < 0.0f) {
                dm = fminf(dm, temp / LfCp);
            }
            sAcc -= dm;
            v_qr += dm;
            float cool = LfCp * dm;
            temp -= cool;
            dTLatent -= cool;
        }
    }

    // 6. Snow & Rain surface contact on open air cells directly above solid ground or water
    if (ctype == WEATHER_CELL_AIR) {
        bool floorSolid = (j == 0) || (cellType[id - W] == WEATHER_CELL_SOLID);
        bool floorWater = (j > 0) && (cellType[id - W] == WEATHER_CELL_WATER);
        if (floorSolid) {
            if (v_qi > 1e-6f && (vc - cfg.snowFallSpeed) < 0.0f) {
                float downFlux = fminf(cfg.snowFallSpeed - vc, 12.0f);
                float depFrac = 1.0f - expf(-cfg.snowDepositRate * downFlux * dt);
                float dep = v_qi * depFrac;
                v_qi -= dep;
                sAcc += dep;
            }
            if (v_qr > 1e-6f && (vc - cfg.rainFallSpeed) < 0.0f) {
                // Capture downward rainfall flux hitting solid floor into ground water accumulator
                float fallRate = fmaxf(cfg.rainFallSpeed - vc, 1.0f) / fmaxf(cfg.cellSizeTiles, 0.25f);
                float depFrac = 1.0f - expf(-fallRate * dt);
                float dep = v_qr * depFrac;
                v_qr -= dep;
                rainAccum[id] += dep;
            }
        } else if (floorWater) {
            if (v_qi > 1e-6f) v_qi *= expf(-3.0f * dt);
            if (v_qr > 1e-6f && (vc - cfg.rainFallSpeed) < 0.0f) {
                float fallRate = fmaxf(cfg.rainFallSpeed - vc, 1.0f) / fmaxf(cfg.cellSizeTiles, 0.25f);
                float depFrac = 1.0f - expf(-fallRate * dt);
                float dep = v_qr * depFrac;
                v_qr -= dep;
                rainAccum[id] += dep;
            }
        }
    } else if (ctype == WEATHER_CELL_WATER) {
        // Falling precipitation absorbed inside water pools
        if (v_qi > 1e-6f) v_qi *= expf(-4.0f * dt);
        if (v_qr > 1e-6f) v_qr *= expf(-6.0f * dt);
        sAcc = 0.0f;
    } else {
        // Solid rock intercepts falling rain/snow so precipitation doesn't ghost through cave roofs
        if (v_qr > 1e-6f) v_qr *= expf(-14.0f * dt);
        if (v_qi > 1e-6f) v_qi *= expf(-8.0f * dt);
        sAcc = 0.0f;
    }

    T[id] = fminf(fmaxf(temp, -60.0f), 80.0f);
    qv[id] = fmaxf(v_qv, 0.0f);
    qc[id] = fmaxf(v_qc, 0.0f);
    qr[id] = fmaxf(v_qr, 0.0f);
    qi[id] = fmaxf(v_qi, 0.0f);
    snowAccum[id] = fmaxf(sAcc, 0.0f);
    latentRate[id] = (dt > 1e-6f) ? (dTLatent / dt) : 0.0f;
}

__global__ void kSamplePoints(
    WeatherSimConfig cfg,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ T,
    const float* __restrict__ qv,
    const float* __restrict__ qc,
    const float* __restrict__ qr,
    const float* __restrict__ qi,
    const float* __restrict__ cx,
    const float* __restrict__ cy,
    WeatherEntitySample* __restrict__ outSamples,
    int32_t count)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) return;
    int W = cfg.width, H = cfg.height;
    float x = fminf(fmaxf(cx[k], 0.5f), (float)W - 0.5f);
    float y = fminf(fmaxf(cy[k], 0.5f), (float)H - 0.5f);

    WeatherEntitySample s;
    s.u = sampleU(u, x, y, W, H);
    s.v = sampleV(v, x, y, W, H);
    float mn, mx;
    s.tempC = sampleScalar(T, x, y, W, H, mn, mx);
    float qvVal = sampleScalar(qv, x, y, W, H, mn, mx);
    float qsw, dq, qsi, dqi;
    satWater(s.tempC, qsw, dq);
    satIce(s.tempC, qsi, dqi);
    float qs = (s.tempC <= 0.0f) ? qsi : qsw;
    s.relHumidity = (qs > 1e-7f) ? (qvVal / qs) : 0.0f;
    s.cloud = sampleScalar(qc, x, y, W, H, mn, mx);
    s.rain  = sampleScalar(qr, x, y, W, H, mn, mx);
    s.snow  = sampleScalar(qi, x, y, W, H, mn, mx);
    outSamples[k] = s;
}

__global__ void kComputeCenterU(int W, int H, const float* __restrict__ u, float* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= W || j >= H) return;
    out[cellIdx(i, j, W)] = 0.5f * (u[uIdx(i, j, W)] + u[uIdx(i + 1, j, W)]);
}

__global__ void kComputeCenterV(int W, int H, const float* __restrict__ v, float* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= W || j >= H) return;
    out[cellIdx(i, j, W)] = 0.5f * (v[vIdx(i, j, W)] + v[vIdx(i, j + 1, W)]);
}

__global__ void kComputeRelHumidity(
    int W, int H,
    const float* __restrict__ T,
    const float* __restrict__ qv,
    float* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= W || j >= H) return;
    int id = cellIdx(i, j, W);
    float temp = T[id];
    float qsw, dq, qsi, dqi;
    satWater(temp, qsw, dq);
    satIce(temp, qsi, dqi);
    float qs = (temp <= 0.0f) ? qsi : qsw;
    out[id] = (qs > 1e-7f) ? (qv[id] / qs) : 0.0f;
}

__global__ void kPackViewWindowRGBA8(
    WeatherSimConfig cfg,
    int x0, int y0, int winW, int winH, int mode,
    float playerCx, float playerCy,
    float playerClearInnerCells, float playerClearOuterCells,
    float playerMinFogFactor, float wallFogMultiplier,
    const uint8_t* __restrict__ cellType,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ T,
    const float* __restrict__ qv,
    const float* __restrict__ qc,
    const float* __restrict__ qr,
    const float* __restrict__ qi,
    const float* __restrict__ snowAccum,
    const float* __restrict__ latentRate,
    uint8_t* __restrict__ outRgba)
{
    int lx = blockIdx.x * blockDim.x + threadIdx.x;
    int ly = blockIdx.y * blockDim.y + threadIdx.y;
    if (lx >= winW || ly >= winH) return;
    int gx = x0 + lx;
    int gy = y0 + ly;
    int outBase = (ly * winW + lx) * 4;

    if (gx < 0 || gx >= cfg.width || gy < 0 || gy >= cfg.height) {
        outRgba[outBase + 0] = 0;
        outRgba[outBase + 1] = 0;
        outRgba[outBase + 2] = 0;
        outRgba[outBase + 3] = 0;
        return;
    }

    int id = cellIdx(gx, gy, cfg.width);
    uint8_t ctype = cellType[id];
    bool isWall = (ctype == WEATHER_CELL_SOLID);

    float r = 0.0f, g = 0.0f, b = 0.0f, a = 0.0f;
    if (mode == 0) {
        // Natural in-game overlay: ONLY genuine condensed clouds/fog (qc), rain (qr), and snow (qi/snowAccum)
        // (No artificial RH<100% haze so air stays crisp unless true condensation occurs!)
        float cA = fminf(qc[id] * 300.0f, 0.58f);
        float rA = fminf(qr[id] * 420.0f, 0.48f);
        float iA = fminf((qi[id] * 360.0f + snowAccum[id] * 100.0f), 0.72f);
        float sum = cA + rA + iA;
        a = fminf(sum, 0.72f);
        if (sum > 1e-4f) {
            r = (cA * 0.88f + rA * 0.36f + iA * 0.94f) / sum;
            g = (cA * 0.91f + rA * 0.64f + iA * 0.98f) / sum;
            b = (cA * 0.96f + rA * 0.98f + iA * 1.00f) / sum;
        }

        // Halve fog/cloud effect inside solid walls compared to open air
        if (isWall) {
            a *= wallFogMultiplier;
        }

        // Smoothly attenuate fog near the player so local visibility stays clear
        if (playerClearOuterCells > 0.0f) {
            float dx = ((float)gx + 0.5f) - playerCx;
            float dy = ((float)gy + 0.5f) - playerCy;
            float dist = sqrtf(dx * dx + dy * dy);
            float t = smoothstep01((dist - playerClearInnerCells) / fmaxf(playerClearOuterCells - playerClearInnerCells, 1.0f));
            float pFactor = playerMinFogFactor + (1.0f - playerMinFogFactor) * t;
            a *= pFactor;
        }
    } else if (mode == 1) {
        float norm = fminf(fmaxf((T[id] - (cfg.midLayerTempC - 15.0f)) / 30.0f, 0.0f), 1.0f);
        r = fminf(fmaxf(1.5f * norm - 0.25f, 0.0f), 1.0f);
        g = 1.0f - fabsf(2.0f * norm - 1.0f);
        b = fminf(fmaxf(1.25f - 1.5f * norm, 0.0f), 1.0f);
        a = isWall ? 0.32f : 0.55f;
    } else if (mode == 2) {
        float qsw, dq, qsi, dqi;
        satWater(T[id], qsw, dq);
        satIce(T[id], qsi, dqi);
        float qs = (T[id] <= 0.0f) ? qsi : qsw;
        float rh = (qs > 1e-7f) ? fminf(qv[id] / qs, 1.2f) : 0.0f;
        r = 0.08f; g = 0.52f * fminf(rh, 1.0f); b = fminf(rh, 1.0f);
        a = (isWall ? 0.32f : 0.55f) * fminf(rh, 1.0f);
    } else if (mode == 3) {
        float uc = 0.5f * (u[uIdx(gx, gy, cfg.width)] + u[uIdx(gx + 1, gy, cfg.width)]);
        float vc = 0.5f * (v[vIdx(gx, gy, cfg.width)] + v[vIdx(gx, gy + 1, cfg.width)]);
        float spd = fminf(sqrtf(uc * uc + vc * vc) / 8.0f, 1.0f);
        r = spd; g = 0.85f * spd; b = 1.0f - 0.5f * spd;
        a = (isWall ? 0.34f : 0.60f) * spd;
    } else {
        float lr = fminf(fmaxf(latentRate[id] / 0.5f, -1.0f), 1.0f);
        float baseA = isWall ? 0.35f : 0.65f;
        if (lr >= 0.0f) {
            r = 1.0f; g = 0.45f * (1.0f - lr); b = 0.10f; a = baseA * lr;
        } else {
            float c = -lr;
            r = 0.15f; g = 0.65f; b = 1.0f; a = baseA * c;
        }
    }

    outRgba[outBase + 0] = (uint8_t)(fminf(fmaxf(r, 0.0f), 1.0f) * 255.0f);
    outRgba[outBase + 1] = (uint8_t)(fminf(fmaxf(g, 0.0f), 1.0f) * 255.0f);
    outRgba[outBase + 2] = (uint8_t)(fminf(fmaxf(b, 0.0f), 1.0f) * 255.0f);
    outRgba[outBase + 3] = (uint8_t)(fminf(fmaxf(a, 0.0f), 1.0f) * 255.0f);
}

__global__ void kSampleViewPrecipGrid(
    WeatherSimConfig cfg,
    int x0, int y0, int winW, int winH,
    int outW, int outH,
    const uint8_t* __restrict__ cellType,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ T,
    const float* __restrict__ qc,
    const float* __restrict__ qr,
    const float* __restrict__ qi,
    const float* __restrict__ snowAccum,
    WeatherPrecipCell* __restrict__ outCells)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= outW || oy >= outH) return;

    int gx = x0 + (int)(((float)ox + 0.5f) * (float)winW / (float)outW);
    int gy = y0 + (int)(((float)oy + 0.5f) * (float)winH / (float)outH);

    WeatherPrecipCell c;
    c.u = 0.0f; c.v = 0.0f; c.tempC = cfg.midLayerTempC;
    c.qc = 0.0f; c.qr = 0.0f; c.qi = 0.0f; c.snowAccum = 0.0f; c.isAir = 0;

    if (gx >= 0 && gx < cfg.width && gy >= 0 && gy < cfg.height) {
        int id = cellIdx(gx, gy, cfg.width);
        // Only spawn visible rain/snow particles in open cave air (not inside solid rock or submerged underwater)
        if (cellType[id] == WEATHER_CELL_AIR) {
            c.isAir = 1;
            c.u = 0.5f * (u[uIdx(gx, gy, cfg.width)] + u[uIdx(gx + 1, gy, cfg.width)]);
            c.v = 0.5f * (v[vIdx(gx, gy, cfg.width)] + v[vIdx(gx, gy + 1, cfg.width)]);
            c.tempC = T[id];
            c.qc = qc[id];
            c.qr = qr[id];
            c.qi = qi[id];
            c.snowAccum = snowAccum[id];
        }
    }
    outCells[oy * outW + ox] = c;
}

// Aggregate sub-tile evaporation & rainfall deposition into discrete game-tile fluid events (-1 / +1)
__global__ void kResolveFluidEvents(
    WeatherSimConfig cfg,
    int tilesW, int tilesH, int sub,
    uint8_t* __restrict__ cellType,
    float* __restrict__ evapAccum,
    float* __restrict__ rainAccum,
    WeatherFluidEvent* __restrict__ outEvents,
    int32_t* __restrict__ outEventCount,
    int32_t maxEvents)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= tilesW || ty >= tilesH) return;

    int W = cfg.width;
    int baseSx = tx * sub;
    int baseSy = ty * sub;

    uint8_t tileType = cellType[cellIdx(baseSx, baseSy, W)];

    // 1. Evaporation depletion in any water tile (surface or interior)
    if (tileType == WEATHER_CELL_WATER && cfg.evapMassPerWaterTile > 0.0f) {
        float sumEvap = 0.0f;
        for (int dy = 0; dy < sub; ++dy) {
            int row = (baseSy + dy) * W + baseSx;
            for (int dx = 0; dx < sub; ++dx) {
                sumEvap += evapAccum[row + dx];
            }
        }
        if (sumEvap >= cfg.evapMassPerWaterTile) {
            int slot = atomicAdd(outEventCount, 1);
            if (slot < maxEvents) {
                outEvents[slot].tileX = tx;
                outEvents[slot].tileY = ty;
                outEvents[slot].delta = -1;
                // Immediately mark tile AIR on GPU & clear accum to avoid duplicate removal events
                for (int dy = 0; dy < sub; ++dy) {
                    int row = (baseSy + dy) * W + baseSx;
                    for (int dx = 0; dx < sub; ++dx) {
                        evapAccum[row + dx] = 0.0f;
                        cellType[row + dx] = WEATHER_CELL_AIR;
                    }
                }
            }
        }
        return;
    }

    // 2. Rainfall ground water generation on open-air tiles sitting on solid floor or existing water
    if (tileType == WEATHER_CELL_AIR && cfg.rainMassPerWaterTile > 0.0f) {
        bool supported = (ty == 0);
        if (!supported) {
            uint8_t belowType = cellType[cellIdx(baseSx, (ty - 1) * sub, W)];
            supported = (belowType == WEATHER_CELL_SOLID || belowType == WEATHER_CELL_WATER);
        }
        if (!supported) return;

        float sumRain = 0.0f;
        for (int dy = 0; dy < sub; ++dy) {
            int row = (baseSy + dy) * W + baseSx;
            for (int dx = 0; dx < sub; ++dx) {
                sumRain += rainAccum[row + dx];
            }
        }
        if (sumRain >= cfg.rainMassPerWaterTile) {
            int slot = atomicAdd(outEventCount, 1);
            if (slot < maxEvents) {
                outEvents[slot].tileX = tx;
                outEvents[slot].tileY = ty;
                outEvents[slot].delta = +1;
                float keepRatio = fmaxf((sumRain - cfg.rainMassPerWaterTile) / fmaxf(sumRain, 1e-6f), 0.0f);
                for (int dy = 0; dy < sub; ++dy) {
                    int row = (baseSy + dy) * W + baseSx;
                    for (int dx = 0; dx < sub; ++dx) {
                        rainAccum[row + dx] *= keepRatio;
                    }
                }
            }
        }
    }
}

void advectScalar(
    WeatherSimHandle* h,
    float dtOverDx,
    float extraVyCells,
    int nonNegative,
    bool useMacCormack,
    float* field,
    float* tmp,
    float* hat,
    dim3 grid,
    dim3 block)
{
    int W = h->cfg.width, H = h->cfg.height;
    if (useMacCormack && hat != nullptr) {
        kAdvectScalarSL<<<grid, block, 0, h->stream>>>(
            W, H, dtOverDx, extraVyCells, nonNegative,
            h->d.u, h->d.v, field, hat);
        kAdvectScalarMacCormackCorrect<<<grid, block, 0, h->stream>>>(
            W, H, dtOverDx, extraVyCells, nonNegative,
            h->d.u, h->d.v, field, hat, tmp);
    } else {
        kAdvectScalarSL<<<grid, block, 0, h->stream>>>(
            W, H, dtOverDx, extraVyCells, nonNegative,
            h->d.u, h->d.v, field, tmp);
    }
}

} // namespace

cudaError_t launchResetFields(WeatherSimHandle* h) {
    int W = h->cfg.width, H = h->cfg.height;
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);

    cudaMemsetAsync(h->d.u, 0, (size_t)(W + 1) * H * sizeof(float), h->stream);
    cudaMemsetAsync(h->d.v, 0, (size_t)W * (H + 1) * sizeof(float), h->stream);
    cudaMemsetAsync(h->d.uTmp, 0, (size_t)(W + 1) * H * sizeof(float), h->stream);
    cudaMemsetAsync(h->d.vTmp, 0, (size_t)W * (H + 1) * sizeof(float), h->stream);

    kResetFields<<<grid, block, 0, h->stream>>>(
        h->cfg,
        h->d.T, h->d.qv, h->d.qc, h->d.qr, h->d.qi,
        h->d.p, h->d.div, h->d.snowAccum, h->d.evapAccum, h->d.rainAccum, h->d.latentRate);
    if (h->d.d_fluidEventCount) {
        cudaMemsetAsync(h->d.d_fluidEventCount, 0, sizeof(int32_t), h->stream);
    }
    return cudaGetLastError();
}

cudaError_t launchApplyInitialHumidityGradient(
    WeatherSimHandle* h,
    float bottomRh,
    float topRh,
    float yBottomFrac,
    float yTopFrac)
{
    int W = h->cfg.width, H = h->cfg.height;
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    kApplyInitialHumidityGradient<<<grid, block, 0, h->stream>>>(
        h->cfg,
        bottomRh,
        topRh,
        yBottomFrac,
        yTopFrac,
        h->d.T,
        h->d.qv);
    return cudaGetLastError();
}

cudaError_t launchApplyInitialPerlinNoise(
    WeatherSimHandle* h,
    int32_t seed,
    float tempAmpC,
    float rhAmp,
    float wavelengthTiles,
    int32_t octaves)
{
    if (tempAmpC == 0.0f && rhAmp == 0.0f) return cudaSuccess;
    int W = h->cfg.width, H = h->cfg.height;
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    float invWave = 1.0f / fmaxf(wavelengthTiles, 8.0f);
    int32_t oct = octaves < 1 ? 1 : (octaves > 6 ? 6 : octaves);

    kApplyInitialPerlinNoise<<<grid, block, 0, h->stream>>>(
        h->cfg,
        (uint32_t)seed,
        tempAmpC,
        rhAmp,
        invWave,
        oct,
        h->d.T,
        h->d.qv);
    return cudaGetLastError();
}

cudaError_t launchPatchCells(WeatherSimHandle* h, int32_t count) {
    if (count <= 0) return cudaSuccess;
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    kPatchCells<<<blocks, threads, 0, h->stream>>>(
        h->cfg.width, h->cfg.height,
        h->d.d_patchX, h->d.d_patchY, h->d.d_patchType, count,
        h->d.cellType, h->d.snowAccum);
    return cudaGetLastError();
}

cudaError_t launchStepSimulation(WeatherSimHandle* h, float dt) {
    const WeatherSimConfig& cfg = h->cfg;
    int W = cfg.width, H = cfg.height;
    float dx = cfg.cellSizeTiles;
    float invDx = 1.0f / dx;
    float dtOverDx = dt * invDx;

    dim3 block(32, 8);
    dim3 gridCell((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    dim3 gridU(((W + 1) + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    dim3 gridV((W + block.x - 1) / block.x, ((H + 1) + block.y - 1) / block.y);

    // 1. Vorticity + Buoyancy / Wind forces / Viscosity -> uTmp, vTmp
    if (cfg.vorticityEps > 0.0f) {
        kComputeVorticity<<<gridCell, block, 0, h->stream>>>(
            W, H, invDx, h->d.u, h->d.v, h->d.vort);
    }
    kApplyForcesU<<<gridU, block, 0, h->stream>>>(
        cfg, dt, h->d.windForceX, h->d.vort, h->d.u, h->d.uTmp);
    kApplyForcesV<<<gridV, block, 0, h->stream>>>(
        cfg, dt, h->d.windForceY,
        h->d.T, h->d.qv, h->d.qc, h->d.qr, h->d.qi,
        h->d.vort, h->d.v, h->d.vTmp);

    // 2. Velocity self-advection: uTmp, vTmp -> u, v
    kAdvectU<<<gridU, block, 0, h->stream>>>(
        W, H, dtOverDx, cfg.maxSpeedTilesPerSec,
        h->d.uTmp, h->d.vTmp, h->d.uTmp, h->d.u);
    kAdvectV<<<gridV, block, 0, h->stream>>>(
        W, H, dtOverDx, cfg.maxSpeedTilesPerSec,
        h->d.uTmp, h->d.vTmp, h->d.vTmp, h->d.v);

    // 3. Enforce outer domain border impermeability only
    kEnforceBoundaryU<<<gridU, block, 0, h->stream>>>(W, H, h->d.u);
    kEnforceBoundaryV<<<gridV, block, 0, h->stream>>>(W, H, h->d.v);

    // 4. Pressure projection (Geometric Multigrid V-Cycle for exact global mass conservation!)
    kComputeDivergence<<<gridCell, block, 0, h->stream>>>(
        W, H, cfg.pressureWarmStart, h->d.u, h->d.v, h->d.div, h->d.p);

    if (h->d.mgNumLevels >= 2) {
        // Each V(1,1) cycle solves all spatial frequencies globally (eliminates horizontal-line updraft bias!)
        int iters = cfg.sorIterations > 0 ? cfg.sorIterations : 18;
        int numVCycles = (iters + 4) / 9; // 18 => 2 V-cycles (<5ms on 2048x2048, <0.01% flux error!)
        if (numVCycles < 1) numVCycles = 1;
        if (numVCycles > 4) numVCycles = 4;
        int coarsest = h->d.mgNumLevels - 1;

        for (int cyc = 0; cyc < numVCycles; ++cyc) {
            // Downward leg: pre-smooth (1x Red-Black GS, omega=1.0), compute residual, restrict 2x2
            for (int lvl = 0; lvl < coarsest; ++lvl) {
                const WeatherMgLevel& fine = h->d.mgLevels[lvl];
                const WeatherMgLevel& coarse = h->d.mgLevels[lvl + 1];

                int halfW = (fine.w + 1) / 2;
                dim3 gSOR((halfW + block.x - 1) / block.x, (fine.h + block.y - 1) / block.y);
                dim3 gFine((fine.w + block.x - 1) / block.x, (fine.h + block.y - 1) / block.y);
                dim3 gCoarse((coarse.w + block.x - 1) / block.x, (coarse.h + block.y - 1) / block.y);

                // On coarse levels (lvl > 0) p starts at 0, so a single Red-Black GS sweep is sufficient
                kPressureRedBlackSOR<<<gSOR, block, 0, h->stream>>>(fine.w, fine.h, 0, 1.0f, fine.rhs, fine.p);
                kPressureRedBlackSOR<<<gSOR, block, 0, h->stream>>>(fine.w, fine.h, 1, 1.0f, fine.rhs, fine.p);

                kMgComputeResidual<<<gFine, block, 0, h->stream>>>(fine.w, fine.h, fine.rhs, fine.p, fine.res);
                kMgRestrictSum<<<gCoarse, block, 0, h->stream>>>(fine.w, fine.h, coarse.w, coarse.h, fine.res, coarse.rhs);
                if (lvl + 1 < coarsest || coarse.w * coarse.h > 1024) {
                    cudaMemsetAsync(coarse.p, 0, (size_t)coarse.w * (size_t)coarse.h * sizeof(float), h->stream);
                }
            }

            // Coarsest grid exact solve in 1 shared-memory kernel launch (<0.01 ms!)
            {
                const WeatherMgLevel& C = h->d.mgLevels[coarsest];
                if (C.w * C.h <= 1024) {
                    kMgCoarsestSolveShared<<<1, 256, 0, h->stream>>>(C.w, C.h, 40, 1.55f, C.rhs, C.p);
                } else {
                    int halfW = (C.w + 1) / 2;
                    dim3 gSOR((halfW + block.x - 1) / block.x, (C.h + block.y - 1) / block.y);
                    for (int s = 0; s < 36; ++s) {
                        kPressureRedBlackSOR<<<gSOR, block, 0, h->stream>>>(C.w, C.h, 0, 1.55f, C.rhs, C.p);
                        kPressureRedBlackSOR<<<gSOR, block, 0, h->stream>>>(C.w, C.h, 1, 1.55f, C.rhs, C.p);
                    }
                }
            }

            // Upward leg: bilinear prolongate & add, then post-smooth (1x Red-Black GS, omega=1.0)
            for (int lvl = coarsest - 1; lvl >= 0; --lvl) {
                const WeatherMgLevel& fine = h->d.mgLevels[lvl];
                const WeatherMgLevel& coarse = h->d.mgLevels[lvl + 1];

                int halfW = (fine.w + 1) / 2;
                dim3 gSOR((halfW + block.x - 1) / block.x, (fine.h + block.y - 1) / block.y);
                dim3 gFine((fine.w + block.x - 1) / block.x, (fine.h + block.y - 1) / block.y);

                kMgProlongateBilinearAdd<<<gFine, block, 0, h->stream>>>(
                    fine.w, fine.h, coarse.w, coarse.h, coarse.p, fine.p);

                kPressureRedBlackSOR<<<gSOR, block, 0, h->stream>>>(fine.w, fine.h, 0, 1.0f, fine.rhs, fine.p);
                kPressureRedBlackSOR<<<gSOR, block, 0, h->stream>>>(fine.w, fine.h, 1, 1.0f, fine.rhs, fine.p);
            }
        }
    } else {
        int halfW = (W + 1) / 2;
        dim3 gridSOR((halfW + block.x - 1) / block.x, (H + block.y - 1) / block.y);
        int iters = cfg.sorIterations > 0 ? cfg.sorIterations : 18;
        for (int it = 0; it < iters; ++it) {
            kPressureRedBlackSOR<<<gridSOR, block, 0, h->stream>>>(
                W, H, 0, cfg.sorOmega, h->d.div, h->d.p);
            kPressureRedBlackSOR<<<gridSOR, block, 0, h->stream>>>(
                W, H, 1, cfg.sorOmega, h->d.div, h->d.p);
        }
    }

    kProjectU<<<gridU, block, 0, h->stream>>>(W, H, h->d.p, h->d.u);
    kProjectV<<<gridV, block, 0, h->stream>>>(W, H, h->d.p, h->d.v);

    // 5. Scalar advection
    bool mc = (cfg.useMacCormack != 0);

    // 5a. Temperature via Potential Temperature (exact adiabatic cooling/warming)
    kTempToTheta<<<gridCell, block, 0, h->stream>>>(cfg, +1.0f, h->d.T, h->d.TTmp);
    advectScalar(h, dtOverDx, 0.0f, 0, mc, h->d.TTmp, h->d.T, h->d.THat, gridCell, block);
    kTempToTheta<<<gridCell, block, 0, h->stream>>>(cfg, -1.0f, h->d.T, h->d.TTmp);

    float alphaT = fminf(cfg.thermalDiffusivity * dt / (dx * dx), 0.15f);
    if (alphaT > 0.0f) {
        kDiffuseScalar<<<gridCell, block, 0, h->stream>>>(W, H, alphaT, h->d.TTmp, h->d.T);
    } else {
        float* swapPtr = h->d.T; h->d.T = h->d.TTmp; h->d.TTmp = swapPtr;
    }

    // 5b. Water vapor qv & Cloud droplets qc
    advectScalar(h, dtOverDx, 0.0f, 1, mc, h->d.qv, h->d.qvTmp, h->d.qvHat, gridCell, block);
    advectScalar(h, dtOverDx, 0.0f, 1, mc, h->d.qc, h->d.qcTmp, h->d.qcHat, gridCell, block);

    float alphaQ = fminf(cfg.moistureDiffusivity * dt / (dx * dx), 0.15f);
    if (alphaQ > 0.0f) {
        kDiffuseScalar<<<gridCell, block, 0, h->stream>>>(W, H, alphaQ, h->d.qvTmp, h->d.qv);
        kDiffuseScalar<<<gridCell, block, 0, h->stream>>>(W, H, alphaQ, h->d.qcTmp, h->d.qc);
    } else {
        float* s1 = h->d.qv; h->d.qv = h->d.qvTmp; h->d.qvTmp = s1;
        float* s2 = h->d.qc; h->d.qc = h->d.qcTmp; h->d.qcTmp = s2;
    }

    // 5c. Rain qr & Snow qi
    advectScalar(h, dtOverDx, -cfg.rainFallSpeed, 1, false, h->d.qr, h->d.qrTmp, nullptr, gridCell, block);
    advectScalar(h, dtOverDx, -cfg.snowFallSpeed, 1, false, h->d.qi, h->d.qiTmp, nullptr, gridCell, block);
    {
        float* s3 = h->d.qr; h->d.qr = h->d.qrTmp; h->d.qrTmp = s3;
        float* s4 = h->d.qi; h->d.qi = h->d.qiTmp; h->d.qiTmp = s4;
    }

    // 6. Microphysics
    kMicrophysicsStep<<<gridCell, block, 0, h->stream>>>(
        cfg, dt,
        h->d.cellType, h->d.heatSource, h->d.moistureSource, h->d.dynHeat,
        h->d.u, h->d.v,
        h->d.T, h->d.qv, h->d.qc, h->d.qr, h->d.qi,
        h->d.snowAccum, h->d.evapAccum, h->d.rainAccum, h->d.latentRate);

    // 7. Resolve discrete water-cycle events (evaporation depletion / rainfall water spawn)
    if (h->d.d_fluidEvents && h->d.d_fluidEventCount &&
        (cfg.evapMassPerWaterTile > 0.0f || cfg.rainMassPerWaterTile > 0.0f)) {
        launchResolveFluidEvents(h);
    }

    return cudaGetLastError();
}

cudaError_t launchSamplePoints(WeatherSimHandle* h, int32_t count) {
    if (count <= 0) return cudaSuccess;
    int threads = 128;
    int blocks = (count + threads - 1) / threads;
    kSamplePoints<<<blocks, threads, 0, h->stream>>>(
        h->cfg,
        h->d.u, h->d.v,
        h->d.T, h->d.qv, h->d.qc, h->d.qr, h->d.qi,
        h->d.d_sampleX, h->d.d_sampleY, h->d.d_sampleOut, count);
    return cudaGetLastError();
}

cudaError_t launchPrepareFieldForCopy(WeatherSimHandle* h, int32_t fieldId, const float** outDevicePtr) {
    int W = h->cfg.width, H = h->cfg.height;
    dim3 block(32, 8);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    switch (fieldId) {
        case 0:
            kComputeCenterU<<<grid, block, 0, h->stream>>>(W, H, h->d.u, h->d.scratchScalar);
            *outDevicePtr = h->d.scratchScalar;
            break;
        case 1:
            kComputeCenterV<<<grid, block, 0, h->stream>>>(W, H, h->d.v, h->d.scratchScalar);
            *outDevicePtr = h->d.scratchScalar;
            break;
        case 2: *outDevicePtr = h->d.T; break;
        case 3: *outDevicePtr = h->d.qv; break;
        case 4: *outDevicePtr = h->d.qc; break;
        case 5: *outDevicePtr = h->d.qr; break;
        case 6: *outDevicePtr = h->d.qi; break;
        case 7: *outDevicePtr = h->d.snowAccum; break;
        case 8:
            kComputeRelHumidity<<<grid, block, 0, h->stream>>>(W, H, h->d.T, h->d.qv, h->d.scratchScalar);
            *outDevicePtr = h->d.scratchScalar;
            break;
        case 9:
            kComputeDivergence<<<grid, block, 0, h->stream>>>(W, H, 1.0f, h->d.u, h->d.v, h->d.scratchScalar, h->d.p);
            *outDevicePtr = h->d.scratchScalar;
            break;
        case 10: *outDevicePtr = h->d.latentRate; break;
        case 11: *outDevicePtr = h->d.evapAccum; break;
        default:
            return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

cudaError_t launchPackViewWindowRGBA8(
    WeatherSimHandle* h,
    int32_t x0, int32_t y0,
    int32_t winW, int32_t winH,
    int32_t mode,
    float playerCx, float playerCy,
    float playerClearInnerCells, float playerClearOuterCells,
    float playerMinFogFactor, float wallFogMultiplier)
{
    dim3 block(16, 16);
    dim3 grid((winW + 15) / 16, (winH + 15) / 16);
    kPackViewWindowRGBA8<<<grid, block, 0, h->stream>>>(
        h->cfg, x0, y0, winW, winH, mode,
        playerCx, playerCy,
        playerClearInnerCells, playerClearOuterCells,
        playerMinFogFactor, wallFogMultiplier,
        h->d.cellType, h->d.u, h->d.v,
        h->d.T, h->d.qv, h->d.qc, h->d.qr, h->d.qi, h->d.snowAccum, h->d.latentRate,
        h->d.d_rgbaWindow);
    return cudaGetLastError();
}

cudaError_t launchSampleViewPrecipGrid(
    WeatherSimHandle* h,
    int32_t x0, int32_t y0,
    int32_t winW, int32_t winH,
    int32_t outW, int32_t outH)
{
    dim3 block(16, 16);
    dim3 grid((outW + 15) / 16, (outH + 15) / 16);
    kSampleViewPrecipGrid<<<grid, block, 0, h->stream>>>(
        h->cfg, x0, y0, winW, winH, outW, outH,
        h->d.cellType, h->d.u, h->d.v,
        h->d.T, h->d.qc, h->d.qr, h->d.qi, h->d.snowAccum,
        h->d.d_precipGrid);
    return cudaGetLastError();
}

// Full-map downsampled RGBA8 weather radar kernel (0=RH, 1=Cloud, 2=Rain, 3=Snow, 4=Temp, 5=Wind, 6=Composite)
__global__ void kRenderFullMapRGBA8(
    WeatherSimConfig cfg,
    int mapW, int mapH, int layerMode, int showWalls,
    const uint8_t* __restrict__ cellType,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ T,
    const float* __restrict__ qv,
    const float* __restrict__ qc,
    const float* __restrict__ qr,
    const float* __restrict__ qi,
    const float* __restrict__ snowAccum,
    uint8_t* __restrict__ outRgba)
{
    int mx = blockIdx.x * blockDim.x + threadIdx.x;
    int my = blockIdx.y * blockDim.y + threadIdx.y;
    if (mx >= mapW || my >= mapH) return;

    int W = cfg.width, H = cfg.height;
    float fx0 = ((float)mx + 0.25f) * (float)W / (float)mapW;
    float fx1 = ((float)mx + 0.75f) * (float)W / (float)mapW;
    float fy0 = ((float)my + 0.25f) * (float)H / (float)mapH;
    float fy1 = ((float)my + 0.75f) * (float)H / (float)mapH;

    int xs[2] = { min(max((int)fx0, 0), W - 1), min(max((int)fx1, 0), W - 1) };
    int ys[2] = { min(max((int)fy0, 0), H - 1), min(max((int)fy1, 0), H - 1) };

    float sT = 0.0f, sRH = 0.0f, sQc = 0.0f, sQr = 0.0f, sQi = 0.0f, sSacc = 0.0f, sSpd = 0.0f;
    int solidHits = 0, waterHits = 0;

    for (int ky = 0; ky < 2; ++ky) {
        int gy = ys[ky];
        for (int kx = 0; kx < 2; ++kx) {
            int gx = xs[kx];
            int id = cellIdx(gx, gy, W);
            uint8_t ct = cellType[id];
            if (ct == WEATHER_CELL_SOLID) solidHits++;
            else if (ct == WEATHER_CELL_WATER) waterHits++;

            float temp = T[id];
            float qsw, dq, qsi, dqi;
            satWater(temp, qsw, dq);
            satIce(temp, qsi, dqi);
            float qs = (temp <= 0.0f) ? qsi : qsw;
            float rh = (qs > 1e-7f) ? fminf(qv[id] / qs, 1.25f) : 0.0f;

            float uc = 0.5f * (u[uIdx(gx, gy, W)] + u[uIdx(gx + 1, gy, W)]);
            float vc = 0.5f * (v[vIdx(gx, gy, W)] + v[vIdx(gx, gy + 1, W)]);

            sT += temp;
            sRH += rh;
            sQc += qc[id];
            sQr += qr[id];
            sQi += qi[id];
            sSacc += snowAccum[id];
            sSpd += sqrtf(uc * uc + vc * vc);
        }
    }
    sT *= 0.25f; sRH *= 0.25f; sQc *= 0.25f; sQr *= 0.25f; sQi *= 0.25f; sSacc *= 0.25f; sSpd *= 0.25f;
    float solidFrac = solidHits * 0.25f;
    float waterFrac = waterHits * 0.25f;

    float r = 0.07f, g = 0.09f, b = 0.13f;

    if (layerMode == 0) {
        // Relative Humidity (0% dark ochre -> 50% teal -> 100% vivid cyan-blue)
        float h = fminf(fmaxf(sRH, 0.0f), 1.0f);
        if (h < 0.5f) {
            float t = h * 2.0f;
            r = 0.30f * (1.0f - t) + 0.08f * t;
            g = 0.18f * (1.0f - t) + 0.42f * t;
            b = 0.10f * (1.0f - t) + 0.48f * t;
        } else {
            float t = (h - 0.5f) * 2.0f;
            r = 0.08f * (1.0f - t) + 0.25f * t;
            g = 0.42f * (1.0f - t) + 0.85f * t;
            b = 0.48f * (1.0f - t) + 1.00f * t;
        }
    } else if (layerMode == 1) {
        // Cloud density qc
        float c = fminf(sQc * 650.0f, 1.0f);
        r = 0.07f + 0.85f * c;
        g = 0.09f + 0.86f * c;
        b = 0.14f + 0.84f * c;
    } else if (layerMode == 2) {
        // Rain qr
        float rn = fminf(sQr * 900.0f, 1.0f);
        r = 0.06f + 0.18f * rn + 0.45f * fmaxf(rn - 0.65f, 0.0f);
        g = 0.09f + 0.55f * rn;
        b = 0.15f + 0.85f * rn;
    } else if (layerMode == 3) {
        // Snow qi + ground snowAccum
        float sn = fminf(sQi * 800.0f + sSacc * 120.0f, 1.0f);
        r = 0.07f + 0.83f * sn;
        g = 0.10f + 0.88f * sn;
        b = 0.16f + 0.84f * sn;
    } else if (layerMode == 4) {
        // Temperature centered around midLayerTempC +/- 18C with 0C white isotherm highlight
        float norm = fminf(fmaxf((sT - (cfg.midLayerTempC - 18.0f)) / 36.0f, 0.0f), 1.0f);
        r = fminf(fmaxf(1.5f * norm - 0.25f, 0.05f), 1.0f);
        g = fmaxf(0.15f, 1.0f - fabsf(2.0f * norm - 1.0f) * 0.85f);
        b = fminf(fmaxf(1.25f - 1.5f * norm, 0.08f), 1.0f);
        if (fabsf(sT) < 0.45f) {
            r = 0.96f; g = 0.98f; b = 1.0f;
        }
    } else if (layerMode == 5) {
        // Wind speed (0..18 tiles/s: indigo -> teal -> emerald -> bright gold)
        float w = fminf(sSpd / 18.0f, 1.0f);
        if (w < 0.5f) {
            float t = w * 2.0f;
            r = 0.10f * (1.0f - t) + 0.12f * t;
            g = 0.08f * (1.0f - t) + 0.58f * t;
            b = 0.26f * (1.0f - t) + 0.56f * t;
        } else {
            float t = (w - 0.5f) * 2.0f;
            r = 0.12f * (1.0f - t) + 0.98f * t;
            g = 0.58f * (1.0f - t) + 0.90f * t;
            b = 0.56f * (1.0f - t) + 0.18f * t;
        }
    } else {
        // Composite weather map
        float cA = fminf(sQc * 500.0f, 0.78f);
        float rA = fminf(sQr * 750.0f, 0.82f);
        float iA = fminf(sQi * 650.0f + sSacc * 120.0f, 0.92f);
        r = r * (1.0f - cA) + 0.86f * cA;
        g = g * (1.0f - cA) + 0.90f * cA;
        b = b * (1.0f - cA) + 0.94f * cA;
        r = r * (1.0f - rA) + 0.20f * rA;
        g = g * (1.0f - rA) + 0.62f * rA;
        b = b * (1.0f - rA) + 1.00f * rA;
        r = r * (1.0f - iA) + 0.85f * iA;
        g = g * (1.0f - iA) + 0.97f * iA;
        b = b * (1.0f - iA) + 1.00f * iA;
    }

    if (showWalls != 0) {
        if (solidFrac > 0.0f) {
            float k = (layerMode == 6) ? (0.45f * solidFrac) : (0.28f * solidFrac);
            r = r * (1.0f - k) + 0.18f * k;
            g = g * (1.0f - k) + 0.17f * k;
            b = b * (1.0f - k) + 0.20f * k;
        }
        if (waterFrac > 0.0f) {
            float k = 0.55f * waterFrac;
            r = r * (1.0f - k) + 0.10f * k;
            g = g * (1.0f - k) + 0.45f * k;
            b = b * (1.0f - k) + 0.95f * k;
        }
    }

    int outBase = (my * mapW + mx) * 4;
    outRgba[outBase + 0] = (uint8_t)(fminf(fmaxf(r, 0.0f), 1.0f) * 255.0f);
    outRgba[outBase + 1] = (uint8_t)(fminf(fmaxf(g, 0.0f), 1.0f) * 255.0f);
    outRgba[outBase + 2] = (uint8_t)(fminf(fmaxf(b, 0.0f), 1.0f) * 255.0f);
    outRgba[outBase + 3] = 255;
}

__global__ void kSampleMapWindGrid(
    int W, int H, int gridW, int gridH,
    const float* __restrict__ u,
    const float* __restrict__ v,
    float* __restrict__ outU,
    float* __restrict__ outV)
{
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= gridW || gy >= gridH) return;

    // 3x3 local average around each wind-arrow anchor point for smooth readable arrows
    float cx = ((float)gx + 0.5f) * (float)W / (float)gridW;
    float cy = ((float)gy + 0.5f) * (float)H / (float)gridH;
    float stepX = fmaxf(1.0f, 0.25f * (float)W / (float)gridW);
    float stepY = fmaxf(1.0f, 0.25f * (float)H / (float)gridH);

    float sumU = 0.0f, sumV = 0.0f;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            float sx = fminf(fmaxf(cx + dx * stepX, 0.5f), (float)W - 0.5f);
            float sy = fminf(fmaxf(cy + dy * stepY, 0.5f), (float)H - 0.5f);
            sumU += sampleU(u, sx, sy, W, H);
            sumV += sampleV(v, sx, sy, W, H);
        }
    }
    int outIdx = gy * gridW + gx;
    outU[outIdx] = sumU / 9.0f;
    outV[outIdx] = sumV / 9.0f;
}

cudaError_t launchPatchStaticHeat(WeatherSimHandle* h, int32_t count) {
    if (count <= 0) return cudaSuccess;
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    kPatchStaticHeat<<<blocks, threads, 0, h->stream>>>(
        h->cfg.width, h->cfg.height,
        h->d.d_heatPatchX, h->d.d_heatPatchY, h->d.d_heatPatchVal, count,
        h->d.heatSource);
    return cudaGetLastError();
}

cudaError_t launchStampDynamicHeaters(WeatherSimHandle* h, int32_t count) {
    size_t N = (size_t)h->cfg.width * (size_t)h->cfg.height;
    cudaError_t err = cudaMemsetAsync(h->d.dynHeat, 0, N * sizeof(float), h->stream);
    if (err != cudaSuccess || count <= 0) return err;
    dim3 block(16, 16);
    dim3 grid(count, 1);
    kStampDynamicHeaters<<<grid, block, 0, h->stream>>>(
        h->cfg.width, h->cfg.height,
        h->d.d_dynSources, count,
        h->d.dynHeat);
    return cudaGetLastError();
}

cudaError_t launchRenderFullMapRGBA8(
    WeatherSimHandle* h,
    int32_t mapW, int32_t mapH,
    int32_t layerMode, int32_t showWalls)
{
    dim3 block(16, 16);
    dim3 grid((mapW + 15) / 16, (mapH + 15) / 16);
    kRenderFullMapRGBA8<<<grid, block, 0, h->stream>>>(
        h->cfg, mapW, mapH, layerMode, showWalls,
        h->d.cellType, h->d.u, h->d.v,
        h->d.T, h->d.qv, h->d.qc, h->d.qr, h->d.qi, h->d.snowAccum,
        h->d.d_fullMapRgba);
    return cudaGetLastError();
}

cudaError_t launchSampleMapWindGrid(
    WeatherSimHandle* h,
    int32_t gridW, int32_t gridH)
{
    dim3 block(16, 16);
    dim3 grid((gridW + 15) / 16, (gridH + 15) / 16);
    kSampleMapWindGrid<<<grid, block, 0, h->stream>>>(
        h->cfg.width, h->cfg.height, gridW, gridH,
        h->d.u, h->d.v,
        h->d.d_windGridU, h->d.d_windGridV);
    return cudaGetLastError();
}

cudaError_t launchResolveFluidEvents(WeatherSimHandle* h) {
    int sub = (h->cfg.subdivision >= 1 && h->cfg.subdivision <= 4) ? h->cfg.subdivision : 2;
    int tilesW = h->cfg.width / sub;
    int tilesH = h->cfg.height / sub;
    if (tilesW <= 0 || tilesH <= 0) return cudaSuccess;
    dim3 block(32, 8);
    dim3 grid((tilesW + block.x - 1) / block.x, (tilesH + block.y - 1) / block.y);
    kResolveFluidEvents<<<grid, block, 0, h->stream>>>(
        h->cfg, tilesW, tilesH, sub,
        h->d.cellType, h->d.evapAccum, h->d.rainAccum,
        h->d.d_fluidEvents, h->d.d_fluidEventCount, h->d.fluidEventCapacity);
    return cudaGetLastError();
}
