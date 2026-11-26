using System;
using System.IO;
using System.Runtime.InteropServices;

namespace WeatherExpansionMod.Native
{
    internal enum WeatherCellType : byte
    {
        Air = 0,
        Solid = 1,
        Water = 2
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct WeatherSimConfig
    {
        public int width;
        public int height;
        public float cellSizeTiles;

        public float midLayerTempC;
        public float bgLapseRateCPerTile;
        public float adiabaticLapseCPerTile;
        public float buoyancyScale;

        public float viscosity;
        public float thermalDiffusivity;
        public float moistureDiffusivity;
        public float vorticityEps;
        public float velocityDamping;
        public float maxSpeedTilesPerSec;

        public float wallTempRelaxRate;
        public float evapRateCoeff;
        public float evapWindFactor;
        public float evapCoolingScale;
        public float latentHeatScale;

        public float autoconvertThresh;
        public float autoconvertRate;
        public float freezeRate;
        public float meltRate;
        public float rainFallSpeed;
        public float snowFallSpeed;
        public float snowDepositRate;

        public int sorIterations;
        public float sorOmega;
        public float pressureWarmStart;
        public int useMacCormack;
        public float initialRelHumidity;

        public float bottomTempBoostC;
        public float topTempDropC;
        public float bottomMoistureSource;
        public float ambientHumidityTarget;
        public float ambientHumidityRate;

        public int subdivision;
        public float evapMassPerWaterTile;
        public float rainMassPerWaterTile;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct WeatherFluidEvent
    {
        public int tileX;
        public int tileY;
        public int delta; // -1 = evaporated water tile, +1 = rainfall spawned water tile
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct WeatherDynamicHeatSource
    {
        public float cx;
        public float cy;
        public float radiusCells;
        public float rateCPerSec;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct WeatherSimStats
    {
        public float lastStepGpuMs;
        public ulong allocatedVramBytes;
        public ulong freeVramBytes;
        public ulong totalVramBytes;
        public int stepCount;
        public int lastCudaError;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct WeatherEntitySample
    {
        public float u;
        public float v;
        public float tempC;
        public float relHumidity;
        public float cloud;
        public float rain;
        public float snow;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct WeatherPrecipCell
    {
        public float u;
        public float v;
        public float tempC;
        public float qc;
        public float qr;
        public float qi;
        public float snowAccum;
        public int isAir;
    }

    internal static class WeatherSimNative
    {
        private const string DllName = "WeatherSimCuda.dll";
        private static IntPtr _nativeModule = IntPtr.Zero;

        public static bool IsNativeLoaded => _nativeModule != IntPtr.Zero;

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern IntPtr LoadLibraryW(string lpFileName);

        public static bool TryPreloadNativeDll(string pluginDirectory, out string errorMessage)
        {
            if (_nativeModule != IntPtr.Zero)
            {
                errorMessage = string.Empty;
                return true;
            }

            string candidate = Path.Combine(pluginDirectory, DllName);
            if (!File.Exists(candidate))
            {
                errorMessage = $"Missing native CUDA library at: {candidate}";
                return false;
            }

            _nativeModule = LoadLibraryW(candidate);
            if (_nativeModule == IntPtr.Zero)
            {
                int winErr = Marshal.GetLastWin32Error();
                errorMessage = $"LoadLibraryW failed for {candidate} (Win32 error {winErr}). Ensure an NVIDIA GPU & compatible driver are present.";
                return false;
            }

            errorMessage = string.Empty;
            return true;
        }

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void WeatherSim_DefaultConfig(out WeatherSimConfig outConfig, int width, int height);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr WeatherSim_Create(ref WeatherSimConfig config);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void WeatherSim_Destroy(IntPtr handle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void WeatherSim_UpdateConfig(IntPtr handle, ref WeatherSimConfig config);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void WeatherSim_SetMidLayerTemp(IntPtr handle, float midLayerTempC);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int WeatherSim_ResetFields(IntPtr handle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int WeatherSim_UploadCellsFull(IntPtr handle, byte[] cellTypes);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int WeatherSim_UploadCellsPatch(IntPtr handle, int[] xs, int[] ys, byte[] types, int count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int WeatherSim_UploadHeatSources(IntPtr handle, float[] heatRateCPerSec);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int WeatherSim_UploadStaticHeatPatch(
            IntPtr handle,
            int[] xs,
            int[] ys,
            float[] rates,
            int count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int WeatherSim_UploadDynamicHeaters(
            IntPtr handle,
            WeatherDynamicHeatSource[]? sources,
            int count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int WeatherSim_UploadWindSources(IntPtr handle, float[]? forceX, float[]? forceY);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int WeatherSim_Step(IntPtr handle, float dt);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int WeatherSim_SamplePoints(
            IntPtr handle,
            float[] cx,
            float[] cy,
            [Out] WeatherEntitySample[] outSamples,
            int count);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int WeatherSim_ReadViewWindowRGBA8(
            IntPtr handle,
            int x0,
            int y0,
            int winW,
            int winH,
            int mode,
            float playerCx,
            float playerCy,
            float playerClearInnerCells,
            float playerClearOuterCells,
            float playerMinFogFactor,
            float wallFogMultiplier,
            [Out] byte[] outRgbaBytes);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int WeatherSim_ReadViewPrecipGrid(
            IntPtr handle,
            int x0,
            int y0,
            int winW,
            int winH,
            int outW,
            int outH,
            [Out] WeatherPrecipCell[] outCells);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int WeatherSim_RenderFullMapRGBA8(
            IntPtr handle,
            int mapW,
            int mapH,
            int layerMode,
            int showWalls,
            [Out] byte[] outRgbaBytes);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int WeatherSim_SampleMapWindGrid(
            IntPtr handle,
            int gridW,
            int gridH,
            [Out] float[] outU,
            [Out] float[] outV);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int WeatherSim_PopFluidEvents(
            IntPtr handle,
            [Out] WeatherFluidEvent[] outEvents,
            int maxEvents);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int WeatherSim_GetStats(IntPtr handle, out WeatherSimStats outStats);
    }
}
