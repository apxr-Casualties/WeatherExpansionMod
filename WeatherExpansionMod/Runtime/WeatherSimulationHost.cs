using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection;
using System.Threading;
using HarmonyLib;
using UnityEngine;
using WeatherExpansionMod.Native;

namespace WeatherExpansionMod.Runtime
{
    internal struct ViewportRequest
    {
        public bool Valid;
        public int X0, Y0, WinW, WinH;
        public int Mode;
        public float PlayerCx, PlayerCy;
        public float ClearInnerCells, ClearOuterCells;
        public float MinFogFactor, WallFogMultiplier;
        public bool WantParticles;
    }

    /// <summary>
    /// Owns the CUDA 2D weather simulation on a dedicated background worker thread so Unity's
    /// Main Thread NEVER stalls waiting for GPU steps (~0.2 ms main-thread cost per frame!).
    /// Also synchronizes terrain, FluidManager water-cycle (evaporation depletion & rainfall generation),
    /// and aerodynamic wind forces on nearby Rigidbody2D objects.
    /// </summary>
    internal sealed class WeatherSimulationHost : MonoBehaviour
    {
        public static WeatherSimulationHost? Instance { get; private set; }

        public const int PrecipGridW = 44;
        public const int PrecipGridH = 28;

        private static readonly FieldInfo? WorldBlocksField =
            AccessTools.Field(typeof(WorldGeneration), "worldBlocks");

        private IntPtr _simHandle = IntPtr.Zero;
        private WeatherSimConfig _cfg;
        private WeatherSimStats _lastStats;

        // Mapping between game block grid and CUDA simulation grid
        public int SimWidth { get; private set; }
        public int SimHeight { get; private set; }
        public int Subdivision { get; private set; } = 2;
        public int OriginTileX { get; private set; }
        public int OriginTileY { get; private set; }
        public int CoveredTilesW { get; private set; }
        public int CoveredTilesH { get; private set; }
        public bool IsWindowedFallback { get; private set; }

        public bool IsReady => _simHandle != IntPtr.Zero;
        public string StatusMessage { get; private set; } = "等待地层生成...";
        public WeatherLayerProfile ActiveProfile { get; private set; } = new WeatherLayerProfile();
        public WeatherEntitySample PlayerSample { get; private set; }
        public float PlayerEffectiveTempC { get; private set; } = 24f;
        public float VanillaMidLayerTempC { get; private set; } = 24f;
        public WeatherSimStats LastStats => _lastStats;
        public int TotalEvaporatedTiles { get; private set; }
        public int TotalRainWaterTiles { get; private set; }

        private WorldGeneration? _boundWorld;
        private bool _wasGenerating = true;
        private bool _pendingRebuild;
        private float _fluidSyncTimer;
        private float _recenterCheckTimer;

        // Cached tile-level cell types for cheap FluidManager diff scanning
        private byte[]? _tileCellTypes;

        // Saved vanilla fungalRainMap to suppress Layer 6/7 fake rain cleanly
        private Texture2D? _black1x1Tex;
        private Texture2D? _savedVanillaRainMap;
        private WorldGeneration? _rainMapPatchedWorld;

        // ==================== Async Worker Thread State ====================
        private Thread? _workerThread;
        private volatile bool _workerStop;
        private readonly object _inLock = new object();
        private readonly object _outLock = new object();

        // Main -> Worker inputs
        private float _inMidLayerTempC = 20f;
        private bool _inPaused;
        private float _inTargetSimHz = 30f;
        private ViewportRequest _inViewReq;

        private const int MaxWindBodies = 256;
        private int _inSampleCount;
        private readonly float[] _inSampleCx = new float[MaxWindBodies + 1];
        private readonly float[] _inSampleCy = new float[MaxWindBodies + 1];
        private readonly int[] _inSampleRbIds = new int[MaxWindBodies + 1];

        private readonly List<int> _inPatchX = new List<int>(64);
        private readonly List<int> _inPatchY = new List<int>(64);
        private readonly List<byte> _inPatchType = new List<byte>(64);

        // Worker -> Main double-buffered outputs
        private WeatherSimStats _outStats;
        private WeatherEntitySample _outPlayerSample;
        private bool _outHasPlayerSample;

        private int _outWindCount;
        private readonly int[] _outWindRbIds = new int[MaxWindBodies + 1];
        private readonly WeatherEntitySample[] _outWindSamples = new WeatherEntitySample[MaxWindBodies + 1];
        private readonly Dictionary<int, WeatherEntitySample> _windLookup = new Dictionary<int, WeatherEntitySample>(MaxWindBodies);

        private byte[] _outRgba = Array.Empty<byte>();
        private int _outRgbaW, _outRgbaH, _outRgbaX0, _outRgbaY0, _outRgbaMode, _outRgbaSeq;

        private readonly WeatherPrecipCell[] _outPrecip = new WeatherPrecipCell[PrecipGridW * PrecipGridH];
        private int _outPrecipX0, _outPrecipY0, _outPrecipW, _outPrecipH, _outPrecipSeq;

        private readonly List<WeatherFluidEvent> _outFluidEvents = new List<WeatherFluidEvent>(256);

        // Main-thread reusable physics buffers
        private readonly Collider2D[] _overlapColliders = new Collider2D[512];
        private readonly Rigidbody2D[] _activeWindBodies = new Rigidbody2D[MaxWindBodies];
        private readonly float[] _activeWindAreas = new float[MaxWindBodies];
        private readonly HashSet<int> _seenRbIds = new HashSet<int>();
        private readonly List<WeatherFluidEvent> _mainFluidScratch = new List<WeatherFluidEvent>(256);

        public static void EnsureAttachedToWorld(WorldGeneration world)
        {
            if (world == null) return;
            if (Instance != null && Instance._boundWorld == world) return;
            if (Instance != null)
            {
                Instance.ForceShutdown();
                Destroy(Instance.gameObject);
            }

            var hostGo = new GameObject("WeatherExpansion_Runtime");
            hostGo.transform.SetParent(world.transform, false);
            var host = hostGo.AddComponent<WeatherSimulationHost>();
            host._boundWorld = world;
            hostGo.AddComponent<WeatherVisualRenderer>();
            hostGo.AddComponent<WeatherDebugHud>();

            WeatherPlugin.Log?.LogInfo(
                $"[WeatherExpansion] Attached async runtime host to WorldGeneration (biomeDepth={world.biomeDepth}, worldExists={world.worldExists}, generating={world.generatingWorld}).");
        }

        public void ForceShutdown()
        {
            DestroySimulation();
            RestoreVanillaRainMap();
        }

        private void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(gameObject);
                return;
            }
            Instance = this;

            _black1x1Tex = new Texture2D(1, 1, TextureFormat.RGBA32, false);
            _black1x1Tex.SetPixel(0, 0, Color.black);
            _black1x1Tex.Apply(false, false);
        }

        private void OnDestroy()
        {
            DestroySimulation();
            RestoreVanillaRainMap();
            if (_black1x1Tex != null) Destroy(_black1x1Tex);
            if (Instance == this) Instance = null;
        }

        public void RequestRebuildForCurrentWorld() => _pendingRebuild = true;

        public void NotifyWorldRegenerating()
        {
            DestroySimulation();
            _pendingRebuild = false;
            _wasGenerating = true;
        }

        public void NotifyBlockChanged(Vector2Int tilePos, ushort blockId)
        {
            if (!IsReady) return;
            int localTx = tilePos.x - OriginTileX;
            int localTy = tilePos.y - OriginTileY;
            if (localTx < 0 || localTx >= CoveredTilesW || localTy < 0 || localTy >= CoveredTilesH)
                return;

            byte cell = DetermineTileCellType(tilePos.x, tilePos.y, blockId, FluidManager.main?.fluid);
            if (_tileCellTypes != null)
            {
                int tIdx = localTy * CoveredTilesW + localTx;
                if (tIdx >= 0 && tIdx < _tileCellTypes.Length)
                    _tileCellTypes[tIdx] = cell;
            }

            QueueTilePatch(localTx, localTy, cell);
        }

        private void QueueTilePatch(int localTx, int localTy, byte cell)
        {
            int sub = Subdivision;
            lock (_inLock)
            {
                for (int dy = 0; dy < sub; dy++)
                {
                    for (int dx = 0; dx < sub; dx++)
                    {
                        _inPatchX.Add(localTx * sub + dx);
                        _inPatchY.Add(localTy * sub + dy);
                        _inPatchType.Add(cell);
                    }
                }
            }
        }

        public bool WorldToSimCell(Vector2 worldPos, out float cx, out float cy)
        {
            cx = 0f;
            cy = 0f;
            if (_boundWorld == null || !IsReady) return false;
            float tileX = worldPos.x + _boundWorld.halfWidth;
            float tileY = worldPos.y + _boundWorld.halfHeight;
            cx = (tileX - OriginTileX) * Subdivision;
            cy = (tileY - OriginTileY) * Subdivision;
            return cx >= 0f && cx < SimWidth && cy >= 0f && cy < SimHeight;
        }

        public Vector2 SimCellToWorld(float cx, float cy)
        {
            if (_boundWorld == null) return Vector2.zero;
            float tileX = OriginTileX + cx / Subdivision;
            float tileY = OriginTileY + cy / Subdivision;
            return new Vector2(tileX - _boundWorld.halfWidth, tileY - _boundWorld.halfHeight);
        }

        // Called from WeatherVisualRenderer.LateUpdate (main thread, non-blocking!)
        public void SubmitViewportRequest(ViewportRequest req)
        {
            lock (_inLock)
            {
                _inViewReq = req;
            }
        }

        public bool TryCopyLatestOverlay(
            ref int lastSeq,
            ref byte[] dstRgba,
            out int x0, out int y0, out int winW, out int winH, out int mode)
        {
            x0 = y0 = winW = winH = mode = 0;
            lock (_outLock)
            {
                if (_outRgbaSeq == 0 || _outRgbaSeq == lastSeq || _outRgbaW <= 0 || _outRgbaH <= 0)
                    return false;

                int needed = _outRgbaW * _outRgbaH * 4;
                if (dstRgba.Length != needed)
                    dstRgba = new byte[needed];
                Buffer.BlockCopy(_outRgba, 0, dstRgba, 0, needed);

                x0 = _outRgbaX0;
                y0 = _outRgbaY0;
                winW = _outRgbaW;
                winH = _outRgbaH;
                mode = _outRgbaMode;
                lastSeq = _outRgbaSeq;
                return true;
            }
        }

        public bool TryCopyLatestPrecipGrid(
            WeatherPrecipCell[] dstCells,
            out int x0, out int y0, out int winW, out int winH)
        {
            x0 = y0 = winW = winH = 0;
            lock (_outLock)
            {
                if (_outPrecipSeq == 0 || _outPrecipW <= 0 || _outPrecipH <= 0)
                    return false;

                Array.Copy(_outPrecip, dstCells, _outPrecip.Length);
                x0 = _outPrecipX0;
                y0 = _outPrecipY0;
                winW = _outPrecipW;
                winH = _outPrecipH;
                return true;
            }
        }

        private void Update()
        {
            try
            {
                UpdateInternal();
            }
            catch (Exception ex)
            {
                StatusMessage = $"运行异常: {ex.GetType().Name}";
                WeatherPlugin.Log?.LogError($"[WeatherExpansion] Exception in Update: {ex}");
            }
        }

        private void UpdateInternal()
        {
            if (!WeatherConfig.Enabled.Value || !WeatherSimNative.IsNativeLoaded)
            {
                StatusMessage = WeatherSimNative.IsNativeLoaded ? "已在配置中禁用" : "CUDA 原生库未加载";
                if (IsReady) DestroySimulation();
                return;
            }

            var world = WorldGeneration.world;
            if (world == null || !world.worldExists || world.width == 0 || world.height == 0)
            {
                StatusMessage = "等待世界实例化...";
                if (IsReady) DestroySimulation();
                _wasGenerating = true;
                return;
            }

            if (world != _boundWorld)
            {
                DestroySimulation();
                _boundWorld = world;
                _wasGenerating = true;
            }

            if (world.generatingWorld)
            {
                StatusMessage = "地层生成中...";
                if (IsReady) DestroySimulation();
                _wasGenerating = true;
                return;
            }

            if (_wasGenerating || _pendingRebuild || !IsReady)
            {
                _wasGenerating = false;
                _pendingRebuild = false;
                BuildSimulationForWorld(world, recenterOnly: false);
            }

            if (!IsReady) return;
            StatusMessage = "异步运行中";

            ApplyVanillaWeatherSuppression(world);

            VanillaMidLayerTempC = world.ambientTemperature;
            bool paused = (Time.timeScale <= 0f || world.cutsceneActive);

            lock (_inLock)
            {
                _inMidLayerTempC = VanillaMidLayerTempC;
                _inPaused = paused;
                _inTargetSimHz = Mathf.Clamp(WeatherConfig.SimHz.Value, 10f, 60f);

                // Always keep player coordinate fresh in sample slot 0
                if (PlayerCamera.main != null && PlayerCamera.main.body != null &&
                    WorldToSimCell(PlayerCamera.main.body.transform.position, out float pcx, out float pcy))
                {
                    _inSampleCx[0] = pcx;
                    _inSampleCy[0] = pcy;
                    _inSampleRbIds[0] = -1;
                    if (_inSampleCount < 1) _inSampleCount = 1;
                }
            }

            // Apply completed GPU water-cycle events (evaporation depletion & rainfall water spawn) to FluidManager
            if (WeatherConfig.EnableWaterCycleFluidSync.Value && !paused)
            {
                ApplyCompletedFluidEvents(world);
            }

            // Periodic FluidManager movement sync around camera (~0.15ms on 128x128 tiles)
            _fluidSyncTimer += Time.deltaTime;
            if (_fluidSyncTimer >= WeatherConfig.FluidSyncIntervalSec.Value)
            {
                _fluidSyncTimer = 0f;
                SyncNearbyFluidChanges(world);
            }

            if (IsWindowedFallback)
            {
                _recenterCheckTimer += Time.deltaTime;
                if (_recenterCheckTimer >= 1.5f)
                {
                    _recenterCheckTimer = 0f;
                    CheckWindowedRecenter(world);
                }
            }

            // Read latest stats & player sample from worker (non-blocking, <0.01ms)
            WeatherEntitySample pSample;
            bool hasPlayer;
            lock (_outLock)
            {
                _lastStats = _outStats;
                pSample = _outPlayerSample;
                hasPlayer = _outHasPlayerSample;
            }

            if (hasPlayer)
            {
                UpdatePlayerLocalWeather(world, pSample);
            }
        }

        private void FixedUpdate()
        {
            if (!IsReady || !WeatherConfig.Enabled.Value || !WeatherConfig.EnableWindForceOnObjects.Value)
                return;
            if (_boundWorld == null || _boundWorld.generatingWorld || _boundWorld.cutsceneActive)
                return;
            if (Time.timeScale <= 0f)
                return;

            ApplyWindForcesAsync();
        }

        private void BuildSimulationForWorld(WorldGeneration world, bool recenterOnly)
        {
            DestroySimulation();

            int mapW = (int)world.width;
            int mapH = (int)world.height;
            if (mapW < 8 || mapH < 8) return;

            int preferredSub = Mathf.Clamp(WeatherConfig.PreferredSubdivision.Value, 1, 2);
            int maxCells = Mathf.Max(WeatherConfig.MaxSimCells.Value, 262_144);

            long cellsFullPref = (long)mapW * mapH * preferredSub * preferredSub;
            long cellsFull1x1 = (long)mapW * mapH;

            if (cellsFullPref <= maxCells)
            {
                Subdivision = preferredSub;
                OriginTileX = 0;
                OriginTileY = 0;
                CoveredTilesW = mapW;
                CoveredTilesH = mapH;
                IsWindowedFallback = false;
            }
            else if (cellsFull1x1 <= maxCells)
            {
                Subdivision = 1;
                OriginTileX = 0;
                OriginTileY = 0;
                CoveredTilesW = mapW;
                CoveredTilesH = mapH;
                IsWindowedFallback = false;
            }
            else
            {
                Subdivision = preferredSub;
                int winTiles = Mathf.Clamp(WeatherConfig.FallbackWindowTiles.Value, 128, Math.Min(mapW, mapH));
                while ((long)winTiles * winTiles * Subdivision * Subdivision > maxCells && Subdivision > 1)
                    Subdivision--;

                Vector2Int centerTile = new Vector2Int(mapW / 2, mapH / 2);
                if (PlayerCamera.main != null && PlayerCamera.main.body != null)
                    centerTile = world.WorldToBlockPos(PlayerCamera.main.body.transform.position);

                CoveredTilesW = Math.Min(winTiles, mapW);
                CoveredTilesH = Math.Min(winTiles, mapH);
                OriginTileX = Mathf.Clamp(centerTile.x - CoveredTilesW / 2, 0, mapW - CoveredTilesW);
                OriginTileY = Mathf.Clamp(centerTile.y - CoveredTilesH / 2, 0, mapH - CoveredTilesH);
                IsWindowedFallback = true;
            }

            SimWidth = CoveredTilesW * Subdivision;
            SimHeight = CoveredTilesH * Subdivision;

            WeatherSimNative.WeatherSim_DefaultConfig(out _cfg, SimWidth, SimHeight);
            _cfg.cellSizeTiles = 1f / Subdivision;
            _cfg.midLayerTempC = world.ambientTemperature;
            _cfg.sorIterations = Mathf.Clamp(WeatherConfig.SorIterations.Value, 10, 48);
            _cfg.subdivision = Subdivision;
            _cfg.evapMassPerWaterTile = WeatherConfig.EnableWaterCycleFluidSync.Value
                ? WeatherConfig.EvapMassPerWaterTile.Value : 0f;
            _cfg.rainMassPerWaterTile = WeatherConfig.EnableWaterCycleFluidSync.Value
                ? WeatherConfig.RainMassPerWaterTile.Value : 0f;

            ActiveProfile = WeatherLayerProfile.DetectFromWorld(world);
            if (WeatherConfig.OverrideVanillaLayerWeather.Value)
            {
                ActiveProfile.ApplyToConfig(ref _cfg);
            }

            _simHandle = WeatherSimNative.WeatherSim_Create(ref _cfg);
            if (_simHandle == IntPtr.Zero)
            {
                WeatherPlugin.Log?.LogError($"Failed to create CUDA weather simulation ({SimWidth}x{SimHeight}).");
                return;
            }

            TotalEvaporatedTiles = 0;
            TotalRainWaterTiles = 0;

            UploadTerrainAndFluids(world);
            UploadHeatSourcesOnce(world);
            WeatherSimNative.WeatherSim_ResetFields(_simHandle);

            // Warmup on init (suppress fluid events during pre-sim warmup by draining them after)
            int warmup = recenterOnly ? Math.Min(20, WeatherConfig.WarmupStepsOnLayerGen.Value) : WeatherConfig.WarmupStepsOnLayerGen.Value;
            for (int i = 0; i < warmup; i++)
            {
                WeatherSimNative.WeatherSim_Step(_simHandle, 1f / 30f);
            }
            var discard = new WeatherFluidEvent[4096];
            WeatherSimNative.WeatherSim_PopFluidEvents(_simHandle, discard, discard.Length);

            WeatherSimNative.WeatherSim_GetStats(_simHandle, out _lastStats);
            lock (_outLock)
            {
                _outStats = _lastStats;
                _outRgbaSeq = 0;
                _outPrecipSeq = 0;
                _outHasPlayerSample = false;
                _outFluidEvents.Clear();
            }

            lock (_inLock)
            {
                _inMidLayerTempC = world.ambientTemperature;
                _inPaused = false;
                _inTargetSimHz = Mathf.Clamp(WeatherConfig.SimHz.Value, 10f, 60f);
                _inPatchX.Clear();
                _inPatchY.Clear();
                _inPatchType.Clear();
            }

            // Start dedicated async CUDA worker thread
            _workerStop = false;
            _workerThread = new Thread(WorkerLoop)
            {
                IsBackground = true,
                Name = "WeatherSimCudaWorker"
            };
            _workerThread.Start();

            WeatherPlugin.Log?.LogInfo(
                $"[WeatherExpansion] Initialized async CUDA grid {SimWidth}x{SimHeight} " +
                $"(tiles {CoveredTilesW}x{CoveredTilesH}, subdiv={Subdivision}x{Subdivision}, SOR={_cfg.sorIterations}, " +
                $"T_mid={_cfg.midLayerTempC:F1}C, profile=[{ActiveProfile.SummaryTags}], " +
                $"VRAM={_lastStats.allocatedVramBytes / (1024f * 1024f):F1}MB, step={_lastStats.lastStepGpuMs:F2}ms).");
        }

        private void WorkerLoop()
        {
            // Worker-local scratch buffers (never touched by main thread)
            var localSampleCx = new float[MaxWindBodies + 1];
            var localSampleCy = new float[MaxWindBodies + 1];
            var localSampleIds = new int[MaxWindBodies + 1];
            var localSampleOut = new WeatherEntitySample[MaxWindBodies + 1];
            var localPrecip = new WeatherPrecipCell[PrecipGridW * PrecipGridH];
            var localEvents = new WeatherFluidEvent[512];
            byte[] localRgba = Array.Empty<byte>();

            var sw = Stopwatch.StartNew();

            while (!_workerStop)
            {
                IntPtr handle = _simHandle;
                if (handle == IntPtr.Zero) break;

                float midTemp;
                bool paused;
                float targetHz;
                ViewportRequest viewReq;
                int sampleCount;
                int[]? patchX = null;
                int[]? patchY = null;
                byte[]? patchT = null;

                lock (_inLock)
                {
                    midTemp = _inMidLayerTempC;
                    paused = _inPaused;
                    targetHz = _inTargetSimHz;
                    viewReq = _inViewReq;

                    sampleCount = _inSampleCount;
                    if (sampleCount > 0)
                    {
                        Array.Copy(_inSampleCx, localSampleCx, sampleCount);
                        Array.Copy(_inSampleCy, localSampleCy, sampleCount);
                        Array.Copy(_inSampleRbIds, localSampleIds, sampleCount);
                    }

                    if (_inPatchX.Count > 0)
                    {
                        patchX = _inPatchX.ToArray();
                        patchY = _inPatchY.ToArray();
                        patchT = _inPatchType.ToArray();
                        _inPatchX.Clear();
                        _inPatchY.Clear();
                        _inPatchType.Clear();
                    }
                }

                if (paused)
                {
                    Thread.Sleep(16);
                    continue;
                }

                double t0 = sw.Elapsed.TotalMilliseconds;
                float simDt = 1f / Math.Max(targetHz, 10f);

                try
                {
                    WeatherSimNative.WeatherSim_SetMidLayerTemp(handle, midTemp);

                    if (patchX != null && patchY != null && patchT != null)
                    {
                        WeatherSimNative.WeatherSim_UploadCellsPatch(handle, patchX, patchY, patchT, patchX.Length);
                    }

                    int rc = WeatherSimNative.WeatherSim_Step(handle, simDt);
                    if (rc != 0)
                    {
                        WeatherPlugin.Log?.LogWarning($"[WeatherExpansion] Worker WeatherSim_Step CUDA error {rc}");
                        Thread.Sleep(50);
                        continue;
                    }

                    // Sample player + active rigidbodies in one small GPU batch
                    bool sampledOk = false;
                    if (sampleCount > 0)
                    {
                        sampledOk = WeatherSimNative.WeatherSim_SamplePoints(
                            handle, localSampleCx, localSampleCy, localSampleOut, sampleCount) == 0;
                    }

                    // Pack RGBA8 viewport & precip grid directly on worker thread
                    bool rgbaOk = false;
                    bool precipOk = false;
                    if (viewReq.Valid && viewReq.WinW > 0 && viewReq.WinH > 0)
                    {
                        if (viewReq.Mode != (int)WeatherOverlayMode.Off)
                        {
                            int needed = viewReq.WinW * viewReq.WinH * 4;
                            if (localRgba.Length != needed)
                                localRgba = new byte[needed];
                            rgbaOk = WeatherSimNative.WeatherSim_ReadViewWindowRGBA8(
                                handle, viewReq.X0, viewReq.Y0, viewReq.WinW, viewReq.WinH, viewReq.Mode,
                                viewReq.PlayerCx, viewReq.PlayerCy,
                                viewReq.ClearInnerCells, viewReq.ClearOuterCells,
                                viewReq.MinFogFactor, viewReq.WallFogMultiplier,
                                localRgba) == 0;
                        }

                        if (viewReq.WantParticles)
                        {
                            precipOk = WeatherSimNative.WeatherSim_ReadViewPrecipGrid(
                                handle, viewReq.X0, viewReq.Y0, viewReq.WinW, viewReq.WinH,
                                PrecipGridW, PrecipGridH, localPrecip) == 0;
                        }
                    }

                    // Drain discrete hydrological water-cycle events (-1 evap / +1 rainfall)
                    int evCount = WeatherSimNative.WeatherSim_PopFluidEvents(handle, localEvents, localEvents.Length);
                    WeatherSimNative.WeatherSim_GetStats(handle, out WeatherSimStats st);

                    // Publish completed results atomically (<0.02ms lock)
                    lock (_outLock)
                    {
                        _outStats = st;

                        if (sampledOk && sampleCount > 0)
                        {
                            _outPlayerSample = localSampleOut[0];
                            _outHasPlayerSample = true;

                            _outWindCount = sampleCount;
                            Array.Copy(localSampleIds, _outWindRbIds, sampleCount);
                            Array.Copy(localSampleOut, _outWindSamples, sampleCount);
                        }

                        if (rgbaOk)
                        {
                            int bytes = viewReq.WinW * viewReq.WinH * 4;
                            if (_outRgba.Length != bytes)
                                _outRgba = new byte[bytes];
                            Buffer.BlockCopy(localRgba, 0, _outRgba, 0, bytes);
                            _outRgbaX0 = viewReq.X0;
                            _outRgbaY0 = viewReq.Y0;
                            _outRgbaW = viewReq.WinW;
                            _outRgbaH = viewReq.WinH;
                            _outRgbaMode = viewReq.Mode;
                            _outRgbaSeq++;
                        }

                        if (precipOk)
                        {
                            Array.Copy(localPrecip, _outPrecip, _outPrecip.Length);
                            _outPrecipX0 = viewReq.X0;
                            _outPrecipY0 = viewReq.Y0;
                            _outPrecipW = viewReq.WinW;
                            _outPrecipH = viewReq.WinH;
                            _outPrecipSeq++;
                        }

                        if (evCount > 0)
                        {
                            for (int i = 0; i < evCount && _outFluidEvents.Count < 2048; i++)
                                _outFluidEvents.Add(localEvents[i]);
                        }
                    }
                }
                catch (Exception ex)
                {
                    WeatherPlugin.Log?.LogError($"[WeatherExpansion] Worker thread exception: {ex}");
                    Thread.Sleep(100);
                }

                // Pace worker to targetHz (never spin-burns GPU past target tick rate)
                double elapsedMs = sw.Elapsed.TotalMilliseconds - t0;
                double budgetMs = 1000.0 / Math.Max(targetHz, 10f);
                int sleepMs = (int)(budgetMs - elapsedMs);
                if (sleepMs > 1)
                    Thread.Sleep(sleepMs);
            }
        }

        private void ApplyCompletedFluidEvents(WorldGeneration world)
        {
            if (FluidManager.main == null || FluidManager.main.fluid == null) return;

            _mainFluidScratch.Clear();
            lock (_outLock)
            {
                if (_outFluidEvents.Count == 0) return;
                _mainFluidScratch.AddRange(_outFluidEvents);
                _outFluidEvents.Clear();
            }

            byte[,] fluid = FluidManager.main.fluid;
            int fw = fluid.GetLength(0);
            int fh = fluid.GetLength(1);
            ushort[,]? blocks = WorldBlocksField?.GetValue(world) as ushort[,];

            for (int i = 0; i < _mainFluidScratch.Count; i++)
            {
                WeatherFluidEvent ev = _mainFluidScratch[i];
                int gx = OriginTileX + ev.tileX;
                int gy = OriginTileY + ev.tileY;
                if (gx < 1 || gx >= fw - 1 || gy < 1 || gy >= fh - 1) continue;

                int tIdx = ev.tileY * CoveredTilesW + ev.tileX;

                if (ev.delta < 0)
                {
                    byte f = fluid[gx, gy];
                    if (f == 1 || f == 2 || f == 5)
                    {
                        fluid[gx, gy] = 0;
                        if (_tileCellTypes != null && tIdx >= 0 && tIdx < _tileCellTypes.Length)
                            _tileCellTypes[tIdx] = (byte)WeatherCellType.Air;
                        TotalEvaporatedTiles++;
                    }
                }
                else if (ev.delta > 0)
                {
                    ushort b = blocks != null ? blocks[gx, gy] : world.GetBlock(new Vector2Int(gx, gy));
                    if (b == 0 && fluid[gx, gy] == 0)
                    {
                        fluid[gx, gy] = 1; // Groundwater puddle formed by rainfall!
                        if (_tileCellTypes != null && tIdx >= 0 && tIdx < _tileCellTypes.Length)
                            _tileCellTypes[tIdx] = (byte)WeatherCellType.Water;
                        QueueTilePatch(ev.tileX, ev.tileY, (byte)WeatherCellType.Water);
                        TotalRainWaterTiles++;
                    }
                }
            }
        }

        private void UploadTerrainAndFluids(WorldGeneration world)
        {
            ushort[,]? blocks = WorldBlocksField?.GetValue(world) as ushort[,];
            byte[,]? fluid = FluidManager.main?.fluid;

            byte[] cells = new byte[SimWidth * SimHeight];
            _tileCellTypes = new byte[CoveredTilesW * CoveredTilesH];
            int sub = Subdivision;
            int solidTiles = 0;
            int waterTiles = 0;

            for (int ty = 0; ty < CoveredTilesH; ty++)
            {
                int gy = OriginTileY + ty;
                int tileRow = ty * CoveredTilesW;
                for (int tx = 0; tx < CoveredTilesW; tx++)
                {
                    int gx = OriginTileX + tx;
                    ushort b = blocks != null ? blocks[gx, gy] : world.GetBlock(new Vector2Int(gx, gy));
                    byte cellType = DetermineTileCellType(gx, gy, b, fluid);
                    _tileCellTypes[tileRow + tx] = cellType;

                    if (cellType == 0) continue;
                    if (cellType == (byte)WeatherCellType.Solid) solidTiles++;
                    else if (cellType == (byte)WeatherCellType.Water) waterTiles++;

                    int baseSy = ty * sub;
                    int baseSx = tx * sub;
                    for (int dy = 0; dy < sub; dy++)
                    {
                        int row = (baseSy + dy) * SimWidth + baseSx;
                        for (int dx = 0; dx < sub; dx++)
                        {
                            cells[row + dx] = cellType;
                        }
                    }
                }
            }

            WeatherSimNative.WeatherSim_UploadCellsFull(_simHandle, cells);
            WeatherPlugin.Log?.LogInfo(
                $"[WeatherExpansion] Uploaded terrain mask: solidTiles={solidTiles}, waterTiles={waterTiles}, airTiles={CoveredTilesW * CoveredTilesH - solidTiles - waterTiles}.");
        }

        private void SyncNearbyFluidChanges(WorldGeneration world)
        {
            if (!IsReady || _tileCellTypes == null || FluidManager.main == null || FluidManager.main.fluid == null)
                return;

            ushort[,]? blocks = WorldBlocksField?.GetValue(world) as ushort[,];
            byte[,] fluid = FluidManager.main.fluid;

            var range = FluidManager.main.SimulationRange();
            int minX = Mathf.Max(OriginTileX, range.Item1.min);
            int maxX = Mathf.Min(OriginTileX + CoveredTilesW - 1, range.Item1.max);
            int minY = Mathf.Max(OriginTileY, range.Item2.min);
            int maxY = Mathf.Min(OriginTileY + CoveredTilesH - 1, range.Item2.max);
            int sub = Subdivision;

            lock (_inLock)
            {
                for (int gy = minY; gy <= maxY; gy++)
                {
                    int localTy = gy - OriginTileY;
                    int tileRow = localTy * CoveredTilesW;
                    for (int gx = minX; gx <= maxX; gx++)
                    {
                        int localTx = gx - OriginTileX;
                        ushort b = blocks != null ? blocks[gx, gy] : world.GetBlock(new Vector2Int(gx, gy));
                        byte cur = DetermineTileCellType(gx, gy, b, fluid);
                        int tIdx = tileRow + localTx;
                        if (cur == _tileCellTypes[tIdx]) continue;

                        _tileCellTypes[tIdx] = cur;
                        for (int dy = 0; dy < sub; dy++)
                        {
                            for (int dx = 0; dx < sub; dx++)
                            {
                                _inPatchX.Add(localTx * sub + dx);
                                _inPatchY.Add(localTy * sub + dy);
                                _inPatchType.Add(cur);
                            }
                        }
                    }
                }
            }
        }

        private static byte DetermineTileCellType(int gx, int gy, ushort blockId, byte[,]? fluid)
        {
            if (blockId != 0)
                return (byte)WeatherCellType.Solid;

            if (fluid != null && gx >= 0 && gx < fluid.GetLength(0) && gy >= 0 && gy < fluid.GetLength(1))
            {
                byte f = fluid[gx, gy];
                if (f == 1 || f == 2 || f == 5)
                    return (byte)WeatherCellType.Water;
                if (f == 6)
                    return (byte)WeatherCellType.Solid;
            }
            return (byte)WeatherCellType.Air;
        }

        private void UploadHeatSourcesOnce(WorldGeneration world)
        {
            if (!IsReady) return;
            float[] heat = new float[SimWidth * SimHeight];
            ushort[,]? blocks = WorldBlocksField?.GetValue(world) as ushort[,];
            byte[,]? fluid = FluidManager.main?.fluid;
            int sub = Subdivision;

            for (int ty = 1; ty < CoveredTilesH - 1; ty++)
            {
                int gy = OriginTileY + ty;
                for (int tx = 1; tx < CoveredTilesW - 1; tx++)
                {
                    int gx = OriginTileX + tx;
                    float rate = 0f;

                    if (fluid != null && gx < fluid.GetLength(0) && gy < fluid.GetLength(1) && fluid[gx, gy] == 6)
                    {
                        rate = 5.5f;
                    }
                    else
                    {
                        ushort b = blocks != null ? blocks[gx, gy] : world.GetBlock(new Vector2Int(gx, gy));
                        if (b != 0)
                        {
                            BlockInfo? info = world.GetBlockInfo(b);
                            if (info != null && info.hot)
                                rate = 3.0f;
                        }
                    }

                    if (rate <= 0f) continue;

                    StampTileHeat(heat, tx, ty + 1, sub, rate);
                    StampTileHeat(heat, tx - 1, ty, sub, rate * 0.45f);
                    StampTileHeat(heat, tx + 1, ty, sub, rate * 0.45f);
                }
            }

            Heater[] heaters = FindObjectsOfType<Heater>();
            foreach (var heater in heaters)
            {
                if (heater == null || !heater.isActiveAndEnabled) continue;
                if (!WorldToSimCell(heater.transform.position, out float hcx, out float hcy)) continue;

                float radiusCells = Mathf.Clamp(heater.maxDistance * sub * 0.65f, 2f, 24f);
                float deltaT = heater.desiredTemp - world.ambientTemperature;
                float peakRate = Mathf.Clamp(deltaT * 0.14f, -4.5f, 5.5f);
                if (Mathf.Abs(peakRate) < 0.05f) continue;

                int xMin = Mathf.Max(0, Mathf.FloorToInt(hcx - radiusCells));
                int xMax = Mathf.Min(SimWidth - 1, Mathf.CeilToInt(hcx + radiusCells));
                int yMin = Mathf.Max(0, Mathf.FloorToInt(hcy - radiusCells));
                int yMax = Mathf.Min(SimHeight - 1, Mathf.CeilToInt(hcy + radiusCells));
                float invR2 = 1f / (radiusCells * radiusCells);

                for (int y = yMin; y <= yMax; y++)
                {
                    float dy = y - hcy;
                    int row = y * SimWidth;
                    for (int x = xMin; x <= xMax; x++)
                    {
                        float dx = x - hcx;
                        float d2 = (dx * dx + dy * dy) * invR2;
                        if (d2 < 1f)
                            heat[row + x] += peakRate * (1f - d2);
                    }
                }
            }

            WeatherSimNative.WeatherSim_UploadHeatSources(_simHandle, heat);
        }

        private void StampTileHeat(float[] heat, int localTx, int localTy, int sub, float rate)
        {
            if (localTx < 0 || localTx >= CoveredTilesW || localTy < 0 || localTy >= CoveredTilesH)
                return;
            int baseSy = localTy * sub;
            int baseSx = localTx * sub;
            for (int dy = 0; dy < sub; dy++)
            {
                int row = (baseSy + dy) * SimWidth + baseSx;
                for (int dx = 0; dx < sub; dx++)
                    heat[row + dx] += rate;
            }
        }

        private void CheckWindowedRecenter(WorldGeneration world)
        {
            if (PlayerCamera.main == null || PlayerCamera.main.body == null) return;
            Vector2Int pTile = world.WorldToBlockPos(PlayerCamera.main.body.transform.position);
            int relX = pTile.x - OriginTileX;
            int relY = pTile.y - OriginTileY;
            int marginX = CoveredTilesW / 4;
            int marginY = CoveredTilesH / 4;

            bool outOfMargin =
                (relX < marginX && OriginTileX > 0) ||
                (relX > CoveredTilesW - marginX && OriginTileX + CoveredTilesW < world.width) ||
                (relY < marginY && OriginTileY > 0) ||
                (relY > CoveredTilesH - marginY && OriginTileY + CoveredTilesH < world.height);

            if (outOfMargin)
                BuildSimulationForWorld(world, recenterOnly: true);
        }

        private void ApplyVanillaWeatherSuppression(WorldGeneration world)
        {
            if (!WeatherConfig.OverrideVanillaLayerWeather.Value)
            {
                RestoreVanillaRainMap();
                return;
            }

            if (world.fogAmount > 0f) world.fogAmount = 0f;
            if (world.fogSprite != null && world.fogSprite.color.a > 0f)
                world.fogSprite.color = new Color(0.8f, 0.8f, 0.8f, 0f);
            if (world.forceRain) world.forceRain = false;

            if (_rainMapPatchedWorld != world && _black1x1Tex != null)
            {
                RestoreVanillaRainMap();
                _savedVanillaRainMap = world.fungalRainMap;
                _rainMapPatchedWorld = world;
                world.fungalRainMap = _black1x1Tex;
            }
            else if (world.fungalRainMap != _black1x1Tex && _black1x1Tex != null)
            {
                _savedVanillaRainMap = world.fungalRainMap;
                world.fungalRainMap = _black1x1Tex;
            }

            if (world.rainSprite != null && world.rainSprite.color.a > 0f)
                world.rainSprite.color = new Color(1f, 1f, 1f, 0f);
        }

        private void RestoreVanillaRainMap()
        {
            if (_rainMapPatchedWorld != null && _savedVanillaRainMap != null)
                _rainMapPatchedWorld.fungalRainMap = _savedVanillaRainMap;
            _rainMapPatchedWorld = null;
            _savedVanillaRainMap = null;
        }

        private void UpdatePlayerLocalWeather(WorldGeneration world, WeatherEntitySample s)
        {
            if (PlayerCamera.main == null || PlayerCamera.main.body == null) return;
            var body = PlayerCamera.main.body;

            PlayerSample = s;

            float windSpeed = Mathf.Sqrt(s.u * s.u + s.v * s.v);
            float windChill = (s.tempC < 30f) ? Mathf.Min(windSpeed * 0.22f, 6.0f) : 0f;
            PlayerEffectiveTempC = s.tempC - windChill;

            float rainNorm = Mathf.Clamp01(s.rain * 850f);
            float snowNorm = Mathf.Clamp01(s.snow * 550f);

            if (WeatherConfig.EnableBodyTemperatureCoupling.Value)
            {
                float precipWet = rainNorm * 8.5f + snowNorm * 2.2f;
                if (precipWet > 0.05f && body.wetness < 100f)
                {
                    body.wetness = Mathf.MoveTowards(body.wetness, 100f, precipWet * Time.deltaTime);
                    body.dirtyness = Mathf.MoveTowards(body.dirtyness, 0f, rainNorm * Time.deltaTime);
                }

                if (rainNorm > 0.15f && UnityEngine.Random.value < Time.deltaTime * rainNorm * 1.4f && PlayerCamera.main.dropletAmount <= 0.05f)
                {
                    PlayerCamera.main.SetDroplets(Color.white);
                }
            }

            if (WeatherConfig.OverrideVanillaLayerWeather.Value && world.rainLoopSource != null)
            {
                world.rainLoopSource.volume = Mathf.Lerp(world.rainLoopSource.volume, rainNorm * 0.7f, Time.deltaTime * 3f);
            }
        }

        // Completely non-blocking! Submits nearby Rigidbody2D coords to worker and applies latest completed samples.
        private void ApplyWindForcesAsync()
        {
            if (PlayerCamera.main == null || PlayerCamera.main.body == null) return;

            float scale = WeatherConfig.WindForceScale.Value;
            if (scale <= 0.0001f) return;

            float radius = WeatherConfig.WindForceRadiusTiles.Value;
            float maxAccel = WeatherConfig.MaxWindAccel.Value;
            Vector2 center = PlayerCamera.main.body.transform.position;

            int hitCount = Physics2D.OverlapCircleNonAlloc(center, radius, _overlapColliders);
            _seenRbIds.Clear();

            int bodyCount = 0;
            for (int i = 0; i < hitCount && bodyCount < MaxWindBodies; i++)
            {
                Collider2D col = _overlapColliders[i];
                if (col == null || col.isTrigger) continue;
                Rigidbody2D rb = col.attachedRigidbody;
                if (rb == null || rb.bodyType != RigidbodyType2D.Dynamic || !rb.simulated) continue;

                int rbId = rb.GetInstanceID();
                if (!_seenRbIds.Add(rbId)) continue;

                if (!WorldToSimCell(rb.worldCenterOfMass, out float cx, out float cy)) continue;

                Bounds b = col.bounds;
                float area = Mathf.Clamp((b.size.x + b.size.y) * 0.5f, 0.25f, 3.0f);

                _activeWindBodies[bodyCount] = rb;
                _activeWindAreas[bodyCount] = area;
                bodyCount++;
            }

            // 1. Submit next coordinates for worker thread
            lock (_inLock)
            {
                int slot = 0;
                if (WorldToSimCell(center, out float pcx, out float pcy))
                {
                    _inSampleCx[0] = pcx;
                    _inSampleCy[0] = pcy;
                    _inSampleRbIds[0] = -1;
                    slot = 1;
                }
                for (int i = 0; i < bodyCount && slot <= MaxWindBodies; i++)
                {
                    if (WorldToSimCell(_activeWindBodies[i].worldCenterOfMass, out float cx, out float cy))
                    {
                        _inSampleCx[slot] = cx;
                        _inSampleCy[slot] = cy;
                        _inSampleRbIds[slot] = _activeWindBodies[i].GetInstanceID();
                        slot++;
                    }
                }
                _inSampleCount = slot;
            }

            // 2. Read latest worker wind samples into lookup dictionary
            _windLookup.Clear();
            lock (_outLock)
            {
                for (int i = 1; i < _outWindCount; i++)
                {
                    _windLookup[_outWindRbIds[i]] = _outWindSamples[i];
                }
            }

            // 3. Apply aerodynamic wind drag (F = scale * area * |u_rel| * u_rel, clamped by maxAccel * mass)
            for (int i = 0; i < bodyCount; i++)
            {
                Rigidbody2D rb = _activeWindBodies[i];
                if (rb == null) continue;
                if (!_windLookup.TryGetValue(rb.GetInstanceID(), out WeatherEntitySample s))
                    continue;

                Vector2 relVel = new Vector2(s.u, s.v) - rb.velocity;
                float relMag = relVel.magnitude;
                if (relMag < 0.05f) continue;

                Vector2 force = relVel * (relMag * scale * _activeWindAreas[i]);
                float maxForce = maxAccel * Mathf.Max(rb.mass, 0.05f);
                if (force.sqrMagnitude > maxForce * maxForce)
                    force = force.normalized * maxForce;

                rb.AddForce(force, ForceMode2D.Force);
            }
        }

        private void DestroySimulation()
        {
            _workerStop = true;
            if (_workerThread != null)
            {
                if (_workerThread.IsAlive)
                    _workerThread.Join(600);
                _workerThread = null;
            }

            if (_simHandle != IntPtr.Zero)
            {
                WeatherSimNative.WeatherSim_Destroy(_simHandle);
                _simHandle = IntPtr.Zero;
            }

            lock (_inLock)
            {
                _inPatchX.Clear();
                _inPatchY.Clear();
                _inPatchType.Clear();
                _inSampleCount = 0;
                _inViewReq.Valid = false;
            }
            lock (_outLock)
            {
                _outFluidEvents.Clear();
                _outRgbaSeq = 0;
                _outPrecipSeq = 0;
                _outWindCount = 0;
                _outHasPlayerSample = false;
            }
        }
    }
}
