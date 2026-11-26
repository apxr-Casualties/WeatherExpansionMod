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

    internal struct FullMapRequest
    {
        public bool Open;
        public int LayerMode; // 0=RH, 1=Cloud, 2=Rain, 3=Snow, 4=Temp, 5=Wind, 6=Composite
        public bool ShowWalls;
        public bool WantWindGrid;
        public bool HasHover;
        public float HoverCx;
        public float HoverCy;
        public int ReqVersion;
    }

    /// <summary>
    /// Owns the CUDA 2D weather simulation on a dedicated background worker thread (zero main-thread stall).
    /// Handles 1:1 mass-conserved hydrology with soft net-water cap, live campfire/torch/Heater/magma sync,
    /// aerodynamic wind forces, and async F7 full-map weather radar rendering.
    /// </summary>
    internal sealed class WeatherSimulationHost : MonoBehaviour
    {
        public static WeatherSimulationHost? Instance { get; private set; }

        public const int PrecipGridW = 44;
        public const int PrecipGridH = 28;
        public const int FullMapTexW = 384;
        public const int FullMapTexH = 384;
        public const int FullMapWindGridW = 24;
        public const int FullMapWindGridH = 24;

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
        public int ActiveDynamicHeaterCount { get; private set; }

        private WorldGeneration? _boundWorld;
        private bool _wasGenerating = true;
        private bool _pendingRebuild;
        private float _fluidSyncTimer;
        private float _heaterSyncTimer;
        private float _recenterCheckTimer;

        // Cached tile-level cell types & static terrain heat for cheap incremental diffing
        private byte[]? _tileCellTypes;
        private byte[]? _tileIsStaticHeat;

        // Saved vanilla fungalRainMap to suppress Layer 6/7 fake rain cleanly
        private Texture2D? _black1x1Tex;
        private Texture2D? _savedVanillaRainMap;
        private WorldGeneration? _rainMapPatchedWorld;

        // Dynamic heater snapshot cache (only uploads to GPU when heaters actually change!)
        private readonly List<WeatherDynamicHeatSource> _scannedHeaters = new List<WeatherDynamicHeatSource>(64);
        private readonly List<WeatherDynamicHeatSource> _lastUploadedHeaters = new List<WeatherDynamicHeatSource>(64);
        private readonly HashSet<int> _seenHeaterRoots = new HashSet<int>();

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
        private FullMapRequest _inFullMapReq;

        private const int MaxWindBodies = 256;
        private int _inSampleCount;
        private readonly float[] _inSampleCx = new float[MaxWindBodies + 1];
        private readonly float[] _inSampleCy = new float[MaxWindBodies + 1];
        private readonly int[] _inSampleRbIds = new int[MaxWindBodies + 1];

        private readonly List<int> _inPatchX = new List<int>(64);
        private readonly List<int> _inPatchY = new List<int>(64);
        private readonly List<byte> _inPatchType = new List<byte>(64);

        private readonly List<int> _inHeatPatchX = new List<int>(32);
        private readonly List<int> _inHeatPatchY = new List<int>(32);
        private readonly List<float> _inHeatPatchVal = new List<float>(32);

        private bool _inDynHeatersDirty;
        private WeatherDynamicHeatSource[] _inDynHeaters = Array.Empty<WeatherDynamicHeatSource>();

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

        // Full-map weather radar outputs
        private readonly byte[] _outFullMapRgba = new byte[FullMapTexW * FullMapTexH * 4];
        private readonly float[] _outFullMapWindU = new float[FullMapWindGridW * FullMapWindGridH];
        private readonly float[] _outFullMapWindV = new float[FullMapWindGridW * FullMapWindGridH];
        private int _outFullMapSeq;
        private bool _outFullMapHasWind;
        private WeatherEntitySample _outHoverSample;
        private bool _outHasHoverSample;

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
            hostGo.AddComponent<WeatherFullMapUi>();

            WeatherPlugin.Log?.LogInfo(
                $"[WeatherExpansion] Attached async runtime host v{WeatherPlugin.PluginVersion} to WorldGeneration (biomeDepth={world.biomeDepth}).");
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
            if (!IsReady || _boundWorld == null) return;
            int localTx = tilePos.x - OriginTileX;
            int localTy = tilePos.y - OriginTileY;
            if (localTx < 0 || localTx >= CoveredTilesW || localTy < 0 || localTy >= CoveredTilesH)
                return;

            byte[,]? fluid = FluidManager.main?.fluid;
            byte cell = DetermineTileCellType(tilePos.x, tilePos.y, blockId, fluid);
            int tIdx = localTy * CoveredTilesW + localTx;
            if (_tileCellTypes != null && tIdx >= 0 && tIdx < _tileCellTypes.Length)
                _tileCellTypes[tIdx] = cell;

            QueueTilePatch(localTx, localTy, cell);

            // Also sync static heat if a BlockInfo.hot block was placed or demolished!
            bool isHotNow = IsTileStaticHot(_boundWorld, tilePos.x, tilePos.y, blockId, fluid, out float rate);
            byte hotFlag = isHotNow ? (byte)1 : (byte)0;
            if (_tileIsStaticHeat != null && tIdx >= 0 && tIdx < _tileIsStaticHeat.Length && _tileIsStaticHeat[tIdx] != hotFlag)
            {
                _tileIsStaticHeat[tIdx] = hotFlag;
                QueueStaticHeatTilePatch(localTx, localTy, isHotNow ? rate : 0f);
            }
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

        private void QueueStaticHeatTilePatch(int localTx, int localTy, float rate)
        {
            int sub = Subdivision;
            lock (_inLock)
            {
                void AddNeighbor(int tx, int ty, float r)
                {
                    if (tx < 0 || tx >= CoveredTilesW || ty < 0 || ty >= CoveredTilesH) return;
                    for (int dy = 0; dy < sub; dy++)
                    {
                        for (int dx = 0; dx < sub; dx++)
                        {
                            _inHeatPatchX.Add(tx * sub + dx);
                            _inHeatPatchY.Add(ty * sub + dy);
                            _inHeatPatchVal.Add(r);
                        }
                    }
                }
                AddNeighbor(localTx, localTy, rate * 0.5f);
                AddNeighbor(localTx, localTy + 1, rate);
                AddNeighbor(localTx - 1, localTy, rate * 0.45f);
                AddNeighbor(localTx + 1, localTy, rate * 0.45f);
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

        public void SubmitViewportRequest(ViewportRequest req)
        {
            lock (_inLock)
            {
                _inViewReq = req;
            }
        }

        public void SubmitFullMapRequest(FullMapRequest req)
        {
            lock (_inLock)
            {
                _inFullMapReq = req;
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

        public bool TryCopyLatestFullMap(
            ref int lastSeq,
            byte[] dstRgba,
            float[] dstWindU,
            float[] dstWindV,
            out bool hasWind,
            out bool hasHover,
            out WeatherEntitySample hoverSample)
        {
            hasWind = false;
            hasHover = false;
            hoverSample = default;
            lock (_outLock)
            {
                hasHover = _outHasHoverSample;
                hoverSample = _outHoverSample;
                hasWind = _outFullMapHasWind;
                if (hasWind)
                {
                    Array.Copy(_outFullMapWindU, dstWindU, _outFullMapWindU.Length);
                    Array.Copy(_outFullMapWindV, dstWindV, _outFullMapWindV.Length);
                }
                if (_outFullMapSeq == 0 || _outFullMapSeq == lastSeq)
                    return false;

                Buffer.BlockCopy(_outFullMapRgba, 0, dstRgba, 0, _outFullMapRgba.Length);
                lastSeq = _outFullMapSeq;
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

                if (PlayerCamera.main != null && PlayerCamera.main.body != null &&
                    WorldToSimCell(PlayerCamera.main.body.transform.position, out float pcx, out float pcy))
                {
                    _inSampleCx[0] = pcx;
                    _inSampleCy[0] = pcy;
                    _inSampleRbIds[0] = -1;
                    if (_inSampleCount < 1) _inSampleCount = 1;
                }
            }

            // 1. Apply completed GPU water-cycle events (1:1 conserved + soft net-water cap)
            if (WeatherConfig.EnableWaterCycleFluidSync.Value && !paused)
            {
                ApplyCompletedFluidEvents(world);
            }

            // 2. Periodic FluidManager movement & magma cooling sync (~0.15ms)
            _fluidSyncTimer += Time.deltaTime;
            if (_fluidSyncTimer >= WeatherConfig.FluidSyncIntervalSec.Value)
            {
                _fluidSyncTimer = 0f;
                SyncNearbyFluidChanges(world);
            }

            // 3. Periodic dynamic campfire / torch / Heater crystal sync (~0.08ms, zero 16MB stall!)
            _heaterSyncTimer += Time.deltaTime;
            if (_heaterSyncTimer >= WeatherConfig.DynamicHeaterSyncIntervalSec.Value)
            {
                _heaterSyncTimer = 0f;
                SyncDynamicHeaters(forceUpload: false);
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
            _lastUploadedHeaters.Clear();

            UploadTerrainAndFluids(world);
            UploadStaticHeatSources(world);
            SyncDynamicHeaters(forceUpload: true);
            // Upload initial dynamic heaters synchronously before warmup
            lock (_inLock)
            {
                if (_inDynHeatersDirty)
                {
                    WeatherSimNative.WeatherSim_UploadDynamicHeaters(_simHandle, _inDynHeaters, _inDynHeaters.Length);
                    _inDynHeatersDirty = false;
                }
            }

            WeatherSimNative.WeatherSim_ResetFields(_simHandle);

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
                _outFullMapSeq = 0;
                _outFullMapHasWind = false;
                _outHasHoverSample = false;
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
                _inHeatPatchX.Clear();
                _inHeatPatchY.Clear();
                _inHeatPatchVal.Clear();
            }

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
                $"T_mid={_cfg.midLayerTempC:F1}C, profile=[{ActiveProfile.SummaryTags}], dynHeaters={ActiveDynamicHeaterCount}, " +
                $"VRAM={_lastStats.allocatedVramBytes / (1024f * 1024f):F1}MB, step={_lastStats.lastStepGpuMs:F2}ms).");
        }

        private void SyncDynamicHeaters(bool forceUpload)
        {
            if (!IsReady || _boundWorld == null) return;

            _scannedHeaters.Clear();
            _seenHeaterRoots.Clear();
            int sub = Subdivision;
            float refTemp = VanillaMidLayerTempC;

            // 1. All active Heater components (campfires, temp crystals, modded heaters)
            Heater[] heaters = FindObjectsOfType<Heater>();
            for (int i = 0; i < heaters.Length && _scannedHeaters.Count < 256; i++)
            {
                Heater h = heaters[i];
                if (h == null || !h.isActiveAndEnabled) continue;

                Item? item = h.GetComponentInParent<Item>();
                if (item != null)
                {
                    if (item.ParentContainer() != null) continue;
                    if (item.condition <= 0f && (item.id == "campfire" || item.id == "torch")) continue;
                }

                if (!WorldToSimCell(h.transform.position, out float cx, out float cy)) continue;

                int rootId = item != null ? item.gameObject.GetInstanceID() : h.gameObject.GetInstanceID();
                _seenHeaterRoots.Add(rootId);

                float radiusTiles = Mathf.Clamp(h.maxDistance * 0.65f, 2.5f, 14f);
                float deltaT = h.desiredTemp - refTemp;
                float gain = h.cooker ? 0.22f : 0.15f;
                float peakRate = Mathf.Clamp(deltaT * gain, -5.0f, 8.0f);
                if (h.cooker && peakRate < 4.5f) peakRate = 6.5f; // Campfires always produce a clear warm plume
                if (Mathf.Abs(peakRate) < 0.05f) continue;

                _scannedHeaters.Add(new WeatherDynamicHeatSource
                {
                    cx = cx,
                    cy = cy,
                    radiusCells = radiusTiles * sub,
                    rateCPerSec = peakRate
                });
            }

            // 2. All active CampfireAnimation components (catches campfires, lit torches, burning crystals without Heater)
            CampfireAnimation[] fires = FindObjectsOfType<CampfireAnimation>();
            for (int i = 0; i < fires.Length && _scannedHeaters.Count < 256; i++)
            {
                CampfireAnimation f = fires[i];
                if (f == null || !f.isActiveAndEnabled) continue;

                Item? item = f.GetComponentInParent<Item>();
                if (item != null)
                {
                    if (item.ParentContainer() != null) continue;
                    if (item.condition <= 0f) continue;
                }

                int rootId = item != null
                    ? item.gameObject.GetInstanceID()
                    : (f.transform.parent != null ? f.transform.parent.gameObject.GetInstanceID() : f.gameObject.GetInstanceID());
                if (!_seenHeaterRoots.Add(rootId)) continue;

                if (!WorldToSimCell(f.transform.position, out float cx, out float cy)) continue;

                bool isTorch = item != null && item.id == "torch";
                bool isCampfire = item != null && item.id == "campfire";
                float radiusTiles = isTorch ? 3.5f : (isCampfire ? 6.5f : 5.5f);
                float rate = isTorch ? 3.0f : (isCampfire ? 7.0f : 5.5f);

                _scannedHeaters.Add(new WeatherDynamicHeatSource
                {
                    cx = cx,
                    cy = cy,
                    radiusCells = radiusTiles * sub,
                    rateCPerSec = rate
                });
            }

            ActiveDynamicHeaterCount = _scannedHeaters.Count;

            bool changed = forceUpload || (_scannedHeaters.Count != _lastUploadedHeaters.Count);
            if (!changed)
            {
                for (int i = 0; i < _scannedHeaters.Count; i++)
                {
                    var a = _scannedHeaters[i];
                    var b = _lastUploadedHeaters[i];
                    if (Mathf.Abs(a.cx - b.cx) > 0.35f ||
                        Mathf.Abs(a.cy - b.cy) > 0.35f ||
                        Mathf.Abs(a.radiusCells - b.radiusCells) > 0.35f ||
                        Mathf.Abs(a.rateCPerSec - b.rateCPerSec) > 0.2f)
                    {
                        changed = true;
                        break;
                    }
                }
            }

            if (!changed) return;

            _lastUploadedHeaters.Clear();
            _lastUploadedHeaters.AddRange(_scannedHeaters);

            WeatherDynamicHeatSource[] snapshot = _scannedHeaters.ToArray();
            lock (_inLock)
            {
                _inDynHeaters = snapshot;
                _inDynHeatersDirty = true;
            }
        }

        private void WorkerLoop()
        {
            var localSampleCx = new float[MaxWindBodies + 1];
            var localSampleCy = new float[MaxWindBodies + 1];
            var localSampleIds = new int[MaxWindBodies + 1];
            var localSampleOut = new WeatherEntitySample[MaxWindBodies + 1];
            var localPrecip = new WeatherPrecipCell[PrecipGridW * PrecipGridH];
            var localEvents = new WeatherFluidEvent[512];
            byte[] localRgba = Array.Empty<byte>();

            var localFullMapRgba = new byte[FullMapTexW * FullMapTexH * 4];
            var localFullMapWindU = new float[FullMapWindGridW * FullMapWindGridH];
            var localFullMapWindV = new float[FullMapWindGridW * FullMapWindGridH];
            var hoverX = new float[1];
            var hoverY = new float[1];
            var hoverOut = new WeatherEntitySample[1];

            var sw = Stopwatch.StartNew();
            double lastFullMapMs = -1000.0;
            int lastFullMapReqVer = -1;

            while (!_workerStop)
            {
                IntPtr handle = _simHandle;
                if (handle == IntPtr.Zero) break;

                float midTemp;
                bool paused;
                float targetHz;
                ViewportRequest viewReq;
                FullMapRequest mapReq;
                int sampleCount;
                int[]? patchX = null;
                int[]? patchY = null;
                byte[]? patchT = null;
                int[]? hPatchX = null;
                int[]? hPatchY = null;
                float[]? hPatchV = null;
                bool dynDirty = false;
                WeatherDynamicHeatSource[]? dynHeaters = null;

                lock (_inLock)
                {
                    midTemp = _inMidLayerTempC;
                    paused = _inPaused;
                    targetHz = _inTargetSimHz;
                    viewReq = _inViewReq;
                    mapReq = _inFullMapReq;

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

                    if (_inHeatPatchX.Count > 0)
                    {
                        hPatchX = _inHeatPatchX.ToArray();
                        hPatchY = _inHeatPatchY.ToArray();
                        hPatchV = _inHeatPatchVal.ToArray();
                        _inHeatPatchX.Clear();
                        _inHeatPatchY.Clear();
                        _inHeatPatchVal.Clear();
                    }

                    if (_inDynHeatersDirty)
                    {
                        dynDirty = true;
                        dynHeaters = _inDynHeaters;
                        _inDynHeatersDirty = false;
                    }
                }

                double t0 = sw.Elapsed.TotalMilliseconds;
                float simDt = 1f / Math.Max(targetHz, 10f);

                try
                {
                    if (!paused)
                    {
                        WeatherSimNative.WeatherSim_SetMidLayerTemp(handle, midTemp);

                        if (patchX != null && patchY != null && patchT != null)
                            WeatherSimNative.WeatherSim_UploadCellsPatch(handle, patchX, patchY, patchT, patchX.Length);

                        if (hPatchX != null && hPatchY != null && hPatchV != null)
                            WeatherSimNative.WeatherSim_UploadStaticHeatPatch(handle, hPatchX, hPatchY, hPatchV, hPatchX.Length);

                        if (dynDirty)
                            WeatherSimNative.WeatherSim_UploadDynamicHeaters(handle, dynHeaters, dynHeaters?.Length ?? 0);

                        int rc = WeatherSimNative.WeatherSim_Step(handle, simDt);
                        if (rc != 0)
                        {
                            WeatherPlugin.Log?.LogWarning($"[WeatherExpansion] Worker WeatherSim_Step CUDA error {rc}");
                            Thread.Sleep(50);
                            continue;
                        }
                    }

                    bool sampledOk = false;
                    if (!paused && sampleCount > 0)
                    {
                        sampledOk = WeatherSimNative.WeatherSim_SamplePoints(
                            handle, localSampleCx, localSampleCy, localSampleOut, sampleCount) == 0;
                    }

                    bool rgbaOk = false;
                    bool precipOk = false;
                    if (!paused && viewReq.Valid && viewReq.WinW > 0 && viewReq.WinH > 0)
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

                    // Full-map weather radar (F7): renders ONLY when UI is open, throttled to ~8 Hz or on immediate mode change!
                    bool fullMapOk = false;
                    bool fullMapWindOk = false;
                    bool hoverOk = false;
                    if (mapReq.Open)
                    {
                        bool forceNow = (mapReq.ReqVersion != lastFullMapReqVer);
                        if (forceNow || (t0 - lastFullMapMs >= 125.0))
                        {
                            lastFullMapMs = t0;
                            lastFullMapReqVer = mapReq.ReqVersion;
                            fullMapOk = WeatherSimNative.WeatherSim_RenderFullMapRGBA8(
                                handle, FullMapTexW, FullMapTexH,
                                mapReq.LayerMode, mapReq.ShowWalls ? 1 : 0,
                                localFullMapRgba) == 0;

                            if (mapReq.WantWindGrid)
                            {
                                fullMapWindOk = WeatherSimNative.WeatherSim_SampleMapWindGrid(
                                    handle, FullMapWindGridW, FullMapWindGridH,
                                    localFullMapWindU, localFullMapWindV) == 0;
                            }
                        }

                        if (mapReq.HasHover)
                        {
                            hoverX[0] = mapReq.HoverCx;
                            hoverY[0] = mapReq.HoverCy;
                            hoverOk = WeatherSimNative.WeatherSim_SamplePoints(handle, hoverX, hoverY, hoverOut, 1) == 0;
                        }
                    }

                    int evCount = paused ? 0 : WeatherSimNative.WeatherSim_PopFluidEvents(handle, localEvents, localEvents.Length);
                    WeatherSimNative.WeatherSim_GetStats(handle, out WeatherSimStats st);

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

                        if (fullMapOk)
                        {
                            Buffer.BlockCopy(localFullMapRgba, 0, _outFullMapRgba, 0, _outFullMapRgba.Length);
                            _outFullMapSeq++;
                        }
                        if (fullMapWindOk)
                        {
                            Array.Copy(localFullMapWindU, _outFullMapWindU, _outFullMapWindU.Length);
                            Array.Copy(localFullMapWindV, _outFullMapWindV, _outFullMapWindV.Length);
                            _outFullMapHasWind = true;
                        }
                        _outHasHoverSample = hoverOk;
                        if (hoverOk) _outHoverSample = hoverOut[0];

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

                double elapsedMs = sw.Elapsed.TotalMilliseconds - t0;
                double budgetMs = paused ? 33.0 : (1000.0 / Math.Max(targetHz, 10f));
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
            int maxNetWater = Math.Max(0, WeatherConfig.MaxNetRainWaterTiles.Value);

            // Process evaporations (-1) first so freed water budget is immediately available for rainfall (+1)
            for (int pass = 0; pass < 2; pass++)
            {
                for (int i = 0; i < _mainFluidScratch.Count; i++)
                {
                    WeatherFluidEvent ev = _mainFluidScratch[i];
                    if (pass == 0 && ev.delta >= 0) continue;
                    if (pass == 1 && ev.delta <= 0) continue;

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
                        // Strict water balance guard: net rainfall water cannot exceed evaporated + soft cap
                        if ((TotalRainWaterTiles - TotalEvaporatedTiles) >= maxNetWater)
                            continue;

                        ushort b = blocks != null ? blocks[gx, gy] : world.GetBlock(new Vector2Int(gx, gy));
                        if (b == 0 && fluid[gx, gy] == 0)
                        {
                            fluid[gx, gy] = 1;
                            if (_tileCellTypes != null && tIdx >= 0 && tIdx < _tileCellTypes.Length)
                                _tileCellTypes[tIdx] = (byte)WeatherCellType.Water;
                            QueueTilePatch(ev.tileX, ev.tileY, (byte)WeatherCellType.Water);
                            TotalRainWaterTiles++;
                        }
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
                            cells[row + dx] = cellType;
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

                        if (cur != _tileCellTypes[tIdx])
                        {
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

                        if (_tileIsStaticHeat != null)
                        {
                            bool isHot = IsTileStaticHot(world, gx, gy, b, fluid, out float rate);
                            byte hotFlag = isHot ? (byte)1 : (byte)0;
                            if (_tileIsStaticHeat[tIdx] != hotFlag)
                            {
                                _tileIsStaticHeat[tIdx] = hotFlag;
                                float r = isHot ? rate : 0f;
                                for (int dy = 0; dy < sub; dy++)
                                {
                                    for (int dx = 0; dx < sub; dx++)
                                    {
                                        _inHeatPatchX.Add(localTx * sub + dx);
                                        _inHeatPatchY.Add(localTy * sub + dy);
                                        _inHeatPatchVal.Add(r);
                                    }
                                }
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

        private static bool IsTileStaticHot(WorldGeneration world, int gx, int gy, ushort blockId, byte[,]? fluid, out float rate)
        {
            rate = 0f;
            if (fluid != null && gx >= 0 && gx < fluid.GetLength(0) && gy >= 0 && gy < fluid.GetLength(1) && fluid[gx, gy] == 6)
            {
                rate = 5.5f;
                return true;
            }
            if (blockId != 0)
            {
                BlockInfo? info = world.GetBlockInfo(blockId);
                if (info != null && info.hot)
                {
                    rate = 3.0f;
                    return true;
                }
            }
            return false;
        }

        private void UploadStaticHeatSources(WorldGeneration world)
        {
            if (!IsReady) return;
            float[] heat = new float[SimWidth * SimHeight];
            _tileIsStaticHeat = new byte[CoveredTilesW * CoveredTilesH];
            ushort[,]? blocks = WorldBlocksField?.GetValue(world) as ushort[,];
            byte[,]? fluid = FluidManager.main?.fluid;
            int sub = Subdivision;

            for (int ty = 1; ty < CoveredTilesH - 1; ty++)
            {
                int gy = OriginTileY + ty;
                int tileRow = ty * CoveredTilesW;
                for (int tx = 1; tx < CoveredTilesW - 1; tx++)
                {
                    int gx = OriginTileX + tx;
                    ushort b = blocks != null ? blocks[gx, gy] : world.GetBlock(new Vector2Int(gx, gy));
                    if (!IsTileStaticHot(world, gx, gy, b, fluid, out float rate)) continue;

                    _tileIsStaticHeat[tileRow + tx] = 1;
                    StampTileHeat(heat, tx, ty, sub, rate * 0.5f);
                    StampTileHeat(heat, tx, ty + 1, sub, rate);
                    StampTileHeat(heat, tx - 1, ty, sub, rate * 0.45f);
                    StampTileHeat(heat, tx + 1, ty, sub, rate * 0.45f);
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

            _windLookup.Clear();
            lock (_outLock)
            {
                for (int i = 1; i < _outWindCount; i++)
                    _windLookup[_outWindRbIds[i]] = _outWindSamples[i];
            }

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
                _inHeatPatchX.Clear();
                _inHeatPatchY.Clear();
                _inHeatPatchVal.Clear();
                _inDynHeatersDirty = false;
                _inSampleCount = 0;
                _inViewReq.Valid = false;
                _inFullMapReq.Open = false;
            }
            lock (_outLock)
            {
                _outFluidEvents.Clear();
                _outRgbaSeq = 0;
                _outPrecipSeq = 0;
                _outFullMapSeq = 0;
                _outFullMapHasWind = false;
                _outHasHoverSample = false;
                _outWindCount = 0;
                _outHasPlayerSample = false;
            }
        }
    }
}
