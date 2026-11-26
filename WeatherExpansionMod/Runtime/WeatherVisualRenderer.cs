using System;
using UnityEngine;
using WeatherExpansionMod.Native;

namespace WeatherExpansionMod.Runtime
{
    internal enum WeatherOverlayMode
    {
        Natural = 0,
        Temperature = 1,
        Humidity = 2,
        WindSpeed = 3,
        LatentHeat = 4,
        Off = 5
    }

    /// <summary>
    /// Completely non-blocking visual renderer: submits viewport request to the async CUDA worker
    /// and uploads the latest completed RGBA8 texture + spawns spatially uniform wind-driven rain & snow particles.
    /// </summary>
    internal sealed class WeatherVisualRenderer : MonoBehaviour
    {
        public static WeatherVisualRenderer? Instance { get; private set; }

        public WeatherOverlayMode CurrentMode { get; set; } = WeatherOverlayMode.Natural;

        // Viewport overlay
        private GameObject? _overlayObj;
        private SpriteRenderer? _overlayRenderer;
        private Texture2D? _overlayTex;
        private Sprite? _overlaySprite;
        private byte[] _rgbaBuffer = Array.Empty<byte>();
        private int _texW;
        private int _texH;
        private int _lastAppliedRgbaSeq;
        private bool _loggedVisualReady;

        // Precipitation particle systems (Snowflakes & Raindrops)
        private ParticleSystem? _snowPs;
        private ParticleSystem? _rainPs;
        private readonly WeatherPrecipCell[] _precipCells =
            new WeatherPrecipCell[WeatherSimulationHost.PrecipGridW * WeatherSimulationHost.PrecipGridH];

        private float _precipSpawnTimer;

        private void Awake()
        {
            Instance = this;
            CreateOverlayObjects();
            CreateParticleSystems();
        }

        private void OnDestroy()
        {
            if (_overlayObj != null) Destroy(_overlayObj);
            if (_overlayTex != null) Destroy(_overlayTex);
            if (_overlaySprite != null) Destroy(_overlaySprite);
            if (_snowPs != null) Destroy(_snowPs.gameObject);
            if (_rainPs != null) Destroy(_rainPs.gameObject);
            if (Instance == this) Instance = null;
        }

        public void CycleMode()
        {
            int next = ((int)CurrentMode + 1) % 6;
            CurrentMode = (WeatherOverlayMode)next;
            _lastAppliedRgbaSeq = -1;
            WeatherPlugin.Log?.LogInfo($"[WeatherExpansion] Overlay mode switched to: {GetModeLabel()}");
        }

        public string GetModeLabel() => CurrentMode switch
        {
            WeatherOverlayMode.Natural => "自然天气 (云雾/雨/雪)",
            WeatherOverlayMode.Temperature => "调试: 温度热力图 (°C)",
            WeatherOverlayMode.Humidity => "调试: 相对湿度 (%)",
            WeatherOverlayMode.WindSpeed => "调试: 气流风速 (tiles/s)",
            WeatherOverlayMode.LatentHeat => "调试: 相变潜热 (°C/s)",
            WeatherOverlayMode.Off => "关闭叠加层 (仅粒子)",
            _ => "未知"
        };

        private void LateUpdate()
        {
            try
            {
                LateUpdateInternal();
            }
            catch (Exception ex)
            {
                WeatherPlugin.Log?.LogError($"[WeatherExpansion] Exception in VisualRenderer.LateUpdate: {ex}");
            }
        }

        private void LateUpdateInternal()
        {
            var host = WeatherSimulationHost.Instance;
            if (host == null || !host.IsReady || !WeatherConfig.Enabled.Value)
            {
                if (_overlayRenderer != null) _overlayRenderer.enabled = false;
                return;
            }

            Camera? cam = PlayerCamera.main != null ? PlayerCamera.main.GetComponent<Camera>() : null;
            if (cam == null) cam = Camera.main;
            if (cam == null) return;

            float halfH = cam.orthographic ? cam.orthographicSize + 4f : 24f;
            float halfW = halfH * cam.aspect + 4f;
            Vector2 camPos = cam.transform.position;

            Vector2 minWorld = new Vector2(camPos.x - halfW, camPos.y - halfH);
            Vector2 maxWorld = new Vector2(camPos.x + halfW, camPos.y + halfH);

            host.WorldToSimCell(minWorld, out float minCx, out float minCy);
            host.WorldToSimCell(maxWorld, out float maxCx, out float maxCy);

            int x0 = Mathf.Clamp(Mathf.FloorToInt(minCx), 0, host.SimWidth - 1);
            int y0 = Mathf.Clamp(Mathf.FloorToInt(minCy), 0, host.SimHeight - 1);
            int x1 = Mathf.Clamp(Mathf.CeilToInt(maxCx), x0 + 1, host.SimWidth);
            int y1 = Mathf.Clamp(Mathf.CeilToInt(maxCy), y0 + 1, host.SimHeight);

            int winW = Mathf.Clamp(x1 - x0, 8, 512);
            int winH = Mathf.Clamp(y1 - y0, 8, 512);

            float playerCx = -1e6f, playerCy = -1e6f, innerCells = 0f, outerCells = 0f;
            if (PlayerCamera.main != null && PlayerCamera.main.body != null &&
                host.WorldToSimCell(PlayerCamera.main.body.transform.position, out float pcx, out float pcy))
            {
                playerCx = pcx;
                playerCy = pcy;
                innerCells = WeatherConfig.PlayerFogClearInnerTiles.Value * host.Subdivision;
                outerCells = Mathf.Max(WeatherConfig.PlayerFogClearOuterTiles.Value * host.Subdivision, innerCells + 1f);
            }

            // 1. Submit viewport request to async CUDA worker (non-blocking, <0.01ms)
            host.SubmitViewportRequest(new ViewportRequest
            {
                Valid = true,
                X0 = x0,
                Y0 = y0,
                WinW = winW,
                WinH = winH,
                Mode = (int)CurrentMode,
                PlayerCx = playerCx,
                PlayerCy = playerCy,
                ClearInnerCells = innerCells,
                ClearOuterCells = outerCells,
                MinFogFactor = WeatherConfig.PlayerFogMinFactor.Value,
                WallFogMultiplier = WeatherConfig.WallFogMultiplier.Value,
                WantParticles = WeatherConfig.EnableParticles.Value
            });

            // 2. Consume latest completed RGBA8 frame from worker (zero GPU stall!)
            if (CurrentMode == WeatherOverlayMode.Off)
            {
                if (_overlayRenderer != null) _overlayRenderer.enabled = false;
            }
            else if (host.TryCopyLatestOverlay(
                ref _lastAppliedRgbaSeq, ref _rgbaBuffer,
                out int rx0, out int ry0, out int rWinW, out int rWinH, out int rMode))
            {
                ApplyOverlayTexture(host, rx0, ry0, rWinW, rWinH, (WeatherOverlayMode)rMode);
            }

            // 3. Emit balanced full-screen rain & snow particles from latest worker precip grid (~20 Hz)
            if (WeatherConfig.EnableParticles.Value && Time.timeScale > 0f)
            {
                _precipSpawnTimer += Time.deltaTime;
                if (_precipSpawnTimer >= 0.05f)
                {
                    float dtSpawn = Mathf.Min(_precipSpawnTimer, 0.15f);
                    _precipSpawnTimer = 0f;
                    SpawnPrecipitationParticles(host, dtSpawn);
                }
            }
        }

        private static Material CreateUnlitSpriteMaterial(Texture2D? mainTex)
        {
            Shader? shader =
                Shader.Find("Universal Render Pipeline/2D/Sprite-Unlit-Default") ??
                Shader.Find("Sprites/Default") ??
                Shader.Find("UI/Default");

            Material mat;
            if (shader != null)
            {
                mat = new Material(shader);
            }
            else if (WorldGeneration.world != null && WorldGeneration.world.rainSprite != null && WorldGeneration.world.rainSprite.sharedMaterial != null)
            {
                mat = new Material(WorldGeneration.world.rainSprite.sharedMaterial);
            }
            else if (WorldGeneration.world != null && WorldGeneration.world.defaultMat != null)
            {
                mat = new Material(WorldGeneration.world.defaultMat);
            }
            else
            {
                mat = new Material(Shader.Find("Hidden/Internal-Colored"));
            }

            if (mainTex != null)
                mat.mainTexture = mainTex;
            return mat;
        }

        private static int GetWorldSortingLayerId()
        {
            var world = WorldGeneration.world;
            if (world != null)
            {
                if (world.rainSprite != null) return world.rainSprite.sortingLayerID;
                if (world.fogSprite != null) return world.fogSprite.sortingLayerID;
            }
            return 0;
        }

        private void CreateOverlayObjects()
        {
            _overlayObj = new GameObject("WeatherExpansion_Overlay");
            _overlayObj.transform.SetParent(transform, false);
            _overlayRenderer = _overlayObj.AddComponent<SpriteRenderer>();
            _overlayRenderer.sortingLayerID = GetWorldSortingLayerId();
            _overlayRenderer.sortingOrder = 1020;
            _overlayRenderer.material = CreateUnlitSpriteMaterial(null);
            _overlayRenderer.enabled = false;
        }

        private void EnsureOverlayTexture(int w, int h)
        {
            if (_overlayTex != null && _texW == w && _texH == h) return;

            if (_overlayTex != null) Destroy(_overlayTex);
            if (_overlaySprite != null) Destroy(_overlaySprite);

            _texW = w;
            _texH = h;
            _overlayTex = new Texture2D(w, h, TextureFormat.RGBA32, false)
            {
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp
            };
            _overlaySprite = Sprite.Create(
                _overlayTex,
                new Rect(0, 0, w, h),
                new Vector2(0f, 0f),
                1f,
                0,
                SpriteMeshType.FullRect);

            if (_overlayRenderer != null)
                _overlayRenderer.sprite = _overlaySprite;
        }

        private void ApplyOverlayTexture(WeatherSimulationHost host, int x0, int y0, int winW, int winH, WeatherOverlayMode mode)
        {
            if (_overlayRenderer == null || _overlayObj == null) return;
            EnsureOverlayTexture(winW, winH);

            _overlayTex!.LoadRawTextureData(_rgbaBuffer);
            _overlayTex.Apply(false, false);

            Vector2 bottomLeftWorld = host.SimCellToWorld(x0, y0);
            float scale = 1f / host.Subdivision;

            _overlayObj.transform.position = new Vector3(bottomLeftWorld.x, bottomLeftWorld.y, 0f);
            _overlayObj.transform.localScale = new Vector3(scale, scale, 1f);

            float opacity = (mode == WeatherOverlayMode.Natural)
                ? WeatherConfig.OverlayOpacity.Value
                : 0.85f;
            _overlayRenderer.color = new Color(1f, 1f, 1f, opacity);
            _overlayRenderer.enabled = true;

            if (!_loggedVisualReady)
            {
                _loggedVisualReady = true;
                WeatherPlugin.Log?.LogInfo(
                    $"[WeatherExpansion] Async overlay active: shader={_overlayRenderer.sharedMaterial?.shader?.name}, " +
                    $"sortingLayer={_overlayRenderer.sortingLayerName}({_overlayRenderer.sortingLayerID}), order={_overlayRenderer.sortingOrder}, " +
                    $"win={winW}x{winH} at world({bottomLeftWorld.x:F1},{bottomLeftWorld.y:F1}).");
            }
        }

        private void CreateParticleSystems()
        {
            Material snowMat = CreateUnlitSpriteMaterial(CreateSoftCircleTexture(16));
            Material rainMat = CreateUnlitSpriteMaterial(CreateRainStreakTexture(4, 16));
            int sortLayer = GetWorldSortingLayerId();

            // 1) Snowflakes
            var snowGo = new GameObject("WeatherExpansion_SnowParticles");
            snowGo.transform.SetParent(transform, false);
            _snowPs = snowGo.AddComponent<ParticleSystem>();
            var sMain = _snowPs.main;
            sMain.loop = true;
            sMain.playOnAwake = true;
            sMain.simulationSpace = ParticleSystemSimulationSpace.World;
            sMain.maxParticles = 4000;
            sMain.startLifetime = new ParticleSystem.MinMaxCurve(2.2f, 4.5f);
            sMain.startSize = new ParticleSystem.MinMaxCurve(0.12f, 0.28f);
            sMain.startColor = new Color(0.92f, 0.97f, 1.0f, 0.90f);
            sMain.gravityModifier = 0.04f;

            var sEmit = _snowPs.emission;
            sEmit.enabled = false;

            var sCol = _snowPs.collision;
            sCol.enabled = true;
            sCol.type = ParticleSystemCollisionType.World;
            sCol.mode = ParticleSystemCollisionMode.Collision2D;
            sCol.collidesWith = 1 << 6;
            sCol.dampen = 0.95f;
            sCol.bounce = 0.02f;
            sCol.lifetimeLoss = 0.35f;

            var sNoise = _snowPs.noise;
            sNoise.enabled = true;
            sNoise.strength = 0.45f;
            sNoise.frequency = 0.6f;

            var sRend = snowGo.GetComponent<ParticleSystemRenderer>();
            sRend.material = snowMat;
            sRend.sortingLayerID = sortLayer;
            sRend.sortingOrder = 1030;

            // 2) Raindrops
            var rainGo = new GameObject("WeatherExpansion_RainParticles");
            rainGo.transform.SetParent(transform, false);
            _rainPs = rainGo.AddComponent<ParticleSystem>();
            var rMain = _rainPs.main;
            rMain.loop = true;
            rMain.playOnAwake = true;
            rMain.simulationSpace = ParticleSystemSimulationSpace.World;
            rMain.maxParticles = 4000;
            rMain.startLifetime = new ParticleSystem.MinMaxCurve(0.6f, 1.4f);
            rMain.startSize = new ParticleSystem.MinMaxCurve(0.18f, 0.36f);
            rMain.startColor = new Color(0.65f, 0.85f, 1.0f, 0.72f);
            rMain.gravityModifier = 0.25f;

            var rEmit = _rainPs.emission;
            rEmit.enabled = false;

            var rCol = _rainPs.collision;
            rCol.enabled = true;
            rCol.type = ParticleSystemCollisionType.World;
            rCol.mode = ParticleSystemCollisionMode.Collision2D;
            rCol.collidesWith = 1 << 6;
            rCol.lifetimeLoss = 1.0f;

            var rRend = rainGo.GetComponent<ParticleSystemRenderer>();
            rRend.material = rainMat;
            rRend.renderMode = ParticleSystemRenderMode.Stretch;
            rRend.velocityScale = 0.06f;
            rRend.lengthScale = 1.5f;
            rRend.sortingLayerID = sortLayer;
            rRend.sortingOrder = 1030;
        }

        private void SpawnPrecipitationParticles(WeatherSimulationHost host, float dt)
        {
            if (_snowPs == null || _rainPs == null) return;
            if (!host.TryCopyLatestPrecipGrid(_precipCells, out int x0, out int y0, out int winW, out int winH))
                return;

            const int gridW = WeatherSimulationHost.PrecipGridW;
            const int gridH = WeatherSimulationHost.PrecipGridH;
            int totalCells = gridW * gridH;
            float stepX = (float)winW / gridW;
            float stepY = (float)winH / gridH;

            float expectedSnow = 0f;
            float expectedRain = 0f;
            for (int idx = 0; idx < totalCells; idx++)
            {
                WeatherPrecipCell cell = _precipCells[idx];
                if (cell.isAir == 0) continue;
                if (cell.qi > 0.00004f) expectedSnow += Mathf.Clamp01(cell.qi * 1400f * dt * 2.5f);
                if (cell.qr > 0.00004f) expectedRain += Mathf.Clamp01(cell.qr * 1800f * dt * 3.0f);
            }

            int snowRoom = Mathf.Max(0, _snowPs.main.maxParticles - _snowPs.particleCount);
            int rainRoom = Mathf.Max(0, _rainPs.main.maxParticles - _rainPs.particleCount);
            float snowBudget = Mathf.Min(110f, snowRoom);
            float rainBudget = Mathf.Min(140f, rainRoom);

            float snowScale = (expectedSnow > snowBudget && expectedSnow > 1e-4f) ? (snowBudget / expectedSnow) : 1f;
            float rainScale = (expectedRain > rainBudget && expectedRain > 1e-4f) ? (rainBudget / expectedRain) : 1f;

            const int coprimeStride = 347; // Coprime to 1232 (44 * 28)
            int startIdx = UnityEngine.Random.Range(0, totalCells);
            int snowEmitted = 0;
            int rainEmitted = 0;
            int maxSnowThisTick = (int)snowBudget;
            int maxRainThisTick = (int)rainBudget;

            for (int k = 0; k < totalCells; k++)
            {
                int idx = (startIdx + k * coprimeStride) % totalCells;
                WeatherPrecipCell cell = _precipCells[idx];
                if (cell.isAir == 0) continue;

                int oy = idx / gridW;
                int ox = idx - oy * gridW;

                if (snowEmitted < maxSnowThisTick && cell.qi > 0.00004f)
                {
                    float prob = Mathf.Clamp01(cell.qi * 1400f * dt * 2.5f) * snowScale;
                    if (UnityEngine.Random.value < prob)
                    {
                        float cx = x0 + (ox + UnityEngine.Random.value) * stepX;
                        float cy = y0 + (oy + UnityEngine.Random.value) * stepY;
                        Vector2 wp = host.SimCellToWorld(cx, cy);

                        var ep = new ParticleSystem.EmitParams
                        {
                            position = new Vector3(wp.x, wp.y, 0f),
                            velocity = new Vector3(
                                cell.u + UnityEngine.Random.Range(-0.4f, 0.4f),
                                cell.v - 1.6f + UnityEngine.Random.Range(-0.3f, 0.2f),
                                0f),
                            startSize = UnityEngine.Random.Range(0.12f, 0.28f)
                        };
                        _snowPs.Emit(ep, 1);
                        snowEmitted++;
                    }
                }

                if (rainEmitted < maxRainThisTick && cell.qr > 0.00004f)
                {
                    float prob = Mathf.Clamp01(cell.qr * 1800f * dt * 3.0f) * rainScale;
                    if (UnityEngine.Random.value < prob)
                    {
                        float cx = x0 + (ox + UnityEngine.Random.value) * stepX;
                        float cy = y0 + (oy + UnityEngine.Random.value) * stepY;
                        Vector2 wp = host.SimCellToWorld(cx, cy);

                        var ep = new ParticleSystem.EmitParams
                        {
                            position = new Vector3(wp.x, wp.y, 0f),
                            velocity = new Vector3(
                                cell.u * 0.85f + UnityEngine.Random.Range(-0.2f, 0.2f),
                                cell.v - 7.5f + UnityEngine.Random.Range(-1.0f, 0.5f),
                                0f),
                            startSize = UnityEngine.Random.Range(0.18f, 0.34f)
                        };
                        _rainPs.Emit(ep, 1);
                        rainEmitted++;
                    }
                }
            }
        }

        private static Texture2D CreateSoftCircleTexture(int size)
        {
            var tex = new Texture2D(size, size, TextureFormat.RGBA32, false);
            float half = (size - 1) * 0.5f;
            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    float dx = (x - half) / half;
                    float dy = (y - half) / half;
                    float d = Mathf.Sqrt(dx * dx + dy * dy);
                    float a = Mathf.Clamp01(1f - d * d);
                    tex.SetPixel(x, y, new Color(1f, 1f, 1f, a * a));
                }
            }
            tex.Apply(false, false);
            return tex;
        }

        private static Texture2D CreateRainStreakTexture(int w, int h)
        {
            var tex = new Texture2D(w, h, TextureFormat.RGBA32, false);
            for (int y = 0; y < h; y++)
            {
                float vy = (float)y / (h - 1);
                float fadeY = Mathf.Sin(vy * Mathf.PI);
                for (int x = 0; x < w; x++)
                {
                    float vx = (float)x / (w - 1);
                    float fadeX = Mathf.Sin(vx * Mathf.PI);
                    tex.SetPixel(x, y, new Color(0.8f, 0.9f, 1f, fadeX * fadeY));
                }
            }
            tex.Apply(false, false);
            return tex;
        }
    }
}
