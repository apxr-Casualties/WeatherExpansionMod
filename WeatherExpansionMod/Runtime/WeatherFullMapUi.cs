using UnityEngine;
using WeatherExpansionMod.Native;

namespace WeatherExpansionMod.Runtime
{
    /// <summary>
    /// Interactive F7 Full-Map Weather Radar UI.
    /// Allows viewing the entire stratum's Relative Humidity, Cloud, Rain, Snow, Temperature,
    /// Wind Speed & Direction arrows, and Composite layers with smooth pan/zoom, player marker, and hover probe.
    /// </summary>
    internal sealed class WeatherFullMapUi : MonoBehaviour
    {
        public static WeatherFullMapUi? Instance { get; private set; }

        private static readonly string[] LayerNames =
        {
            "相对湿度", "云层", "降水", "降雪", "温度", "风向风力", "综合"
        };

        private bool _isOpen;
        private int _layerMode = 4; // Default to Temperature or Wind/Composite; user can switch freely
        private bool _showWalls = true;
        private bool _showWindArrows = true;
        private int _reqVersion = 1;

        // Window geometry (draggable)
        private Rect _windowRect = new Rect(120f, 70f, 650f, 610f);
        private const int WindowId = 0x57454154;

        // Map view transform: center in normalized [0..1] map coordinates & zoom [1..8]
        private float _zoom = 1.0f;
        private Vector2 _viewCenter = new Vector2(0.5f, 0.5f);
        private bool _draggingMap;
        private Vector2 _lastMousePos;

        // Hover state
        private bool _hasHoverPos;
        private Vector2 _hoverNorm; // [0..1] x [0..1]

        // GPU readback buffers & texture
        private Texture2D? _mapTex;
        private readonly byte[] _mapRgba = new byte[WeatherSimulationHost.FullMapTexW * WeatherSimulationHost.FullMapTexH * 4];
        private readonly float[] _windU = new float[WeatherSimulationHost.FullMapWindGridW * WeatherSimulationHost.FullMapWindGridH];
        private readonly float[] _windV = new float[WeatherSimulationHost.FullMapWindGridW * WeatherSimulationHost.FullMapWindGridH];
        private int _lastMapSeq;
        private bool _hasWindGrid;
        private bool _hasHoverSample;
        private WeatherEntitySample _hoverSample;

        // Minimal procedural IMGUI drawing textures
        private Texture2D? _whiteTex;
        private Texture2D? _bgTex;
        private GUIStyle? _winStyle;
        private GUIStyle? _btnStyle;
        private GUIStyle? _btnActiveStyle;
        private GUIStyle? _infoStyle;

        private void Awake()
        {
            Instance = this;
            _mapTex = new Texture2D(
                WeatherSimulationHost.FullMapTexW,
                WeatherSimulationHost.FullMapTexH,
                TextureFormat.RGBA32,
                false)
            {
                filterMode = FilterMode.Bilinear,
                wrapMode = TextureWrapMode.Clamp
            };

            _whiteTex = new Texture2D(1, 1, TextureFormat.RGBA32, false);
            _whiteTex.SetPixel(0, 0, Color.white);
            _whiteTex.Apply();

            _bgTex = new Texture2D(1, 1, TextureFormat.RGBA32, false);
            _bgTex.SetPixel(0, 0, new Color(0.06f, 0.08f, 0.12f, 0.94f));
            _bgTex.Apply();
        }

        private void OnDestroy()
        {
            if (_mapTex != null) Destroy(_mapTex);
            if (_whiteTex != null) Destroy(_whiteTex);
            if (_bgTex != null) Destroy(_bgTex);
            if (Instance == this) Instance = null;
        }

        private void Update()
        {
            if (Input.GetKeyDown(WeatherConfig.ToggleFullMapKey.Value))
            {
                _isOpen = !_isOpen;
                _reqVersion++;
                WeatherPlugin.Log?.LogInfo($"[WeatherExpansion] Full-map radar UI toggled: {_isOpen}");
            }

            var host = WeatherSimulationHost.Instance;
            if (host == null || !host.IsReady) return;

            // Submit full-map request to async worker (zero cost when closed!)
            bool wantWind = _isOpen && (_layerMode == 5 || _showWindArrows);
            host.SubmitFullMapRequest(new FullMapRequest
            {
                Open = _isOpen,
                LayerMode = _layerMode,
                ShowWalls = _showWalls,
                WantWindGrid = wantWind,
                HasHover = _isOpen && _hasHoverPos,
                HoverCx = _hoverNorm.x * host.SimWidth,
                HoverCy = _hoverNorm.y * host.SimHeight,
                ReqVersion = _reqVersion
            });

            if (!_isOpen) return;

            if (host.TryCopyLatestFullMap(
                ref _lastMapSeq, _mapRgba, _windU, _windV,
                out bool hasWind, out bool hasHover, out WeatherEntitySample hSample))
            {
                _mapTex!.LoadRawTextureData(_mapRgba);
                _mapTex.Apply(false, false);
            }
            _hasWindGrid = hasWind;
            _hasHoverSample = hasHover;
            _hoverSample = hSample;
        }

        private void OnGUI()
        {
            if (!_isOpen || !WeatherConfig.Enabled.Value) return;
            var host = WeatherSimulationHost.Instance;
            if (host == null || !host.IsReady) return;

            EnsureStyles();

            // Clamp window inside screen bounds
            _windowRect.x = Mathf.Clamp(_windowRect.x, 4f, Mathf.Max(4f, Screen.width - _windowRect.width - 4f));
            _windowRect.y = Mathf.Clamp(_windowRect.y, 4f, Mathf.Max(4f, Screen.height - _windowRect.height - 4f));

            _windowRect = GUI.Window(WindowId, _windowRect, DrawWindow, "", _winStyle);
        }

        private void DrawWindow(int id)
        {
            var host = WeatherSimulationHost.Instance;
            if (host == null) return;

            // Title bar (drag handle)
            GUI.Label(
                new Rect(12f, 6f, _windowRect.width - 90f, 22f),
                $"<b>天气拓展 · 全图气象雷达 [{LayerNames[_layerMode]}]</b>  ({host.CoveredTilesW}×{host.CoveredTilesH}格 | 缩放 {_zoom:F1}x)",
                _infoStyle);

            if (GUI.Button(new Rect(_windowRect.width - 74f, 5f, 64f, 22f), "关闭 F7", _btnStyle))
            {
                _isOpen = false;
                _reqVersion++;
            }

            // Row 1: 7 Layer selector buttons
            float bx = 10f;
            float bw = (_windowRect.width - 20f - 6f * 4f) / 7f;
            for (int i = 0; i < LayerNames.Length; i++)
            {
                bool active = (_layerMode == i);
                if (GUI.Button(new Rect(bx + i * (bw + 4f), 32f, bw, 24f), LayerNames[i], active ? _btnActiveStyle : _btnStyle))
                {
                    _layerMode = i;
                    _reqVersion++;
                }
            }

            // Row 2: View manipulation controls
            float cy = 62f;
            if (GUI.Button(new Rect(10f, cy, 72f, 22f), "重置全图", _btnStyle))
            {
                _zoom = 1.0f;
                _viewCenter = new Vector2(0.5f, 0.5f);
            }
            if (GUI.Button(new Rect(86f, cy, 76f, 22f), "居中玩家", _btnStyle))
            {
                if (TryGetPlayerNorm(host, out Vector2 pn))
                {
                    _viewCenter = pn;
                    if (_zoom < 2.5f) _zoom = 2.5f;
                    ClampViewCenter();
                }
            }
            if (GUI.Button(new Rect(166f, cy, 34f, 22f), "-", _btnStyle))
            {
                _zoom = Mathf.Clamp(_zoom / 1.35f, 1f, 8f);
                ClampViewCenter();
            }
            if (GUI.Button(new Rect(204f, cy, 34f, 22f), "+", _btnStyle))
            {
                _zoom = Mathf.Clamp(_zoom * 1.35f, 1f, 8f);
                ClampViewCenter();
            }
            if (GUI.Button(new Rect(244f, cy, 92f, 22f), _showWalls ? "地形: 开" : "地形: 关", _showWalls ? _btnActiveStyle : _btnStyle))
            {
                _showWalls = !_showWalls;
                _reqVersion++;
            }
            bool arrowsOn = (_layerMode == 5) || _showWindArrows;
            if (GUI.Button(new Rect(340f, cy, 108f, 22f), arrowsOn ? "风向箭头: 开" : "风向箭头: 关", arrowsOn ? _btnActiveStyle : _btnStyle))
            {
                _showWindArrows = !_showWindArrows;
                _reqVersion++;
            }
            GUI.Label(new Rect(454f, cy + 2f, 190f, 20f), "滚轮缩放 · 左键拖拽平移", _infoStyle);

            // Map viewport square
            float mapSize = 468f;
            Rect mapRect = new Rect((_windowRect.width - mapSize) * 0.5f, 90f, mapSize, mapSize);
            HandleMapInput(mapRect);

            // Draw cropped UV map texture
            float span = 1.0f / _zoom;
            float u0 = Mathf.Clamp(_viewCenter.x - span * 0.5f, 0f, 1f - span);
            float v0 = Mathf.Clamp(_viewCenter.y - span * 0.5f, 0f, 1f - span);
            Rect uvRect = new Rect(u0, v0, span, span);

            if (_mapTex != null)
            {
                GUI.DrawTextureWithTexCoords(mapRect, _mapTex, uvRect, false);
            }

            // Clipped overlay group for wind arrows & player crosshair
            GUI.BeginGroup(mapRect);
            if (_hasWindGrid && (_layerMode == 5 || _showWindArrows))
            {
                DrawWindArrows(mapRect.width, mapRect.height, u0, v0, span);
            }
            if (TryGetPlayerNorm(host, out Vector2 playerNorm))
            {
                DrawPlayerMarker(mapRect.width, mapRect.height, u0, v0, span, playerNorm);
            }
            GUI.EndGroup();

            // Thin map border
            DrawBorder(mapRect, new Color(0.35f, 0.45f, 0.58f, 0.9f));

            // Legend + hover probe readout at bottom
            string legend = GetLegendText(host);
            GUI.Label(new Rect(12f, mapRect.yMax + 6f, _windowRect.width - 24f, 20f), legend, _infoStyle);

            string probeText;
            if (_hasHoverPos && _hasHoverSample)
            {
                int tileX = host.OriginTileX + Mathf.FloorToInt(_hoverNorm.x * host.CoveredTilesW);
                int tileY = host.OriginTileY + Mathf.FloorToInt(_hoverNorm.y * host.CoveredTilesH);
                float spd = Mathf.Sqrt(_hoverSample.u * _hoverSample.u + _hoverSample.v * _hoverSample.v);
                probeText =
                    $"指针 ({tileX},{tileY})  |  温度 <b>{_hoverSample.tempC:+0.0;-0.0;0.0}°C</b>  |  " +
                    $"湿度 <b>{_hoverSample.relHumidity * 100f:F0}%</b>  |  风 <b>{spd:F1}</b> ({_hoverSample.u:+0.0;-0.0},{_hoverSample.v:+0.0;-0.0})  |  " +
                    $"云 {_hoverSample.cloud * 1000f:F2} · 雨 {_hoverSample.rain * 1000f:F2} · 雪 {_hoverSample.snow * 1000f:F2} g/kg";
            }
            else
            {
                probeText = $"移动鼠标到地图上可查看任意点实时数据  |  水循环: 蒸发-{host.TotalEvaporatedTiles} / 降雨+{host.TotalRainWaterTiles}  |  活跃热源: {host.ActiveDynamicHeaterCount}";
            }
            GUI.Label(new Rect(12f, mapRect.yMax + 26f, _windowRect.width - 24f, 22f), probeText, _infoStyle);

            // Allow dragging window by top title bar only (doesn't conflict with map pan)
            GUI.DragWindow(new Rect(0f, 0f, _windowRect.width, 28f));
        }

        private void HandleMapInput(Rect mapRect)
        {
            Event e = Event.current;
            if (e == null) return;

            bool inside = mapRect.Contains(e.mousePosition);
            if (inside)
            {
                float span = 1.0f / _zoom;
                float u0 = Mathf.Clamp(_viewCenter.x - span * 0.5f, 0f, 1f - span);
                float v0 = Mathf.Clamp(_viewCenter.y - span * 0.5f, 0f, 1f - span);

                float rx = (e.mousePosition.x - mapRect.x) / mapRect.width;
                // IMGUI Y grows downward, simulation Y grows upward
                float ry = 1.0f - (e.mousePosition.y - mapRect.y) / mapRect.height;

                _hoverNorm = new Vector2(
                    Mathf.Clamp01(u0 + rx * span),
                    Mathf.Clamp01(v0 + ry * span));
                _hasHoverPos = true;

                if (e.type == EventType.ScrollWheel)
                {
                    float factor = e.delta.y < 0 ? 1.22f : (1f / 1.22f);
                    float newZoom = Mathf.Clamp(_zoom * factor, 1f, 8f);
                    // Zoom toward cursor
                    float newSpan = 1f / newZoom;
                    _viewCenter.x = _hoverNorm.x - (rx - 0.5f) * newSpan;
                    _viewCenter.y = _hoverNorm.y - (ry - 0.5f) * newSpan;
                    _zoom = newZoom;
                    ClampViewCenter();
                    e.Use();
                }
                else if (e.type == EventType.MouseDown && e.button == 0)
                {
                    _draggingMap = true;
                    _lastMousePos = e.mousePosition;
                    e.Use();
                }
            }
            else if (!_draggingMap)
            {
                _hasHoverPos = false;
            }

            if (_draggingMap)
            {
                if (e.type == EventType.MouseDrag)
                {
                    Vector2 delta = e.mousePosition - _lastMousePos;
                    _lastMousePos = e.mousePosition;
                    float span = 1f / _zoom;
                    _viewCenter.x -= (delta.x / mapRect.width) * span;
                    _viewCenter.y += (delta.y / mapRect.height) * span;
                    ClampViewCenter();
                    e.Use();
                }
                else if (e.type == EventType.MouseUp)
                {
                    _draggingMap = false;
                    e.Use();
                }
            }
        }

        private void ClampViewCenter()
        {
            float half = 0.5f / _zoom;
            _viewCenter.x = Mathf.Clamp(_viewCenter.x, half, 1f - half);
            _viewCenter.y = Mathf.Clamp(_viewCenter.y, half, 1f - half);
        }

        private void DrawWindArrows(float mapW, float mapH, float u0, float v0, float span)
        {
            if (_whiteTex == null) return;
            const int gw = WeatherSimulationHost.FullMapWindGridW;
            const int gh = WeatherSimulationHost.FullMapWindGridH;

            Color prevColor = GUI.color;
            Matrix4x4 prevMatrix = GUI.matrix;

            for (int gy = 0; gy < gh; gy++)
            {
                float ny = (gy + 0.5f) / gh;
                if (ny < v0 - 0.05f || ny > v0 + span + 0.05f) continue;

                int row = gy * gw;
                for (int gx = 0; gx < gw; gx++)
                {
                    float nx = (gx + 0.5f) / gw;
                    if (nx < u0 - 0.05f || nx > u0 + span + 0.05f) continue;

                    float wu = _windU[row + gx];
                    float wv = _windV[row + gx];
                    float spd = Mathf.Sqrt(wu * wu + wv * wv);
                    if (spd < 0.25f) continue;

                    float px = ((nx - u0) / span) * mapW;
                    float py = (1f - (ny - v0) / span) * mapH;

                    float len = Mathf.Clamp(4f + spd * 1.35f, 5f, 22f);
                    // Screen angle: +wu is right (0 deg), +wv is UP (-Y in IMGUI)
                    float angleDeg = Mathf.Atan2(-wv, wu) * Mathf.Rad2Deg;

                    float alpha = Mathf.Clamp(0.45f + spd * 0.05f, 0.45f, 0.95f);
                    GUI.color = new Color(1f, 1f, 1f, alpha);

                    // Shaft
                    GUI.matrix = prevMatrix;
                    GUIUtility.RotateAroundPivot(angleDeg, new Vector2(px, py));
                    GUI.DrawTexture(new Rect(px - len * 0.4f, py - 0.8f, len, 1.6f), _whiteTex);

                    // Arrowhead (2 short ticks)
                    float tipX = px + len * 0.6f;
                    GUI.matrix = prevMatrix;
                    GUIUtility.RotateAroundPivot(angleDeg + 145f, new Vector2(tipX, py));
                    GUI.DrawTexture(new Rect(tipX, py - 0.7f, 4.5f, 1.4f), _whiteTex);

                    GUI.matrix = prevMatrix;
                    GUIUtility.RotateAroundPivot(angleDeg - 145f, new Vector2(tipX, py));
                    GUI.DrawTexture(new Rect(tipX, py - 0.7f, 4.5f, 1.4f), _whiteTex);
                }
            }

            GUI.matrix = prevMatrix;
            GUI.color = prevColor;
        }

        private void DrawPlayerMarker(float mapW, float mapH, float u0, float v0, float span, Vector2 pNorm)
        {
            if (_whiteTex == null) return;
            float px = ((pNorm.x - u0) / span) * mapW;
            float py = (1f - (pNorm.y - v0) / span) * mapH;
            if (px < -10f || px > mapW + 10f || py < -10f || py > mapH + 10f) return;

            Color prev = GUI.color;
            GUI.color = new Color(1f, 0.25f, 0.2f, 0.95f);
            GUI.DrawTexture(new Rect(px - 7f, py - 1.2f, 14f, 2.4f), _whiteTex);
            GUI.DrawTexture(new Rect(px - 1.2f, py - 7f, 2.4f, 14f), _whiteTex);
            GUI.color = new Color(1f, 0.95f, 0.4f, 1f);
            GUI.DrawTexture(new Rect(px - 2f, py - 2f, 4f, 4f), _whiteTex);
            GUI.color = prev;
        }

        private void DrawBorder(Rect r, Color col)
        {
            if (_whiteTex == null) return;
            Color prev = GUI.color;
            GUI.color = col;
            GUI.DrawTexture(new Rect(r.x, r.y, r.width, 1f), _whiteTex);
            GUI.DrawTexture(new Rect(r.x, r.yMax - 1f, r.width, 1f), _whiteTex);
            GUI.DrawTexture(new Rect(r.x, r.y, 1f, r.height), _whiteTex);
            GUI.DrawTexture(new Rect(r.xMax - 1f, r.y, 1f, r.height), _whiteTex);
            GUI.color = prev;
        }

        private static bool TryGetPlayerNorm(WeatherSimulationHost host, out Vector2 norm)
        {
            norm = new Vector2(0.5f, 0.5f);
            if (PlayerCamera.main == null || PlayerCamera.main.body == null) return false;
            if (!host.WorldToSimCell(PlayerCamera.main.body.transform.position, out float cx, out float cy))
                return false;
            norm = new Vector2(cx / Mathf.Max(host.SimWidth, 1), cy / Mathf.Max(host.SimHeight, 1));
            return true;
        }

        private string GetLegendText(WeatherSimulationHost host)
        {
            return _layerMode switch
            {
                0 => "色标 [相对湿度]: 棕褐 0% → 青绿 50% → 亮青蓝 100%  |  红十字 = 玩家当前位置",
                1 => "色标 [云层 qc]: 深灰 无云 → 亮白 浓云团  |  红十字 = 玩家当前位置",
                2 => "色标 [降水 qr]: 深蓝 无雨 → 天蓝 阵雨 → 亮蓝紫 暴雨区  |  红十字 = 玩家当前位置",
                3 => "色标 [降雪 qi/积雪]: 深灰 无雪 → 冰青/纯白 降雪与积雪区  |  红十字 = 玩家当前位置",
                4 => $"色标 [温度]: 深蓝 {host.VanillaMidLayerTempC - 18f:F0}°C → 中段 {host.VanillaMidLayerTempC:F0}°C → 红 {host.VanillaMidLayerTempC + 18f:F0}°C (白色细线 = 0°C 冰点线)",
                5 => "色标 [风向风力]: 深紫 0 → 青绿 9 → 金黄 18+ tiles/s (白色箭头指示风向与强弱)",
                _ => "色标 [综合气象]: 灰白=云 · 蓝=雨 · 青白=雪 · 深蓝=水体 · 暗灰=岩壁  |  红十字 = 玩家位置"
            };
        }

        private void EnsureStyles()
        {
            if (_winStyle != null) return;

            _winStyle = new GUIStyle(GUI.skin.box)
            {
                normal = { background = _bgTex }
            };

            var btnBg = new Texture2D(1, 1);
            btnBg.SetPixel(0, 0, new Color(0.14f, 0.19f, 0.27f, 0.95f));
            btnBg.Apply();

            var btnActiveBg = new Texture2D(1, 1);
            btnActiveBg.SetPixel(0, 0, new Color(0.18f, 0.48f, 0.72f, 1f));
            btnActiveBg.Apply();

            _btnStyle = new GUIStyle(GUI.skin.button)
            {
                fontSize = 11,
                alignment = TextAnchor.MiddleCenter,
                normal = { background = btnBg, textColor = new Color(0.90f, 0.94f, 0.98f) },
                hover = { background = btnActiveBg, textColor = Color.white }
            };

            _btnActiveStyle = new GUIStyle(_btnStyle)
            {
                fontStyle = FontStyle.Bold,
                normal = { background = btnActiveBg, textColor = Color.white }
            };

            _infoStyle = new GUIStyle(GUI.skin.label)
            {
                richText = true,
                fontSize = 11,
                normal = { textColor = new Color(0.92f, 0.96f, 1f) }
            };
        }
    }
}
