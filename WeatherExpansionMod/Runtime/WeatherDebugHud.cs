using UnityEngine;

namespace WeatherExpansionMod.Runtime
{
    /// <summary>
    /// Handles F8 (cycle overlay/debug visualization) and F9 (toggle status HUD),
    /// displaying local simulated temperature, humidity, wind, precipitation, and CUDA GPU telemetry.
    /// </summary>
    internal sealed class WeatherDebugHud : MonoBehaviour
    {
        private bool _showHud = true;
        private GUIStyle? _boxStyle;
        private GUIStyle? _labelStyle;

        private void Start()
        {
            _showHud = WeatherConfig.ShowHudByDefault.Value;
        }

        private void Update()
        {
            if (Input.GetKeyDown(WeatherConfig.CycleDebugOverlayKey.Value))
            {
                WeatherVisualRenderer.Instance?.CycleMode();
            }

            if (Input.GetKeyDown(WeatherConfig.ToggleHudKey.Value))
            {
                _showHud = !_showHud;
                WeatherPlugin.Log?.LogInfo($"[WeatherExpansion] HUD toggled: {_showHud}");
            }
        }

        private void OnGUI()
        {
            if (!_showHud || !WeatherConfig.Enabled.Value) return;
            var host = WeatherSimulationHost.Instance;
            if (host == null) return;

            EnsureStyles();

            if (!host.IsReady)
            {
                Rect waitRect = new Rect(14f, 14f, 360f, 44f);
                GUI.Box(waitRect, GUIContent.none, _boxStyle);
                GUI.Label(
                    new Rect(waitRect.x + 10f, waitRect.y + 8f, waitRect.width - 20f, waitRect.height - 16f),
                    $"<b>天气拓展 (CUDA 2D)</b>  状态: {host.StatusMessage}",
                    _labelStyle);
                return;
            }

            var s = host.PlayerSample;
            var st = host.LastStats;
            float windSpd = Mathf.Sqrt(s.u * s.u + s.v * s.v);
            string modeLabel = WeatherVisualRenderer.Instance != null
                ? WeatherVisualRenderer.Instance.GetModeLabel()
                : "N/A";

            int netWater = host.TotalRainWaterTiles - host.TotalEvaporatedTiles;
            string text =
                $"<b>天气拓展 v{WeatherPlugin.PluginVersion} (CUDA 异步)</b>  [{host.ActiveProfile.SummaryTags}]\n" +
                $"当地气温: <b>{s.tempC:+0.0;-0.0;0.0}°C</b> (体感 {host.PlayerEffectiveTempC:+0.0;-0.0;0.0}°C | 中段 {host.VanillaMidLayerTempC:+0.0;-0.0;0.0}°C)  |  热源: {host.ActiveDynamicHeaterCount}\n" +
                $"相对湿度: <b>{s.relHumidity * 100f:F0}%</b>  |  风速: <b>{windSpd:F2}</b> ({s.u:+0.0;-0.0}, {s.v:+0.0;-0.0}) tiles/s\n" +
                $"云/雨/雪: qc={s.cloud * 1000f:F2}  qr={s.rain * 1000f:F2}  qi={s.snow * 1000f:F2} g/kg  |  水平衡: 蒸-{host.TotalEvaporatedTiles} 降+{host.TotalRainWaterTiles} (净{netWater:+0;-0;0})\n" +
                $"网格: {host.SimWidth}x{host.SimHeight} ({host.Subdivision}x{host.Subdivision}/格)  |  GPU: {st.lastStepGpuMs:F2}ms  |  显存: {st.allocatedVramBytes / (1024f * 1024f):F1}MB\n" +
                $"[{WeatherConfig.ToggleFullMapKey.Value}] 全图雷达  |  [{WeatherConfig.CycleDebugOverlayKey.Value}] {modeLabel}  |  [{WeatherConfig.ToggleHudKey.Value}] 隐藏";

            Rect rect = new Rect(14f, 14f, 540f, 122f);
            GUI.Box(rect, GUIContent.none, _boxStyle);
            GUI.Label(new Rect(rect.x + 10f, rect.y + 7f, rect.width - 20f, rect.height - 14f), text, _labelStyle);
        }

        private void EnsureStyles()
        {
            if (_boxStyle != null && _labelStyle != null) return;

            var bg = new Texture2D(1, 1);
            bg.SetPixel(0, 0, new Color(0.05f, 0.08f, 0.12f, 0.78f));
            bg.Apply();

            _boxStyle = new GUIStyle(GUI.skin.box)
            {
                normal = { background = bg }
            };
            _labelStyle = new GUIStyle(GUI.skin.label)
            {
                richText = true,
                fontSize = 12,
                wordWrap = false,
                normal = { textColor = new Color(0.93f, 0.96f, 1f, 1f) }
            };
        }
    }
}
