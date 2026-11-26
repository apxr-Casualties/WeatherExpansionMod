using System.IO;
using BepInEx;
using BepInEx.Logging;
using HarmonyLib;
using WeatherExpansionMod.Native;
using WeatherExpansionMod.Patches;
using WeatherExpansionMod.Runtime;

namespace WeatherExpansionMod
{
    [BepInPlugin(PluginGuid, PluginName, PluginVersion)]
    public sealed class WeatherPlugin : BaseUnityPlugin
    {
        public const string PluginGuid = "com.casualties.weather.expansion";
        public const string PluginName = "Weather Expansion Mod (天气拓展)";
        public const string PluginVersion = "1.0.0";

        internal static ManualLogSource? Log;
        private static Harmony? _harmony;

        private void Awake()
        {
            Log = Logger;
            WeatherConfig.Bind(Config);

            string pluginDir = Path.GetDirectoryName(Info.Location) ?? Paths.PluginPath;
            if (!WeatherSimNative.TryPreloadNativeDll(pluginDir, out string err))
            {
                Logger.LogError($"[WeatherExpansion] Native CUDA DLL load failed: {err}");
                Logger.LogError("[WeatherExpansion] Mod will stay dormant until WeatherSimCuda.dll is placed alongside WeatherExpansionMod.dll on an NVIDIA GPU system.");
                return;
            }

            // Patch once and keep patches alive across Unity scene loads (in BepInEx 5 with
            // HideManagerGameObject=false, BaseUnityPlugin.OnDestroy can fire on first scene load!).
            if (_harmony == null)
            {
                _harmony = new Harmony(PluginGuid);
                _harmony.PatchAll(typeof(WorldWeatherPatches));
                _harmony.PatchAll(typeof(LayerModifierPatches));
                _harmony.PatchAll(typeof(BodyTemperaturePatches));
            }

            Logger.LogInfo($"{PluginName} v{PluginVersion} loaded with CUDA backend (native DLL ready, awaiting WorldGeneration).");
        }

        private void OnApplicationQuit()
        {
            WeatherSimulationHost.Instance?.ForceShutdown();
        }
    }
}
