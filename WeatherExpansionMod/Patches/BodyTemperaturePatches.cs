using HarmonyLib;
using WeatherExpansionMod.Runtime;

namespace WeatherExpansionMod.Patches
{
    /// <summary>
    /// Temporarily replaces WorldGeneration.world.ambientTemperature with the player's local
    /// CUDA-simulated effective temperature (including windchill) during Body.HandleBodyTemperature,
    /// then immediately restores the vanilla mid-layer temperature so no other system is disturbed.
    /// </summary>
    internal static class BodyTemperaturePatches
    {
        private struct AmbientSwapState
        {
            public bool Swapped;
            public float OriginalAmbient;
        }

        [HarmonyPatch(typeof(Body), "HandleBodyTemperature")]
        [HarmonyPrefix]
        private static void HandleBodyTemperature_Prefix(out AmbientSwapState __state)
        {
            __state = default;
            if (!WeatherConfig.Enabled.Value || !WeatherConfig.EnableBodyTemperatureCoupling.Value)
                return;

            var host = WeatherSimulationHost.Instance;
            var world = WorldGeneration.world;
            if (host == null || !host.IsReady || world == null)
                return;

            __state.Swapped = true;
            __state.OriginalAmbient = world.ambientTemperature;
            world.ambientTemperature = host.PlayerEffectiveTempC;
        }

        [HarmonyPatch(typeof(Body), "HandleBodyTemperature")]
        [HarmonyPostfix]
        private static void HandleBodyTemperature_Postfix(AmbientSwapState __state)
        {
            if (__state.Swapped && WorldGeneration.world != null)
            {
                WorldGeneration.world.ambientTemperature = __state.OriginalAmbient;
            }
        }
    }
}
