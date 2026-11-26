using HarmonyLib;

namespace WeatherExpansionMod.Patches
{
    /// <summary>
    /// Suppresses vanilla flat visual effects for FoggyModifier and RainyModifier while keeping
    /// LayerModifier.active = true so layer names/alerts work normally and WeatherLayerProfile
    /// translates them into real CUDA humidity & thermal gradients.
    /// </summary>
    internal static class LayerModifierPatches
    {
        [HarmonyPatch(typeof(FoggyModifier), nameof(FoggyModifier.Initialize))]
        [HarmonyPrefix]
        private static bool FoggyModifier_Initialize_Prefix()
        {
            return !(WeatherConfig.Enabled.Value && WeatherConfig.OverrideVanillaLayerWeather.Value);
        }

        [HarmonyPatch(typeof(RainyModifier), nameof(RainyModifier.Initialize))]
        [HarmonyPrefix]
        private static bool RainyModifier_Initialize_Prefix(WorldGeneration world)
        {
            if (!WeatherConfig.Enabled.Value || !WeatherConfig.OverrideVanillaLayerWeather.Value)
                return true;

            if (world != null)
            {
                world.forceRain = false;
            }
            return false;
        }
    }
}
