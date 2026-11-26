using HarmonyLib;
using UnityEngine;
using WeatherExpansionMod.Runtime;

namespace WeatherExpansionMod.Patches
{
    internal static class WorldWeatherPatches
    {
        [HarmonyPatch(typeof(WorldGeneration), "Awake")]
        [HarmonyPostfix]
        private static void WorldGeneration_Awake_Postfix(WorldGeneration __instance)
        {
            if (!WeatherConfig.Enabled.Value) return;
            WeatherSimulationHost.EnsureAttachedToWorld(__instance);
        }

        [HarmonyPatch(typeof(WorldGeneration), "Update")]
        [HarmonyPostfix]
        private static void WorldGeneration_Update_Postfix(WorldGeneration __instance)
        {
            if (!WeatherConfig.Enabled.Value) return;
            if (WeatherSimulationHost.Instance == null)
            {
                WeatherSimulationHost.EnsureAttachedToWorld(__instance);
            }
        }

        [HarmonyPatch(typeof(WorldGeneration), "OnDestroy")]
        [HarmonyPrefix]
        private static void WorldGeneration_OnDestroy_Prefix()
        {
            WeatherSimulationHost.Instance?.ForceShutdown();
        }

        [HarmonyPatch(typeof(WorldGeneration), nameof(WorldGeneration.SetFog))]
        [HarmonyPrefix]
        private static bool SetFog_Prefix(WorldGeneration __instance)
        {
            if (!WeatherConfig.Enabled.Value || !WeatherConfig.OverrideVanillaLayerWeather.Value)
                return true;

            __instance.fogAmount = 0f;
            if (__instance.fogSprite != null)
            {
                __instance.fogSprite.color = new Color(0.8f, 0.8f, 0.8f, 0f);
            }
            return false;
        }

        [HarmonyPatch(typeof(WorldGeneration), nameof(WorldGeneration.SetBlock), typeof(Vector2Int), typeof(ushort))]
        [HarmonyPostfix]
        private static void SetBlock_Postfix(Vector2Int pos, ushort block)
        {
            WeatherSimulationHost.Instance?.NotifyBlockChanged(pos, block);
        }

        [HarmonyPatch(typeof(WorldGeneration), nameof(WorldGeneration.RegenerateWorld))]
        [HarmonyPrefix]
        private static void RegenerateWorld_Prefix()
        {
            WeatherSimulationHost.Instance?.NotifyWorldRegenerating();
        }

        [HarmonyPatch(typeof(WorldGeneration), nameof(WorldGeneration.ApplySpecificModifier))]
        [HarmonyPostfix]
        private static void ApplySpecificModifier_Postfix()
        {
            WeatherSimulationHost.Instance?.RequestRebuildForCurrentWorld();
        }
    }
}
