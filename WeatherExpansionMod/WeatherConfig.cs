using BepInEx.Configuration;
using UnityEngine;

namespace WeatherExpansionMod
{
    internal static class WeatherConfig
    {
        public static ConfigEntry<bool> Enabled = null!;
        public static ConfigEntry<int> PreferredSubdivision = null!;
        public static ConfigEntry<int> MaxSimCells = null!;
        public static ConfigEntry<int> FallbackWindowTiles = null!;
        public static ConfigEntry<float> SimHz = null!;
        public static ConfigEntry<int> SorIterations = null!;
        public static ConfigEntry<int> WarmupStepsOnLayerGen = null!;

        public static ConfigEntry<bool> OverrideVanillaLayerWeather = null!;
        public static ConfigEntry<bool> EnableBodyTemperatureCoupling = null!;
        public static ConfigEntry<bool> EnableWindForceOnObjects = null!;
        public static ConfigEntry<float> WindForceScale = null!;
        public static ConfigEntry<float> MaxWindAccel = null!;
        public static ConfigEntry<float> WindForceRadiusTiles = null!;

        public static ConfigEntry<bool> EnableParticles = null!;
        public static ConfigEntry<float> OverlayOpacity = null!;
        public static ConfigEntry<float> WallFogMultiplier = null!;
        public static ConfigEntry<float> PlayerFogClearInnerTiles = null!;
        public static ConfigEntry<float> PlayerFogClearOuterTiles = null!;
        public static ConfigEntry<float> PlayerFogMinFactor = null!;
        public static ConfigEntry<float> FluidSyncIntervalSec = null!;
        public static ConfigEntry<bool> EnableWaterCycleFluidSync = null!;
        public static ConfigEntry<float> EvapMassPerWaterTile = null!;
        public static ConfigEntry<float> RainMassPerWaterTile = null!;
        public static ConfigEntry<KeyCode> CycleDebugOverlayKey = null!;
        public static ConfigEntry<KeyCode> ToggleHudKey = null!;
        public static ConfigEntry<bool> ShowHudByDefault = null!;

        public static void Bind(ConfigFile cfg)
        {
            Enabled = cfg.Bind("General", "Enabled", true, "Enable WeatherExpansionMod CUDA 2D weather simulation.");

            PreferredSubdivision = cfg.Bind(
                "Grid", "PreferredSubdivision", 2,
                new ConfigDescription("Simulation cells per 1 game tile axis (2 => 1 game tile = 2x2 = 4 sim cells).",
                    new AcceptableValueRange<int>(1, 2)));

            MaxSimCells = cfg.Bind(
                "Grid", "MaxSimCells", 4_200_000,
                "Maximum total CUDA grid cells before automatically falling back from 2x2 to 1x1 or a player-centered window (4.2M ~= 2048x2048 ~= 468MB VRAM, ~48% RTX 5060 @30Hz).");

            FallbackWindowTiles = cfg.Bind(
                "Grid", "FallbackWindowTiles", 512,
                "Window size in game tiles when a layer exceeds MaxSimCells even at 1x1.");

            SimHz = cfg.Bind(
                "Simulation", "SimHz", 30f,
                new ConfigDescription("Weather solver tick rate (Hz).", new AcceptableValueRange<float>(10f, 60f)));

            SorIterations = cfg.Bind(
                "Simulation", "SorIterations", 18,
                new ConfigDescription("Half-width Red-Black SOR pressure projection iterations per step.", new AcceptableValueRange<int>(10, 48)));

            WarmupStepsOnLayerGen = cfg.Bind(
                "Simulation", "WarmupStepsOnLayerGen", 90,
                new ConfigDescription("Pre-simulation steps run right after layer generation so weather is already active when you land.", new AcceptableValueRange<int>(0, 300)));

            OverrideVanillaLayerWeather = cfg.Bind(
                "Gameplay", "OverrideVanillaLayerWeather", true,
                "Disable vanilla flat fog/rain overlays (Foggy/Rainy modifiers & Layer 6/7) and drive them physically inside the CUDA weather system.");

            EnableBodyTemperatureCoupling = cfg.Bind(
                "Gameplay", "EnableBodyTemperatureCoupling", true,
                "Use local simulated temperature + windchill + humidity in Body.HandleBodyTemperature and wet the player under real rain/snow.");

            EnableWindForceOnObjects = cfg.Bind(
                "Physics", "EnableWindForceOnObjects", true,
                "Apply aerodynamic wind drag forces to player, limbs, items, and nearby Rigidbody2D objects.");

            WindForceScale = cfg.Bind(
                "Physics", "WindForceScale", 0.35f,
                new ConfigDescription("Global wind drag force multiplier (tunable while testing).", new AcceptableValueRange<float>(0f, 5f)));

            MaxWindAccel = cfg.Bind(
                "Physics", "MaxWindAccel", 18f,
                new ConfigDescription("Maximum wind-induced acceleration (tiles/s^2) to prevent high-speed physics tunneling.", new AcceptableValueRange<float>(1f, 60f)));

            WindForceRadiusTiles = cfg.Bind(
                "Physics", "WindForceRadiusTiles", 48f,
                new ConfigDescription("Radius around player (in game tiles) to apply aerodynamic wind forces.", new AcceptableValueRange<float>(12f, 128f)));

            EnableParticles = cfg.Bind(
                "Visuals", "EnableParticles", true,
                "Spawn wind-driven Unity ParticleSystem snowflakes and raindrops where snow/rain forms.");

            OverlayOpacity = cfg.Bind(
                "Visuals", "OverlayOpacity", 0.85f,
                new ConfigDescription("Alpha multiplier for the natural cloud/mist/snow visual overlay.", new AcceptableValueRange<float>(0f, 1f)));

            WallFogMultiplier = cfg.Bind(
                "Visuals", "WallFogMultiplier", 0.5f,
                new ConfigDescription("Fog/cloud alpha multiplier inside solid walls compared to open air (0.5 = halved).", new AcceptableValueRange<float>(0f, 1f)));

            PlayerFogClearInnerTiles = cfg.Bind(
                "Visuals", "PlayerFogClearInnerTiles", 5.0f,
                new ConfigDescription("Inner radius around player (game tiles) with minimum fog opacity for clear visibility.", new AcceptableValueRange<float>(0f, 30f)));

            PlayerFogClearOuterTiles = cfg.Bind(
                "Visuals", "PlayerFogClearOuterTiles", 16.0f,
                new ConfigDescription("Outer radius around player (game tiles) where fog smoothly returns to full strength.", new AcceptableValueRange<float>(1f, 60f)));

            PlayerFogMinFactor = cfg.Bind(
                "Visuals", "PlayerFogMinFactor", 0.18f,
                new ConfigDescription("Minimum fog opacity multiplier directly around the player (0 = completely clear, 1 = no reduction).", new AcceptableValueRange<float>(0f, 1f)));

            FluidSyncIntervalSec = cfg.Bind(
                "Simulation", "FluidSyncIntervalSec", 0.5f,
                new ConfigDescription("Seconds between incremental FluidManager water-cell sync scans around the active viewport.", new AcceptableValueRange<float>(0.1f, 5.0f)));

            EnableWaterCycleFluidSync = cfg.Bind(
                "Hydrology", "EnableWaterCycleFluidSync", true,
                "Evaporation depletes FluidManager water tiles and rainfall spawns new water tiles on the ground.");

            EvapMassPerWaterTile = cfg.Bind(
                "Hydrology", "EvapMassPerWaterTile", 0.18f,
                new ConfigDescription("Cumulative evaporated mass per game tile before removing 1 water tile (~15-30s under breezy dry air).", new AcceptableValueRange<float>(0.02f, 2.0f)));

            RainMassPerWaterTile = cfg.Bind(
                "Hydrology", "RainMassPerWaterTile", 0.035f,
                new ConfigDescription("Cumulative rain deposited on a ground tile before spawning 1 water tile (~15-30s under steady rain).", new AcceptableValueRange<float>(0.005f, 1.0f)));

            CycleDebugOverlayKey = cfg.Bind(
                "Debug", "CycleDebugOverlayKey", KeyCode.F8,
                "Hotkey to cycle overlay mode: Natural -> Temperature -> Humidity -> Wind -> Latent Heat -> Off.");

            ToggleHudKey = cfg.Bind(
                "Debug", "ToggleHudKey", KeyCode.F9,
                "Hotkey to toggle the weather status & GPU telemetry HUD.");

            ShowHudByDefault = cfg.Bind(
                "Debug", "ShowHudByDefault", true,
                "Show compact local weather status HUD on startup.");
        }
    }
}
