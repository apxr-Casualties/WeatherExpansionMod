using System.Collections.Generic;
using WeatherExpansionMod.Native;

namespace WeatherExpansionMod.Runtime
{
    /// <summary>
    /// Translates active vanilla LayerModifiers and biomeDepth (e.g., Layer 6 fog, Layer 7 fungal rain)
    /// into physical CUDA thermodynamic & moisture boundary parameters, replacing vanilla flat overlays.
    /// </summary>
    internal sealed class WeatherLayerProfile
    {
        public bool IsFoggyModifier { get; set; }
        public bool IsRainyModifier { get; set; }
        public bool IsDryModifier { get; set; }
        public bool IsFloodedModifier { get; set; }
        public bool IsBiome6ColdFog { get; set; }
        public bool IsBiome7ConvectiveRain { get; set; }

        public string SummaryTags { get; private set; } = "标准地层";

        public static WeatherLayerProfile DetectFromWorld(WorldGeneration world)
        {
            var profile = new WeatherLayerProfile();

            if (LayerModifier.availableModifiers != null)
            {
                foreach (var mod in LayerModifier.availableModifiers)
                {
                    if (mod == null || !mod.active) continue;
                    switch (mod.modifierIndex)
                    {
                        case 0: profile.IsFoggyModifier = true; break;   // FoggyModifier
                        case 5: profile.IsFloodedModifier = true; break; // FloodedModifier
                        case 6: profile.IsDryModifier = true; break;     // DryModifier
                        case 7: profile.IsRainyModifier = true; break;   // RainyModifier
                    }
                }
            }

            if (world != null && world.biomeOverride == WorldGeneration.OverrideSceneType.None)
            {
                // Layer 6 (0-indexed biomeDepth == 5): vanilla forced heavy fog + cold zone
                if (world.biomeDepth == 5)
                    profile.IsBiome6ColdFog = true;

                // Layer 7 (0-indexed biomeDepth == 6): vanilla fungal rain zone
                if (world.biomeDepth == 6)
                    profile.IsBiome7ConvectiveRain = true;
            }

            profile.BuildSummary();
            return profile;
        }

        private void BuildSummary()
        {
            var tags = new List<string>();
            if (IsFoggyModifier) tags.Add("多雾(高湿)");
            if (IsRainyModifier) tags.Add("多雨(下热湿/上冷)");
            if (IsDryModifier) tags.Add("干燥(低湿)");
            if (IsFloodedModifier) tags.Add("泛滥(水汽充沛)");
            if (IsBiome6ColdFog) tags.Add("第6层寒雾/降雪");
            if (IsBiome7ConvectiveRain) tags.Add("第7层强对流雨");
            SummaryTags = tags.Count > 0 ? string.Join(" + ", tags) : "标准地层";
        }

        public void ApplyToConfig(ref WeatherSimConfig cfg)
        {
            // Baseline stratum physics (real evaporation, crisp air until true condensation)
            cfg.initialRelHumidity = 0.58f;
            cfg.bgLapseRateCPerTile = 0.032f;
            cfg.adiabaticLapseCPerTile = 0.060f;
            cfg.evapRateCoeff = 0.18f;
            cfg.bottomTempBoostC = 0.0f;
            cfg.topTempDropC = 0.0f;
            cfg.bottomMoistureSource = 0.0f;
            cfg.ambientHumidityTarget = 0.0f;
            cfg.ambientHumidityRate = 0.0f;

            if (IsDryModifier)
            {
                cfg.initialRelHumidity = 0.22f;
                cfg.evapRateCoeff = 0.16f;
            }

            if (IsFloodedModifier)
            {
                cfg.initialRelHumidity = System.Math.Max(cfg.initialRelHumidity, 0.72f);
                cfg.evapRateCoeff = 0.22f;
            }

            // Foggy modifier: high humidity so condensation clouds/fog form naturally without solid whiteout
            if (IsFoggyModifier)
            {
                cfg.initialRelHumidity = System.Math.Max(cfg.initialRelHumidity, 0.90f);
                cfg.ambientHumidityTarget = System.Math.Max(cfg.ambientHumidityTarget, 0.92f);
                cfg.ambientHumidityRate = System.Math.Max(cfg.ambientHumidityRate, 0.025f);
                cfg.evapRateCoeff = System.Math.Max(cfg.evapRateCoeff, 0.24f);
                cfg.bottomMoistureSource = System.Math.Max(cfg.bottomMoistureSource, 0.00010f);
            }

            // Layer 6 native cold fog -> high humidity + cooler ceiling -> mist & snowfall
            if (IsBiome6ColdFog)
            {
                cfg.initialRelHumidity = System.Math.Max(cfg.initialRelHumidity, 0.86f);
                cfg.ambientHumidityTarget = System.Math.Max(cfg.ambientHumidityTarget, 0.89f);
                cfg.ambientHumidityRate = System.Math.Max(cfg.ambientHumidityRate, 0.022f);
                cfg.topTempDropC = System.Math.Max(cfg.topTempDropC, 3.0f);
                cfg.bottomMoistureSource = System.Math.Max(cfg.bottomMoistureSource, 0.00010f);
            }

            // Rainy modifier: warm & humid lower stratum + cooler upper stratum -> sustained convection & rain/snow
            if (IsRainyModifier)
            {
                cfg.initialRelHumidity = System.Math.Max(cfg.initialRelHumidity, 0.82f);
                cfg.bottomTempBoostC += 6.0f;
                cfg.topTempDropC += 6.0f;
                cfg.bgLapseRateCPerTile = System.Math.Max(cfg.bgLapseRateCPerTile, 0.045f);
                cfg.evapRateCoeff = System.Math.Max(cfg.evapRateCoeff, 0.26f);
                cfg.bottomMoistureSource += 0.00028f;
                cfg.autoconvertThresh = 0.00060f;
                cfg.autoconvertRate = 0.18f;
            }

            // Layer 7 native fungal rain -> strong convective rain profile
            if (IsBiome7ConvectiveRain)
            {
                cfg.initialRelHumidity = System.Math.Max(cfg.initialRelHumidity, 0.84f);
                cfg.bottomTempBoostC = System.Math.Max(cfg.bottomTempBoostC, 6.5f);
                cfg.topTempDropC = System.Math.Max(cfg.topTempDropC, 6.5f);
                cfg.bgLapseRateCPerTile = System.Math.Max(cfg.bgLapseRateCPerTile, 0.048f);
                cfg.evapRateCoeff = System.Math.Max(cfg.evapRateCoeff, 0.28f);
                cfg.bottomMoistureSource = System.Math.Max(cfg.bottomMoistureSource, 0.00032f);
                cfg.autoconvertThresh = 0.00055f;
                cfg.autoconvertRate = 0.20f;
            }
        }
    }
}
