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
            // Strict mass-conserving baseline: NO infinite artificial moisture creation!
            cfg.initialRelHumidity = 0.58f;
            cfg.bgLapseRateCPerTile = 0.032f;
            cfg.adiabaticLapseCPerTile = 0.060f;
            cfg.evapRateCoeff = 0.20f;
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
                cfg.initialRelHumidity = System.Math.Max(cfg.initialRelHumidity, 0.74f);
                cfg.evapRateCoeff = 0.25f;
            }

            // Foggy modifier: high initial humidity + stronger real evaporation + mild upper cooling (zero water printed!)
            if (IsFoggyModifier)
            {
                cfg.initialRelHumidity = System.Math.Max(cfg.initialRelHumidity, 0.92f);
                cfg.topTempDropC = System.Math.Max(cfg.topTempDropC, 2.0f);
                cfg.evapRateCoeff = System.Math.Max(cfg.evapRateCoeff, 0.28f);
            }

            // Layer 6 native cold fog -> high initial humidity + colder upper cavern -> natural mist & snow
            if (IsBiome6ColdFog)
            {
                cfg.initialRelHumidity = System.Math.Max(cfg.initialRelHumidity, 0.88f);
                cfg.topTempDropC = System.Math.Max(cfg.topTempDropC, 3.5f);
                cfg.evapRateCoeff = System.Math.Max(cfg.evapRateCoeff, 0.26f);
            }

            // Rainy modifier: strong bottom-warm / top-cold thermal gradient + high evaporation -> closed-loop convective rain!
            if (IsRainyModifier)
            {
                cfg.initialRelHumidity = System.Math.Max(cfg.initialRelHumidity, 0.84f);
                cfg.bottomTempBoostC += 6.5f;
                cfg.topTempDropC += 6.5f;
                cfg.bgLapseRateCPerTile = System.Math.Max(cfg.bgLapseRateCPerTile, 0.048f);
                cfg.evapRateCoeff = System.Math.Max(cfg.evapRateCoeff, 0.32f);
                cfg.autoconvertThresh = 0.00055f;
                cfg.autoconvertRate = 0.20f;
            }

            // Layer 7 native fungal rain -> strong convective rain driven by thermal gradient & real evaporation
            if (IsBiome7ConvectiveRain)
            {
                cfg.initialRelHumidity = System.Math.Max(cfg.initialRelHumidity, 0.86f);
                cfg.bottomTempBoostC = System.Math.Max(cfg.bottomTempBoostC, 7.0f);
                cfg.topTempDropC = System.Math.Max(cfg.topTempDropC, 7.0f);
                cfg.bgLapseRateCPerTile = System.Math.Max(cfg.bgLapseRateCPerTile, 0.050f);
                cfg.evapRateCoeff = System.Math.Max(cfg.evapRateCoeff, 0.34f);
                cfg.autoconvertThresh = 0.00050f;
                cfg.autoconvertRate = 0.22f;
            }
        }
    }
}
