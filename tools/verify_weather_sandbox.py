import argparse
import os
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from weather_sim_binding import (
    WeatherSim,
    load_dll,
    CELL_AIR,
    CELL_SOLID,
    CELL_WATER,
    FIELD_U,
    FIELD_V,
    FIELD_TEMP,
    FIELD_QV,
    FIELD_QC,
    FIELD_QR,
    FIELD_QI,
    FIELD_SNOW_ACCUM,
    FIELD_RH,
    FIELD_DIVERGENCE,
    FIELD_LATENT_RATE,
    FIELD_EVAP_ACCUM,
)


def configure_matplotlib_theme():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "#0d1117"
    plt.rcParams["axes.facecolor"] = "#0d1117"
    plt.rcParams["axes.edgecolor"] = "#30363d"
    plt.rcParams["axes.labelcolor"] = "#c9d1d9"
    plt.rcParams["text.color"] = "#e6edf3"
    plt.rcParams["xtick.color"] = "#8b949e"
    plt.rcParams["ytick.color"] = "#8b949e"


def test_v03_physics_integrity(dll):
    """Verify v0.3.0 physics: outer-border containment, free internal flow, Dalton in-water evaporation, zero NaN."""
    W, H = 192, 128
    cells = np.zeros((H, W), dtype=np.uint8)
    cells[20:108, 88:96] = CELL_SOLID
    cells[44:84, 30:56] = CELL_WATER

    fx = np.zeros((H, W), dtype=np.float32)
    fx[48:80, 18:174] = 8.0

    with WeatherSim(W, H, dll=dll, midLayerTempC=12.0, initialRelHumidity=0.58, sorIterations=18) as sim:
        sim.upload_cells(cells)
        sim.upload_wind_sources(fx, None)
        sim.reset_fields()

        for _ in range(200):
            sim.step(1.0 / 30.0)

        u = sim.copy_field(FIELD_U)
        v = sim.copy_field(FIELD_V)
        T = sim.copy_field(FIELD_TEMP)
        rh = sim.copy_field(FIELD_RH)
        evap = sim.copy_field(FIELD_EVAP_ACCUM)
        div = sim.copy_field(FIELD_DIVERGENCE)

        wall_u = float(np.mean(u[54:74, 88:96]))
        water_u = float(np.mean(u[54:74, 34:52]))
        downwind_rh = float(np.mean(rh[54:74, 62:92]))
        upwind_rh = float(np.mean(rh[54:74, 8:24]))
        evap_in_water = float(np.sum(evap[cells == CELL_WATER]))
        evap_outside = float(np.sum(evap[cells != CELL_WATER]))
        mean_abs_div = float(np.mean(np.abs(div)))
        has_nan = bool(np.isnan(u).any() or np.isnan(v).any() or np.isnan(T).any())

    passed = (
        (not has_nan)
        and (wall_u > 2.0)
        and (water_u > 2.0)
        and (downwind_rh > upwind_rh + 0.12)
        and (evap_in_water > 0.1)
        and (evap_outside == 0.0)
    )
    print("\n=== [Test 1] v0.3.0 Open-Interior Flow & In-Water Dalton Evaporation ===")
    print(f"  Airflow through wall / water  : wall_u={wall_u:.2f}, water_u={water_u:.2f} tiles/s")
    print(f"  In-water humidification       : upwind RH={upwind_rh*100:.1f}% -> downwind RH={downwind_rh*100:.1f}%")
    print(f"  Evaporation localization      : inside_water={evap_in_water:.3f}, outside={evap_outside:.3f}")
    print(f"  Mean |div| (half-width SOR)   : {mean_abs_div:.5f}")
    print(f"  NaN detected                  : {has_nan}")
    print(f"  RESULT                        : {'PASS' if passed else 'FAIL'}")
    return passed


def build_cavern_scenario(W=384, H=256):
    """
    2D cavern cross-section (192 x 128 game tiles at 1 game tile = 2x2 sim cells):
    - Warm geothermal water pool at bottom center (evaporates & depletes under wind)
    - Uneven rocky cavern floor, ledges, pillar, and roof (catches rain puddles & snow!)
    - Wall interiors do not block airflow; wall fog displays at 0.5x opacity (matching v0.3.0 in-game)
    """
    cells = np.zeros((H, W), dtype=np.uint8)
    xs = np.arange(W)

    # Outer rocky crust
    cells[0:6, :] = CELL_SOLID
    cells[-6:, :] = CELL_SOLID
    cells[:, 0:6] = CELL_SOLID
    cells[:, -6:] = CELL_SOLID

    # Uneven cavern dome roof
    ceil_y = (H - 12 - 14 * np.cos((xs / W) * 2.0 * np.pi) + 5 * np.sin((xs / W) * 6.0 * np.pi)).astype(int)
    for x in range(W):
        cells[max(H - 40, ceil_y[x]) :, x] = CELL_SOLID

    # Rocky floor with sunken basin for central water pool
    floor_y = (16 + 10 * np.sin((xs / W) * 3.0 * np.pi)).astype(int)
    basin_left, basin_right = int(W * 0.26), int(W * 0.68)
    for x in range(W):
        if basin_left <= x <= basin_right:
            depth = int(10 * np.sin(np.pi * (x - basin_left) / max(basin_right - basin_left, 1)))
            fy = max(6, 16 - depth)
        else:
            fy = max(8, floor_y[x])
        cells[:fy, x] = CELL_SOLID

    # Fill basin with deep water (aligned to 2x2 tile grid)
    water_level = 24
    for x in range(basin_left, basin_right + 1):
        for y in range(6, water_level):
            if cells[y, x] == CELL_AIR:
                cells[y, x] = CELL_WATER

    # Left & right elevated rocky ledges (catches both falling rain puddles and snow!)
    cells[76:84, 14:104] = CELL_SOLID
    cells[132:140, 20:86] = CELL_SOLID
    cells[92:118, 206:228] = CELL_SOLID
    for x in range(236, 336):
        ly = 84 + int((x - 236) * 0.22)
        cells[ly : ly + 8, x] = CELL_SOLID

    # Geothermal heat source around & above central pool
    heat = np.zeros((H, W), dtype=np.float32)
    pool_cx = 0.5 * (basin_left + basin_right)
    for y in range(14, water_level + 16):
        for x in range(basin_left + 8, basin_right - 8):
            rx = (x - pool_cx) / (0.45 * (basin_right - basin_left))
            ry = (y - water_level) / 14.0
            w_g = np.exp(-(rx * rx * 1.3 + ry * ry * 2.0))
            heat[y, x] = 2.8 * w_g

    # Cool upper cavern dome
    for x in range(10, W - 10):
        for y in range(H - 36, H - 6):
            frac = (y - (H - 36)) / 30.0
            heat[y, x] -= 0.40 * frac

    return cells, heat


def get_preset(preset_name):
    if preset_name == "rainy":
        return (
            "多雨对流层 (下暖湿 · 上冷 · 降雨积水)",
            dict(
                cellSizeTiles=0.5,
                subdivision=2,
                midLayerTempC=9.5,
                bgLapseRateCPerTile=0.048,
                adiabaticLapseCPerTile=0.072,
                buoyancyScale=60.0,
                vorticityEps=0.90,
                velocityDamping=0.008,
                maxSpeedTilesPerSec=36.0,
                bottomTempBoostC=6.5,
                topTempDropC=6.5,
                bottomMoistureSource=0.0,
                evapRateCoeff=0.32,
                evapWindFactor=0.35,
                thermalDiffusivity=0.01,
                wallTempRelaxRate=0.01,
                autoconvertThresh=0.00055,
                autoconvertRate=0.20,
                rainFallSpeed=8.0,
                snowFallSpeed=1.5,
                sorIterations=18,
                initialRelHumidity=0.84,
                evapMassPerWaterTile=0.70,
                rainMassPerWaterTile=0.70,
            ),
            (-2.0, 22.0),
        )
    # Default "snow": mixed rain + sub-zero deposition snowstorm!
    return (
        "寒带对流雪层 (上升绝热降温 · 凝华成雪 · 蒸发与降雨闭环)",
        dict(
            cellSizeTiles=0.5,
            subdivision=2,
            midLayerTempC=-0.5,
            bgLapseRateCPerTile=0.045,
            adiabaticLapseCPerTile=0.075,
            buoyancyScale=60.0,
            vorticityEps=0.90,
            velocityDamping=0.008,
            maxSpeedTilesPerSec=36.0,
            bottomTempBoostC=3.0,
            topTempDropC=3.0,
            evapRateCoeff=0.20,
            evapWindFactor=0.35,
            evapCoolingScale=0.75,
            latentHeatScale=1.0,
            thermalDiffusivity=0.01,
            wallTempRelaxRate=0.01,
            autoconvertThresh=0.00065,
            autoconvertRate=0.16,
            freezeRate=0.45,
            rainFallSpeed=8.0,
            snowFallSpeed=1.5,
            snowDepositRate=1.1,
            sorIterations=18,
            initialRelHumidity=0.74,
            evapMassPerWaterTile=0.70,
            rainMassPerWaterTile=0.70,
        ),
        (-10.0, 14.0),
    )


def apply_fluid_events_to_cells(sim, cells, sub, counters, do_water_flow=True):
    """Apply GPU water-cycle events (-1 evap / +1 rainfall) and run mini cellular water flow like FluidManager."""
    events = sim.pop_fluid_events(2048)
    H, W = cells.shape
    tiles_h, tiles_w = H // sub, W // sub
    patch_x, patch_y, patch_t = [], [], []

    def set_tile(tx, ty, ctype):
        sx0, sy0 = tx * sub, ty * sub
        cells[sy0 : sy0 + sub, sx0 : sx0 + sub] = ctype
        for dy in range(sub):
            for dx in range(sub):
                patch_x.append(sx0 + dx)
                patch_y.append(sy0 + dy)
                patch_t.append(ctype)

    for tx, ty, delta in events:
        if tx < 0 or ty < 0 or tx >= tiles_w or ty >= tiles_h:
            continue
        cur = cells[ty * sub, tx * sub]
        if delta < 0 and cur == CELL_WATER:
            cells[ty * sub : (ty + 1) * sub, tx * sub : (tx + 1) * sub] = CELL_AIR
            counters["evap"] += 1
        elif delta > 0 and cur == CELL_AIR:
            set_tile(tx, ty, CELL_WATER)
            counters["rain"] += 1

    # Simple cellular-automata gravity + sideways leveling (mirrors FluidManager.SimulationStep)
    # so rain-generated puddles naturally level out and spill off ledges instead of stacking into towers!
    if do_water_flow:
        tile_grid = cells[::sub, ::sub]
        ys, xs = np.where(tile_grid == CELL_WATER)
        if len(xs) > 0:
            order = np.argsort(ys)
            for idx in order:
                tx = int(xs[idx])
                ty = int(ys[idx])
                if tile_grid[ty, tx] != CELL_WATER:
                    continue
                if ty > 1 and tile_grid[ty - 1, tx] == CELL_AIR:
                    set_tile(tx, ty, CELL_AIR)
                    set_tile(tx, ty - 1, CELL_WATER)
                else:
                    left_ok = tx > 1 and tile_grid[ty, tx - 1] == CELL_AIR
                    right_ok = tx < tiles_w - 2 and tile_grid[ty, tx + 1] == CELL_AIR
                    if left_ok and right_ok:
                        nx = tx + (1 if ((tx + ty + counters["rain"]) & 1) == 0 else -1)
                        set_tile(tx, ty, CELL_AIR)
                        set_tile(nx, ty, CELL_WATER)
                    elif left_ok:
                        set_tile(tx, ty, CELL_AIR)
                        set_tile(tx - 1, ty, CELL_WATER)
                    elif right_ok:
                        set_tile(tx, ty, CELL_AIR)
                        set_tile(tx + 1, ty, CELL_WATER)

    if patch_x:
        sim.upload_cells_patch(patch_x, patch_y, patch_t)


def build_wall_outline(solid_mask):
    """1-cell crisp rock silhouette outline so walls stay legible without hiding continuous fields."""
    edge = np.zeros_like(solid_mask, dtype=bool)
    edge[1:, :] |= solid_mask[1:, :] & ~solid_mask[:-1, :]
    edge[:-1, :] |= solid_mask[:-1, :] & ~solid_mask[1:, :]
    edge[:, 1:] |= solid_mask[:, 1:] & ~solid_mask[:, :-1]
    edge[:, :-1] |= solid_mask[:, :-1] & ~solid_mask[:, 1:]
    return edge


def render_4panel(sim, cells, t_sec, preset_label, temp_range, counters, out_path=None, fig_axes=None, colorbars=None):
    W, H = sim.width, sim.height
    u = sim.copy_field(FIELD_U)
    v = sim.copy_field(FIELD_V)
    T = sim.copy_field(FIELD_TEMP)
    qc = sim.copy_field(FIELD_QC)
    qr = sim.copy_field(FIELD_QR)
    qi = sim.copy_field(FIELD_QI)
    sacc = sim.copy_field(FIELD_SNOW_ACCUM)
    latent = sim.copy_field(FIELD_LATENT_RATE)

    air = cells == CELL_AIR
    solid = cells == CELL_SOLID
    water = cells == CELL_WATER
    speed = np.hypot(u, v)
    wall_edge = build_wall_outline(solid)

    # Translucent rock/water overlay for continuous field panels (walls do NOT block airflow in v0.3.0!)
    geom_overlay = np.zeros((H, W, 4), dtype=np.float32)
    geom_overlay[solid] = [0.16, 0.16, 0.19, 0.42]
    geom_overlay[wall_edge] = [0.68, 0.70, 0.76, 0.78]
    geom_overlay[water] = [0.10, 0.52, 0.95, 0.58]

    extent = [0, W * sim.cfg.cellSizeTiles, 0, H * sim.cfg.cellSizeTiles]
    gx = np.linspace(0, W * sim.cfg.cellSizeTiles, W)
    gy = np.linspace(0, H * sim.cfg.cellSizeTiles, H)

    created_new = fig_axes is None
    if created_new:
        fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=120, constrained_layout=True)
        colorbars = {}
    else:
        fig, axes = fig_axes
        for ax in axes.ravel():
            ax.clear()

    st = sim.stats()
    fig.suptitle(
        f"天气拓展 WeatherExpansionMod (CUDA 2D 实时天气模拟 v1.0.1)  |  {preset_label}\n"
        f"t = {t_sec:5.1f}s  |  网格 {W}×{H} (1游戏格=4模拟格)  |  地层中段温度 T_mid = {sim.cfg.midLayerTempC:+.1f}°C  |  "
        f"GPU {st.lastStepGpuMs:.2f}ms · 水循环: 蒸发 -{counters['evap']} 格 / 降雨生水 +{counters['rain']} 格",
        fontsize=12.5,
        fontweight="bold",
        color="#f0f6fc",
    )

    # 1) Airflow Speed + Wind Vectors (2x wind speed, open-interior flow, sealed outer borders)
    ax = axes[0, 0]
    im1 = ax.imshow(speed, origin="lower", extent=extent, cmap="viridis", vmin=0.0, vmax=16.0)
    ax.imshow(geom_overlay, origin="lower", extent=extent)
    step_q = 12
    ax.quiver(
        gx[::step_q], gy[::step_q],
        u[::step_q, ::step_q], v[::step_q, ::step_q],
        color="white", alpha=0.72, scale=145, width=0.0022
    )
    ax.set_title("1. 气流风速 (tiles/s) 与风向 — 仅外边界封闭 · 墙内与水内自由穿流", fontsize=10.5, pad=6)
    ax.set_xlabel("X (游戏格)")
    ax.set_ylabel("Y (游戏格)")
    if created_new:
        cb = plt.colorbar(im1, ax=ax, fraction=0.032, pad=0.02)
        cb.set_label("风速 (tiles/s)", fontsize=8.5)
        colorbars[0] = cb

    # 2) Temperature field + 0°C freezing isotherm + mid-layer temperature anchor
    ax = axes[0, 1]
    t_min, t_max = temp_range
    im2 = ax.imshow(T, origin="lower", extent=extent, cmap="coolwarm", vmin=t_min, vmax=t_max)
    ax.imshow(geom_overlay, origin="lower", extent=extent)
    try:
        cs = ax.contour(gx, gy, T, levels=[0.0], colors=["#ffffff"], linewidths=1.5, linestyles="--")
        ax.clabel(cs, fmt="0°C", fontsize=8.5, colors="#ffffff")
    except Exception:
        pass
    y_mid_tiles = 0.5 * H * sim.cfg.cellSizeTiles
    ax.axhline(y_mid_tiles, color="#ffd866", linestyle=":", linewidth=1.1, alpha=0.9)
    ax.text(
        3, y_mid_tiles + 2.2,
        f"地层中段基准 T_mid = {sim.cfg.midLayerTempC:+.1f}°C",
        color="#ffd866", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#0d1117", edgecolor="none", alpha=0.65),
    )
    ax.set_title("2. 温度场 (°C) — 气流上升绝热膨胀降温 · 0°C 冰点线", fontsize=10.5, pad=6)
    ax.set_xlabel("X (游戏格)")
    ax.set_ylabel("Y (游戏格)")
    if created_new:
        cb = plt.colorbar(im2, ax=ax, fraction=0.032, pad=0.02)
        cb.set_label("温度 (°C)", fontsize=8.5)
        colorbars[1] = cb

    # 3) Water Cycle & Phase Composite (NO artificial RH<100% haze; wall fog halved 0.5x!)
    ax = axes[1, 0]
    comp = np.zeros((H, W, 3), dtype=np.float32)
    comp[air] = [0.06, 0.08, 0.12]
    comp[solid] = [0.22, 0.21, 0.24]
    comp[water] = [0.10, 0.46, 0.88]
    comp[wall_edge] = [0.38, 0.38, 0.42]

    wall_mult = np.where(solid, 0.5, 1.0)
    c_a = np.clip(qc * 340.0, 0.0, 0.72) * wall_mult
    r_a = np.clip(qr * 520.0, 0.0, 0.78) * wall_mult
    i_a = np.clip(qi * 480.0, 0.0, 0.90) * wall_mult
    s_a = np.clip(sacc * 120.0, 0.0, 1.0) * air

    for ch, col in enumerate([0.85, 0.89, 0.93]):
        comp[:, :, ch] = comp[:, :, ch] * (1.0 - c_a) + col * c_a
    for ch, col in enumerate([0.22, 0.62, 1.00]):
        comp[:, :, ch] = comp[:, :, ch] * (1.0 - r_a) + col * r_a
    for ch, col in enumerate([0.60, 0.95, 1.00]):
        comp[:, :, ch] = comp[:, :, ch] * (1.0 - i_a) + col * i_a
    for ch, col in enumerate([1.00, 1.00, 1.00]):
        comp[:, :, ch] = comp[:, :, ch] * (1.0 - s_a) + col * s_a

    ax.imshow(comp, origin="lower", extent=extent)
    ax.set_title("3. 水循环与相变 — 云雾(白) / 降雨(蓝) / 飘雪(青) / 积雪(纯白) · 墙内雾减半", fontsize=10.5, pad=6)
    ax.set_xlabel("X (游戏格)")
    ax.set_ylabel("Y (游戏格)")
    ax.text(
        4, H * sim.cfg.cellSizeTiles - 11,
        f"峰值 云={qc.max()*1e3:.2f} | 雨={qr.max()*1e3:.2f} | 雪={qi.max()*1e3:.2f} g/kg  ·  "
        f"水池蒸发 -{counters['evap']} 格 | 降雨生水 +{counters['rain']} 格",
        color="#f0f6fc", fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#0d1117", edgecolor="#30363d", alpha=0.78),
    )

    # 4) Latent Heat Release / Cooling Rate (shows in-water evaporative cooling & aloft condensation/deposition heating!)
    ax = axes[1, 1]
    im4 = ax.imshow(latent, origin="lower", extent=extent, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    ax.imshow(geom_overlay, origin="lower", extent=extent)
    ax.set_title("4. 相变潜热速率 (°C/s) — 红: 冷凝/凝华放热 · 蓝: 水内蒸发/融雪吸热", fontsize=10.5, pad=6)
    ax.set_xlabel("X (游戏格)")
    ax.set_ylabel("Y (游戏格)")
    if created_new:
        cb = plt.colorbar(im4, ax=ax, fraction=0.032, pad=0.02)
        cb.set_label("潜热速率 (°C/s)", fontsize=8.5)
        colorbars[3] = cb

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=120, facecolor=fig.get_facecolor())
        plt.close(fig)
    return fig, axes, colorbars


def run_cavern_verification(dll, out_dir, preset_name="snow"):
    W, H = 384, 256
    cells, heat = build_cavern_scenario(W, H)
    preset_label, cfg_overrides, temp_range = get_preset(preset_name)
    sub = cfg_overrides.get("subdivision", 2)

    dt = 1.0 / 30.0
    total_steps = 2400  # 80s
    snap_steps = {450: "t15s", 1200: "t40s", 2400: "t80s"}
    counters = {"evap": 0, "rain": 0}

    hist_t, hist_max_spd, hist_qc_sum, hist_qr_sum, hist_qi_sum = [], [], [], [], []
    hist_evap_tiles, hist_rain_tiles, hist_t_low, hist_t_high = [], [], [], []

    t_start = time.perf_counter()
    with WeatherSim(W, H, dll=dll, **cfg_overrides) as sim:
        sim.upload_cells(cells)
        sim.upload_heat_sources(heat)
        sim.reset_fields()

        for step in range(1, total_steps + 1):
            sim.step(dt)
            apply_fluid_events_to_cells(sim, cells, sub, counters)

            if step % 30 == 0:
                t_sec = step * dt
                u = sim.copy_field(FIELD_U)
                v = sim.copy_field(FIELD_V)
                T = sim.copy_field(FIELD_TEMP)
                qc = sim.copy_field(FIELD_QC)
                qr = sim.copy_field(FIELD_QR)
                qi = sim.copy_field(FIELD_QI)

                hist_t.append(t_sec)
                hist_max_spd.append(float(np.max(np.hypot(u, v))))
                hist_qc_sum.append(float(np.sum(qc)))
                hist_qr_sum.append(float(np.sum(qr)))
                hist_qi_sum.append(float(np.sum(qi)))
                hist_evap_tiles.append(counters["evap"])
                hist_rain_tiles.append(counters["rain"])
                hist_t_low.append(float(np.mean(T[16:48, 96:260])))
                hist_t_high.append(float(np.mean(T[192:232, 80:280])))

            if step in snap_steps:
                tag = snap_steps[step]
                out_png = os.path.join(out_dir, f"cavern_weather_{preset_name}_{tag}.png")
                render_4panel(sim, cells, step * dt, preset_label, temp_range, counters, out_path=out_png)

        st = sim.stats()

    elapsed = time.perf_counter() - t_start

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), dpi=120, constrained_layout=True)
    fig.suptitle(f"天气拓展 WeatherExpansionMod v0.3.0 — 80秒物理演化曲线 ({preset_label})", fontsize=12.5, fontweight="bold")

    axes[0, 0].plot(hist_t, hist_t_low, label="底层水池上方均温 (°C)", color="#ff7b72", linewidth=2)
    axes[0, 0].plot(hist_t, hist_t_high, label="高层洞顶均温 (°C)", color="#79c0ff", linewidth=2)
    axes[0, 0].axhline(0.0, color="#8b949e", linestyle="--", linewidth=1)
    axes[0, 0].set_title("垂直温差演化 (底层 vs 高层)")
    axes[0, 0].set_ylabel("温度 (°C)")
    axes[0, 0].legend(facecolor="#161b22", edgecolor="#30363d")
    axes[0, 0].grid(alpha=0.22)

    axes[0, 1].plot(hist_t, hist_max_spd, color="#56d364", linewidth=2, label="最大气流速度 (tiles/s)")
    axes[0, 1].set_title("翻倍风速对流演化")
    axes[0, 1].set_ylabel("速度 (tiles/s)")
    axes[0, 1].grid(alpha=0.22)

    axes[1, 0].plot(hist_t, hist_qc_sum, label="云雾总量 Σqc", color="#d2a8ff", linewidth=2)
    axes[1, 0].plot(hist_t, hist_qr_sum, label="降雨总量 Σqr", color="#58a6ff", linewidth=2)
    axes[1, 0].plot(hist_t, hist_qi_sum, label="雪晶总量 Σqi", color="#76e3ea", linewidth=2)
    axes[1, 0].set_title("冷凝成云 -> 降雨 / 凝华成雪")
    axes[1, 0].set_xlabel("模拟时间 (s)")
    axes[1, 0].set_ylabel("水物质积分")
    axes[1, 0].legend(facecolor="#161b22", edgecolor="#30363d")
    axes[1, 0].grid(alpha=0.22)

    axes[1, 1].plot(hist_t, hist_evap_tiles, label="水池蒸发消耗水格 (-1)", color="#ffa657", linewidth=2)
    axes[1, 1].plot(hist_t, hist_rain_tiles, label="降雨地面生成水格 (+1)", color="#39c5cf", linewidth=2)
    axes[1, 1].set_title("真实水循环: 蒸发扣水 & 降雨生水 (单位: 游戏格)")
    axes[1, 1].set_xlabel("模拟时间 (s)")
    axes[1, 1].set_ylabel("累计水格数")
    axes[1, 1].legend(facecolor="#161b22", edgecolor="#30363d")
    axes[1, 1].grid(alpha=0.22)

    ts_path = os.path.join(out_dir, f"weather_timeseries_{preset_name}.png")
    fig.savefig(ts_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"\n=== [Test 2] 2D Cavern Weather ({preset_name}, 80s / 2400 steps) ===")
    print(f"  Total wall time       : {elapsed:.2f}s (GPU step: {st.lastStepGpuMs:.3f} ms, VRAM: {st.allocatedVramBytes/1e6:.1f} MB)")
    print(f"  Peak wind speed       : {max(hist_max_spd):.2f} tiles/s")
    print(f"  Water cycle tiles     : evaporated=-{counters['evap']}, rain_spawned=+{counters['rain']}")
    print(f"  Outputs saved to      : {out_dir}")


def run_live_or_record(dll, preset_name="snow", record_path=None, duration_sec=30.0, fps=30, sim_speed=2.0):
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    W, H = 384, 256
    cells, heat = build_cavern_scenario(W, H)
    preset_label, cfg_overrides, temp_range = get_preset(preset_name)
    sub = cfg_overrides.get("subdivision", 2)

    sim = WeatherSim(W, H, dll=dll, **cfg_overrides)
    sim.upload_cells(cells)
    sim.upload_heat_sources(heat)
    sim.reset_fields()

    counters = {"evap": 0, "rain": 0}
    state = {"t": 0.0}
    dt = 1.0 / 30.0
    steps_per_frame = max(1, int(round((30.0 / fps) * sim_speed)))

    # 1920x1080 exact 16:9 canvas (16x9 inches @ 120 dpi)
    fig, axes, cbs = render_4panel(sim, cells, 0.0, preset_label, temp_range, counters, out_path=None, fig_axes=None)

    if record_path:
        total_frames = max(1, int(round(duration_sec * fps)))
        os.makedirs(os.path.dirname(os.path.abspath(record_path)), exist_ok=True)
        writer = FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=8000,
            extra_args=["-pix_fmt", "yuv420p", "-crf", "18"],
        )
        print(f"[Record] Rendering {total_frames} frames ({duration_sec:.1f}s @ {fps}fps, {sim_speed}x sim speed) -> {record_path}")
        t0 = time.perf_counter()
        with writer.saving(fig, record_path, dpi=120):
            for f in range(total_frames):
                for _ in range(steps_per_frame):
                    sim.step(dt)
                    state["t"] += dt
                    apply_fluid_events_to_cells(sim, cells, sub, counters)
                render_4panel(sim, cells, state["t"], preset_label, temp_range, counters, fig_axes=(fig, axes), colorbars=cbs)
                writer.grab_frame()
                if (f + 1) % max(1, total_frames // 10) == 0 or f + 1 == total_frames:
                    pct = (f + 1) * 100.0 / total_frames
                    print(f"  Progress: {f+1}/{total_frames} ({pct:.0f}%) | sim_t={state['t']:.1f}s | evap=-{counters['evap']} rain=+{counters['rain']}")
        plt.close(fig)
        sim.close()
        print(f"[Record] Done in {time.perf_counter() - t0:.1f}s -> {record_path}")
        return

    def update(_frame):
        for _ in range(steps_per_frame):
            sim.step(dt)
            state["t"] += dt
            apply_fluid_events_to_cells(sim, cells, sub, counters)
        render_4panel(sim, cells, state["t"], preset_label, temp_range, counters, fig_axes=(fig, axes), colorbars=cbs)

    _anim = FuncAnimation(fig, update, interval=int(1000 / fps), cache_frame_data=False)
    plt.show()
    sim.close()


def main():
    parser = argparse.ArgumentParser(description="WeatherExpansionMod v0.3.0 CUDA 2D 沙盒验证与宣传片录制工具")
    parser.add_argument("--live", action="store_true", help="启动实时交互动画窗口")
    parser.add_argument("--record", type=str, default=None, help="直接导出 1080p H.264 MP4 视频路径 (例如 output/promo_snow.mp4)")
    parser.add_argument("--preset", choices=["snow", "rainy"], default="snow", help="演示预设: snow(寒带对流雪层) 或 rainy(多雨对流层)")
    parser.add_argument("--duration", type=float, default=25.0, help="录制视频时长(秒), 默认 25s")
    parser.add_argument("--fps", type=int, default=30, help="帧率, 默认 30")
    parser.add_argument("--speed", type=float, default=2.5, help="模拟加速倍率 (如 2.5 表示1秒视频演进2.5秒天气), 默认 2.5")
    args = parser.parse_args()

    if not args.live:
        matplotlib.use("Agg")
    configure_matplotlib_theme()

    dll = load_dll()
    if args.live or args.record:
        run_live_or_record(
            dll,
            preset_name=args.preset,
            record_path=args.record,
            duration_sec=args.duration,
            fps=args.fps,
            sim_speed=args.speed,
        )
        return

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, "output")
    os.makedirs(out_dir, exist_ok=True)

    ok1 = test_v03_physics_integrity(dll)
    run_cavern_verification(dll, out_dir, preset_name=args.preset)
    if not ok1:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
