import time
import numpy as np
from weather_sim_binding import WeatherSim, CELL_AIR, CELL_SOLID, CELL_WATER, load_dll


def build_benchmark_cavern(w, h):
    cells = np.zeros((h, w), dtype=np.uint8)
    cells[0:4, :] = CELL_SOLID
    cells[-4:, :] = CELL_SOLID
    cells[:, 0:4] = CELL_SOLID
    cells[:, -4:] = CELL_SOLID
    # Bottom pool
    cells[4:10, w // 4 : (3 * w) // 4] = CELL_WATER
    # A couple of internal ledges/pillars
    cells[h // 3 : h // 3 + 6, w // 5 : (2 * w) // 5] = CELL_SOLID
    cells[(2 * h) // 3 : (2 * h) // 3 + 6, (3 * w) // 5 : (4 * w) // 5] = CELL_SOLID
    heat = np.zeros((h, w), dtype=np.float32)
    heat[10:18, (3 * w) // 8 : (5 * w) // 8] = 3.5
    return cells, heat


def benchmark_config(dll, tiles_w, tiles_h, subdiv=2, sor_iters=28, warmup=15, steps=60):
    sim_w = tiles_w * subdiv
    sim_h = tiles_h * subdiv
    cells, heat = build_benchmark_cavern(sim_w, sim_h)

    with WeatherSim(sim_w, sim_h, dll=dll, cellSizeTiles=1.0 / subdiv, sorIterations=sor_iters) as sim:
        sim.upload_cells(cells)
        sim.upload_heat_sources(heat)
        sim.reset_fields()

        for _ in range(warmup):
            sim.step(1.0 / 30.0)

        gpu_ms = []
        t0 = time.perf_counter()
        for _ in range(steps):
            sim.step(1.0 / 30.0)
            gpu_ms.append(sim.stats().lastStepGpuMs)
        wall_ms = (time.perf_counter() - t0) * 1000.0 / steps

        # Measure realistic per-frame game bridge readbacks:
        # 1) 256 Rigidbody2D entities sampled
        rng = np.random.default_rng(42)
        ex = rng.uniform(8, sim_w - 8, size=256).astype(np.float32)
        ey = rng.uniform(8, sim_h - 8, size=256).astype(np.float32)
        t_s0 = time.perf_counter()
        for _ in range(20):
            sim.sample_points(ex, ey)
        sample_ms = (time.perf_counter() - t_s0) * 1000.0 / 20

        # 2) Camera viewport RGBA8 readback (e.g. 256x160 sim cells = 128x80 game tiles visible)
        vw, vh = min(256, sim_w), min(160, sim_h)
        t_v0 = time.perf_counter()
        for _ in range(20):
            sim.read_view_rgba8((sim_w - vw) // 2, (sim_h - vh) // 2, vw, vh, mode=0)
        view_ms = (time.perf_counter() - t_v0) * 1000.0 / 20

        st = sim.stats()
        avg_gpu = float(np.mean(gpu_ms))
        p95_gpu = float(np.percentile(gpu_ms, 95))
        vram_mb = st.allocatedVramBytes / (1024.0 * 1024.0)
        total_vram_gb = st.totalVramBytes / (1024.0 ** 3)
        free_vram_gb = st.freeVramBytes / (1024.0 ** 3)

        # Estimate GPU busy percentage at 30 Hz and 60 Hz sim rates
        load_30hz = (avg_gpu / (1000.0 / 30.0)) * 100.0
        load_60hz = (avg_gpu / (1000.0 / 60.0)) * 100.0

        return {
            "tiles": f"{tiles_w}x{tiles_h}",
            "sim": f"{sim_w}x{sim_h}",
            "cells_M": (sim_w * sim_h) / 1e6,
            "sor": sor_iters,
            "vram_mb": vram_mb,
            "avg_gpu_ms": avg_gpu,
            "p95_gpu_ms": p95_gpu,
            "wall_ms": wall_ms,
            "sample256_ms": sample_ms,
            "view_ms": view_ms,
            "load_30hz": load_30hz,
            "load_60hz": load_60hz,
            "total_vram_gb": total_vram_gb,
            "free_vram_gb": free_vram_gb,
        }


def main():
    dll = load_dll()
    # All tested with subdiv=2 (1 game tile = 2x2 = 4 simulation cells)
    cases = [
        (256, 256, 2, 28),   # 512 x 512   (262K cells) - Active camera window
        (384, 512, 2, 28),   # 768 x 1024  (786K cells) - Medium layer
        (512, 512, 2, 28),   # 1024 x 1024 (1.05M cells) - Large square layer
        (512, 1024, 2, 28),  # 1024 x 2048 (2.10M cells) - Tall full layer
        (1024, 1024, 2, 28), # 2048 x 2048 (4.19M cells) - Huge stress test
        (512, 1024, 2, 40),  # 1024 x 2048 w/ higher SOR accuracy
    ]

    print("=== WeatherSimCuda RTX 5060 Performance Benchmark (1 game tile = 2x2 = 4 sim cells) ===")
    header = (
        f"{'GameTiles':>10} | {'SimGrid':>10} | {'Cells':>6} | {'SOR':>3} | "
        f"{'VRAM(MB)':>8} | {'GPU(ms)':>7} | {'P95(ms)':>7} | {'Ent256':>6} | {'View':>6} | "
        f"{'@30Hz%':>6} | {'@60Hz%':>6}"
    )
    print(header)
    print("-" * len(header))

    for tw, th, sub, sor in cases:
        r = benchmark_config(dll, tw, th, subdiv=sub, sor_iters=sor)
        print(
            f"{r['tiles']:>10} | {r['sim']:>10} | {r['cells_M']:5.2f}M | {r['sor']:>3} | "
            f"{r['vram_mb']:8.1f} | {r['avg_gpu_ms']:7.3f} | {r['p95_gpu_ms']:7.3f} | "
            f"{r['sample256_ms']:5.2f}ms | {r['view_ms']:5.2f}ms | "
            f"{r['load_30hz']:5.1f}% | {r['load_60hz']:5.1f}%"
        )


if __name__ == "__main__":
    main()
