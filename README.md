# 天气拓展 (WeatherExpansionMod)

[English Documentation](README_EN.md) | **简体中文**

为 2D 生存游戏 **《未知伤亡》（Casualties: Unknown）** 打造的 **CUDA GPU 加速二维物理天气模拟 Mod**（基于 BepInEx 5 + Harmony + 原生 CUDA 求解器）。

本 Mod 在独立异步 CUDA 线程上实时求解二维不可压缩流体力学与大气热力学，将原版静态的“多雾 / 多雨”贴图效果替换为真正受地形、水体、岩浆与玩家篝火驱动的动态气象系统。

---

## 核心特性

### 1. 二维 CUDA 实时大气物理引擎 (`WeatherSimCuda.dll`)
- **MAC 交错网格 + 几何多重网格压力投影（Geometric Multigrid V-Cycle）**：
  - 求解 2D Navier-Stokes 不可压缩气流，采用 8 层多重网格 + 最粗层共享内存求解器，实现全图全局质量守恒（`2048×2048` 网格水平线垂直通量不平衡率 `< 0.01%`，上升与下沉回流自然闭环）。
  - 集成涡量限制（Vorticity Confinement）与 MacCormack 二阶标量平流，呈现细腻的羽流与涡旋结构。
- **位温绝热过程与相变潜热**：
  - 自动接入原版每一层的环境温度（作为地层中段参考温度 `T_mid`）。
  - 气流上升绝热膨胀降温、下沉增温；水汽在 `> 0°C` 冷凝成云/雨、在 `≤ 0°C` 凝华成冰晶/雪花并真实释放潜热，推动更强的湿对流。
- **真实水循环与双向保水/防涝机制**：
  - **道尔顿蒸发定律**：水体内部根据风速与饱和差持续蒸发吸热并消耗水方块，云雨降落到地面形成真实积水（严格 1:1 质量守恒）。
  - **双向 ±2000 格护栏**：当蒸发消水比降雨成水多 2000 格时停止消水（水汽仍正常蒸发）；当降雨成水比蒸发消水多 2000 格时停止生成地面水方块（云雨粒子与淋湿效果仍正常），避免长期运行后地图干涸或被淹没。
- **动态热源实时同步**：
  - 自动感知地形热方块、岩浆，以及玩家实时放置/拆除/熄灭的篝火、火把、温控水晶，形成可见的局地热羽流与对流云雨。

### 2. 深度游戏玩法耦合 (`WeatherExpansionMod.dll`)
- **替换原版静态天气效果**：自动接管原版多雾、多雨、干燥、泛滥词条及第 6/7 层原生天气，转化为物理边界条件（如多雨层自动构建下暖湿、上冷的对流环境）。
- **体感温度与风力物理**：将局部模拟气温、风寒效应、雨雪淋湿耦合至角色体温系统，并对玩家、物品及周围刚体施加真实空气阻力风力。
- **视野友好视觉表现**：玩家周围自动柔化近身雾气、实心墙体内雾效减半，配合全屏均匀分布的随风飘动雨滴/雪花粒子。
- **零主线程阻塞架构**：CUDA 计算与视口读回全部运行在独立后台线程（`WeatherSimCudaWorker`），游戏帧率保持流畅。

### 3. 交互式全图气象雷达与调试工具
- **`F7` — 全图气象雷达 UI**：支持缩放与平移查看全图 **相对湿度 / 云 / 降水 / 降雪 / 温度（含 0°C 等温线） / 风速风向箭头 / 综合天气** 7 种图层，并支持鼠标悬停实时读取任意坐标的气象数据。
- **`F8` — 视口覆盖模式切换**：自然云雾 → 温度场 → 相对湿度 → 风速场 → 潜热场 → 关闭。
- **`F9` — 实时状态与 GPU 遥测 HUD**：显示当地温度、体感温度、湿度、风速、水循环净差值及 CUDA 耗时/显存。

---

## 性能参考（`1024×1024` 地图 → `2048×2048` 模拟网格，1 格 = 4 模拟格）

| 测试平台 | 网格分辨率 | 单步 GPU 耗时 | 显存占用 | 模拟频率 |
| :--- | :---: | :---: | :---: | :---: |
| NVIDIA RTX 5060 (8GB) | `2048 × 2048` (419 万格) | **~6.3 ms** | **~532 MB** | 30 Hz 异步 |

> 超大地图超出显存/算力预算时会自动回退至 `1×1` 分辨率或玩家周边动态窗口，确保稳定运行。

---

## 环境要求

- 游戏：**《未知伤亡》（Casualties: Unknown）**（Windows 64 位）
- Mod 加载器：**BepInEx 5.x**（x64）
- 显卡：**NVIDIA GPU（RTX 20 / 30 / 40 / 50 系列，Compute Capability ≥ 7.5）** 及较新显卡驱动
  > Release 版 `WeatherSimCuda.dll` 已静态链接 CUDA Runtime 并内含多架构代码（`sm_75 / 80 / 86 / 89 / 120 + PTX`），**玩家无需安装 CUDA Toolkit** 即可直接游玩。

---

## 安装指南

1. 确保游戏已正确安装 **BepInEx 5 (x64)**。
2. 从本仓库的 [Releases](../../releases) 页面下载最新的 `WeatherExpansionMod-vX.Y.Z-win64-cuda.zip`。
3. 将压缩包解压到游戏根目录，使目录结构如下：
   ```text
   <游戏根目录>/
   └── BepInEx/
       └── plugins/
           └── WeatherExpansionMod/
               ├── WeatherExpansionMod.dll
               └── WeatherSimCuda.dll
   ```
4. 启动游戏进入任意地层，左上角出现天气 HUD 即表示加载成功。

---

## 主要配置项

首次运行后会在 `BepInEx/config/com.casualties.weather.expansion.cfg` 生成配置文件，可随时按需微调：

| 配置节 | 参数名 | 默认值 | 说明 |
| :--- | :--- | :---: | :--- |
| `[Grid]` | `PreferredSubdivision` | `2` | 单轴细分倍率（`2` = 1 游戏格对应 `2×2=4` 模拟格） |
| `[Simulation]` | `SimHz` | `30` | 物理求解频率 (Hz) |
| `[Simulation]` | `SorIterations` | `18` | 多重网格迭代预算（`18` = 每步 2 次完整 V-Cycle） |
| `[Thermodynamics]` | `WallTempRelaxRate` | `0.01` | 地形方块与气体的温度弛豫速率 |
| `[Thermodynamics]` | `ThermalDiffusivity` | `0.01` | 气体/方块热扩散系数 |
| `[Hydrology]` | `EvapMassPerWaterTile` | `0.70` | 蒸发 1 格游戏水方块所需累计水汽质量 |
| `[Hydrology]` | `RainMassPerWaterTile` | `0.70` | 降雨生成 1 格游戏水方块所需累计降水质量 |
| `[Hydrology]` | `MaxNetEvapWaterTiles` | `2000` | 蒸发超降雨达到该格数后暂停消水（水汽仍产生） |
| `[Hydrology]` | `MaxNetRainWaterTiles` | `2000` | 降雨超蒸发达到该格数后暂停生成地面水方块 |
| `[Physics]` | `WindForceScale` | `0.35` | 风力对游戏刚体的推力倍率 |
| `[Debug]` | `ToggleFullMapKey` | `F7` | 全图气象雷达开关快捷键 |

---

## 从源码编译

### 所需开发工具
- **Visual Studio 2022 / Build Tools**（含 C++ 桌面开发工作负载）
- **NVIDIA CUDA Toolkit 12.x / 13.x**
- **CMake 3.24+**（推荐搭配 Ninja）
- **.NET SDK 8.0+**（目标框架 `netstandard2.1`）

### 编译步骤
1. 复制 [Directory.Build.props.example](Directory.Build.props.example) 为 `Directory.Build.props`（已加入 `.gitignore`），填写你本机的游戏目录 `<GameDir>`。
2. 如 MSVC / CUDA / CMake 不在默认系统 `PATH` 中，可复制 [tools/local_env.bat.example](tools/local_env.bat.example) 为 `tools/local_env.bat` 并填入对应路径。
3. 运行一键编译脚本：
   ```bat
   tools\build_all.bat
   ```
   编译产物将输出至 `dist\WeatherExpansionMod\`。
4. 打包多代显卡通用 GitHub Release 压缩包：
   ```bat
   tools\package_release.bat 1.0.3
   ```

### 独立 Python 物理沙盒 & 宣传片录制工具
无需启动游戏即可独立运行并可视化 CUDA 物理场（需要 `numpy`、`matplotlib`，录屏需 `ffmpeg`）：
```bash
# 实时交互预览（支持 snow / rainy 预设）
python tools/verify_weather_sandbox.py --preset snow --live

# 录制 16:9 高清演示视频
python tools/verify_weather_sandbox.py --preset rainy --record output/weather_demo.mp4
```

---

## 项目结构

```text
WeatherExpansionMod/
├── WeatherSimCuda/              # 原生 C++ / CUDA 2D 流体与大气热力学求解器
│   ├── include/weather_sim_api.h
│   ├── src/weather_kernels.cu   # Multigrid 压力投影、MacCormack、微物理、雷达渲染内核
│   ├── src/weather_sim_api.cpp
│   └── CMakeLists.txt
├── WeatherExpansionMod/         # BepInEx 5 C# 游戏集成层
│   ├── Native/                  # P/Invoke 原生接口绑定
│   ├── Patches/                 # Harmony 补丁（地层天气接管、体温耦合、方块变动同步）
│   └── Runtime/                 # 异步求解宿主、粒子渲染、F7 全图雷达、F9 遥测 HUD
└── tools/                       # 跨环境编译脚本、GPU 基准测试与 Python 可视化沙盒
```

---

## 更新日志与开源协议

- 完整版本迭代记录见 [CHANGELOG.md](CHANGELOG.md)。
- 欢迎通过 Issue / Pull Request 参与改进，详见 [CONTRIBUTING.md](CONTRIBUTING.md)。
- 本项目采用 [MIT License](LICENSE) 开源。
