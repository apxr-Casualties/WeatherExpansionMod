import ctypes
import os
import numpy as np

CELL_AIR = 0
CELL_SOLID = 1
CELL_WATER = 2

FIELD_U = 0
FIELD_V = 1
FIELD_TEMP = 2
FIELD_QV = 3
FIELD_QC = 4
FIELD_QR = 5
FIELD_QI = 6
FIELD_SNOW_ACCUM = 7
FIELD_RH = 8
FIELD_DIVERGENCE = 9
FIELD_LATENT_RATE = 10
FIELD_EVAP_ACCUM = 11


class WeatherSimConfig(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("cellSizeTiles", ctypes.c_float),
        ("midLayerTempC", ctypes.c_float),
        ("bgLapseRateCPerTile", ctypes.c_float),
        ("adiabaticLapseCPerTile", ctypes.c_float),
        ("buoyancyScale", ctypes.c_float),
        ("viscosity", ctypes.c_float),
        ("thermalDiffusivity", ctypes.c_float),
        ("moistureDiffusivity", ctypes.c_float),
        ("vorticityEps", ctypes.c_float),
        ("velocityDamping", ctypes.c_float),
        ("maxSpeedTilesPerSec", ctypes.c_float),
        ("wallTempRelaxRate", ctypes.c_float),
        ("evapRateCoeff", ctypes.c_float),
        ("evapWindFactor", ctypes.c_float),
        ("evapCoolingScale", ctypes.c_float),
        ("latentHeatScale", ctypes.c_float),
        ("autoconvertThresh", ctypes.c_float),
        ("autoconvertRate", ctypes.c_float),
        ("freezeRate", ctypes.c_float),
        ("meltRate", ctypes.c_float),
        ("rainFallSpeed", ctypes.c_float),
        ("snowFallSpeed", ctypes.c_float),
        ("snowDepositRate", ctypes.c_float),
        ("sorIterations", ctypes.c_int32),
        ("sorOmega", ctypes.c_float),
        ("pressureWarmStart", ctypes.c_float),
        ("useMacCormack", ctypes.c_int32),
        ("initialRelHumidity", ctypes.c_float),
        ("bottomTempBoostC", ctypes.c_float),
        ("topTempDropC", ctypes.c_float),
        ("bottomMoistureSource", ctypes.c_float),
        ("ambientHumidityTarget", ctypes.c_float),
        ("ambientHumidityRate", ctypes.c_float),
        ("subdivision", ctypes.c_int32),
        ("evapMassPerWaterTile", ctypes.c_float),
        ("rainMassPerWaterTile", ctypes.c_float),
    ]


class WeatherFluidEvent(ctypes.Structure):
    _fields_ = [
        ("tileX", ctypes.c_int32),
        ("tileY", ctypes.c_int32),
        ("delta", ctypes.c_int32),
    ]


class WeatherSimStats(ctypes.Structure):
    _fields_ = [
        ("lastStepGpuMs", ctypes.c_float),
        ("allocatedVramBytes", ctypes.c_uint64),
        ("freeVramBytes", ctypes.c_uint64),
        ("totalVramBytes", ctypes.c_uint64),
        ("stepCount", ctypes.c_int32),
        ("lastCudaError", ctypes.c_int32),
    ]


class WeatherEntitySample(ctypes.Structure):
    _fields_ = [
        ("u", ctypes.c_float),
        ("v", ctypes.c_float),
        ("tempC", ctypes.c_float),
        ("relHumidity", ctypes.c_float),
        ("cloud", ctypes.c_float),
        ("rain", ctypes.c_float),
        ("snow", ctypes.c_float),
    ]


def load_dll(dll_path=None):
    if dll_path is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dll_path = os.path.join(root, "build_cuda", "bin", "WeatherSimCuda.dll")
    dll = ctypes.CDLL(dll_path)

    dll.WeatherSim_DefaultConfig.argtypes = [ctypes.POINTER(WeatherSimConfig), ctypes.c_int32, ctypes.c_int32]
    dll.WeatherSim_DefaultConfig.restype = None

    dll.WeatherSim_Create.argtypes = [ctypes.POINTER(WeatherSimConfig)]
    dll.WeatherSim_Create.restype = ctypes.c_void_p

    dll.WeatherSim_Destroy.argtypes = [ctypes.c_void_p]
    dll.WeatherSim_Destroy.restype = None

    dll.WeatherSim_UpdateConfig.argtypes = [ctypes.c_void_p, ctypes.POINTER(WeatherSimConfig)]
    dll.WeatherSim_UpdateConfig.restype = None

    dll.WeatherSim_SetMidLayerTemp.argtypes = [ctypes.c_void_p, ctypes.c_float]
    dll.WeatherSim_SetMidLayerTemp.restype = None

    dll.WeatherSim_ResetFields.argtypes = [ctypes.c_void_p]
    dll.WeatherSim_ResetFields.restype = ctypes.c_int32

    dll.WeatherSim_UploadCellsFull.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    dll.WeatherSim_UploadCellsFull.restype = ctypes.c_int32

    dll.WeatherSim_UploadCellsPatch.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32
    ]
    dll.WeatherSim_UploadCellsPatch.restype = ctypes.c_int32

    dll.WeatherSim_UploadHeatSources.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    dll.WeatherSim_UploadHeatSources.restype = ctypes.c_int32

    dll.WeatherSim_UploadWindSources.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    dll.WeatherSim_UploadWindSources.restype = ctypes.c_int32

    dll.WeatherSim_Step.argtypes = [ctypes.c_void_p, ctypes.c_float]
    dll.WeatherSim_Step.restype = ctypes.c_int32

    dll.WeatherSim_SamplePoints.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(WeatherEntitySample), ctypes.c_int32
    ]
    dll.WeatherSim_SamplePoints.restype = ctypes.c_int32

    dll.WeatherSim_CopyField.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
    dll.WeatherSim_CopyField.restype = ctypes.c_int32

    dll.WeatherSim_ReadViewWindowRGBA8.argtypes = [
        ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float,
        ctypes.c_void_p
    ]
    dll.WeatherSim_ReadViewWindowRGBA8.restype = ctypes.c_int32

    dll.WeatherSim_PopFluidEvents.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(WeatherFluidEvent), ctypes.c_int32
    ]
    dll.WeatherSim_PopFluidEvents.restype = ctypes.c_int32

    dll.WeatherSim_GetStats.argtypes = [ctypes.c_void_p, ctypes.POINTER(WeatherSimStats)]
    dll.WeatherSim_GetStats.restype = ctypes.c_int32

    return dll


class WeatherSim:
    def __init__(self, width, height, dll=None, **overrides):
        self.dll = dll or load_dll()
        self.width = int(width)
        self.height = int(height)
        self.cfg = WeatherSimConfig()
        self.dll.WeatherSim_DefaultConfig(ctypes.byref(self.cfg), self.width, self.height)
        for k, v in overrides.items():
            setattr(self.cfg, k, v)
        self.handle = self.dll.WeatherSim_Create(ctypes.byref(self.cfg))
        if not self.handle:
            raise RuntimeError(f"WeatherSim_Create failed for {width}x{height}")

    def close(self):
        if getattr(self, "handle", None):
            self.dll.WeatherSim_Destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def upload_cells(self, cells_u8: np.ndarray):
        arr = np.ascontiguousarray(cells_u8, dtype=np.uint8)
        assert arr.shape == (self.height, self.width)
        rc = self.dll.WeatherSim_UploadCellsFull(self.handle, arr.ctypes.data_as(ctypes.c_void_p))
        if rc != 0:
            raise RuntimeError(f"UploadCellsFull failed: {rc}")

    def upload_cells_patch(self, xs, ys, types):
        if len(xs) == 0:
            return
        ax = np.ascontiguousarray(xs, dtype=np.int32)
        ay = np.ascontiguousarray(ys, dtype=np.int32)
        at = np.ascontiguousarray(types, dtype=np.uint8)
        rc = self.dll.WeatherSim_UploadCellsPatch(
            self.handle,
            ax.ctypes.data_as(ctypes.c_void_p),
            ay.ctypes.data_as(ctypes.c_void_p),
            at.ctypes.data_as(ctypes.c_void_p),
            len(ax),
        )
        if rc != 0:
            raise RuntimeError(f"UploadCellsPatch failed: {rc}")

    def pop_fluid_events(self, max_events=2048):
        buf = (WeatherFluidEvent * max_events)()
        n = self.dll.WeatherSim_PopFluidEvents(self.handle, buf, max_events)
        if n < 0:
            raise RuntimeError(f"PopFluidEvents failed: {n}")
        return [(buf[i].tileX, buf[i].tileY, buf[i].delta) for i in range(n)]

    def upload_heat_sources(self, heat_f32: np.ndarray):
        arr = np.ascontiguousarray(heat_f32, dtype=np.float32)
        assert arr.shape == (self.height, self.width)
        rc = self.dll.WeatherSim_UploadHeatSources(self.handle, arr.ctypes.data_as(ctypes.c_void_p))
        if rc != 0:
            raise RuntimeError(f"UploadHeatSources failed: {rc}")

    def upload_wind_sources(self, fx_f32=None, fy_f32=None):
        px = None
        py = None
        if fx_f32 is not None:
            ax = np.ascontiguousarray(fx_f32, dtype=np.float32)
            assert ax.shape == (self.height, self.width)
            px = ax.ctypes.data_as(ctypes.c_void_p)
        if fy_f32 is not None:
            ay = np.ascontiguousarray(fy_f32, dtype=np.float32)
            assert ay.shape == (self.height, self.width)
            py = ay.ctypes.data_as(ctypes.c_void_p)
        rc = self.dll.WeatherSim_UploadWindSources(self.handle, px, py)
        if rc != 0:
            raise RuntimeError(f"UploadWindSources failed: {rc}")

    def reset_fields(self):
        rc = self.dll.WeatherSim_ResetFields(self.handle)
        if rc != 0:
            raise RuntimeError(f"ResetFields failed: {rc}")

    def step(self, dt: float):
        rc = self.dll.WeatherSim_Step(self.handle, float(dt))
        if rc != 0:
            raise RuntimeError(f"Step failed: {rc}")

    def copy_field(self, field_id: int) -> np.ndarray:
        out = np.empty((self.height, self.width), dtype=np.float32)
        rc = self.dll.WeatherSim_CopyField(self.handle, int(field_id), out.ctypes.data_as(ctypes.c_void_p))
        if rc != 0:
            raise RuntimeError(f"CopyField({field_id}) failed: {rc}")
        return out

    def sample_points(self, xs, ys):
        ax = np.ascontiguousarray(xs, dtype=np.float32)
        ay = np.ascontiguousarray(ys, dtype=np.float32)
        n = len(ax)
        out = (WeatherEntitySample * n)()
        rc = self.dll.WeatherSim_SamplePoints(
            self.handle,
            ax.ctypes.data_as(ctypes.c_void_p),
            ay.ctypes.data_as(ctypes.c_void_p),
            out,
            n,
        )
        if rc != 0:
            raise RuntimeError(f"SamplePoints failed: {rc}")
        return list(out)

    def read_view_rgba8(
        self, x0, y0, win_w, win_h, mode=0,
        player_cx=-1e6, player_cy=-1e6,
        clear_inner=0.0, clear_outer=0.0,
        min_fog_factor=0.18, wall_fog_mult=0.5
    ) -> np.ndarray:
        out = np.empty((win_h, win_w, 4), dtype=np.uint8)
        rc = self.dll.WeatherSim_ReadViewWindowRGBA8(
            self.handle, int(x0), int(y0), int(win_w), int(win_h), int(mode),
            float(player_cx), float(player_cy),
            float(clear_inner), float(clear_outer),
            float(min_fog_factor), float(wall_fog_mult),
            out.ctypes.data_as(ctypes.c_void_p)
        )
        if rc != 0:
            raise RuntimeError(f"ReadViewWindowRGBA8 failed: {rc}")
        return out

    def stats(self) -> WeatherSimStats:
        s = WeatherSimStats()
        self.dll.WeatherSim_GetStats(self.handle, ctypes.byref(s))
        return s
