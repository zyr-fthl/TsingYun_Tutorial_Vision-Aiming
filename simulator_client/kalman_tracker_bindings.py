"""ctypes bindings for the C++ Kalman tracker (task3)."""

from __future__ import annotations

import torch
import ctypes
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _library_path() -> Path:
    env_dir = os.environ.get("TSINGYUN_HW_BUILD_DIR")

    if sys.platform == "darwin":
        build_dir = Path(env_dir) if env_dir else REPO_ROOT / "build" / "hw"
        return build_dir / "tasks" / "task3-tracker" / "libhw_task3_tracker_shared.dylib"

    if sys.platform == "win32":
        build_dir = Path(env_dir) if env_dir else REPO_ROOT / "build" / "hw-ninja"
        return build_dir / "tasks" / "task3-tracker" / "libhw_task3_tracker_shared.dll"

    build_dir = Path(env_dir) if env_dir else REPO_ROOT / "build" / "hw"
    return build_dir / "tasks" / "task3-tracker" / "libhw_task3_tracker_shared.so"


def _resolve_library_path() -> Path:
    candidate = _library_path()
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        "Could not find the task3 shared library. Build the C++ targets first.\n"
        f"Expected:\n{candidate}"
    )


_LIB_PATH = _resolve_library_path()

if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    dll_dirs = {_LIB_PATH.parent}
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    for entry in path_entries:
        if entry:
            dll_dirs.add(Path(entry))
    for dll_dir in dll_dirs:
        try:
            dir_exists = dll_dir.exists()
        except PermissionError:
            dir_exists = False
        if dir_exists:
            os.add_dll_directory(str(dll_dir))

try:
    _lib = ctypes.CDLL(str(_LIB_PATH))
    USE_CPP = True
except Exception:
    USE_CPP = False
    import numpy as np

# --- function signatures ---
if USE_CPP and _lib is not None:
    _lib.tracker_create.restype = ctypes.c_void_p
    _lib.tracker_create_with_params.argtypes = [ctypes.c_double, ctypes.c_double]
    _lib.tracker_create_with_params.restype = ctypes.c_void_p
    _lib.tracker_destroy.argtypes = [ctypes.c_void_p]
    _lib.tracker_is_tracking.argtypes = [ctypes.c_void_p]
    _lib.tracker_is_tracking.restype = ctypes.c_int
    _lib.tracker_reset.argtypes = [ctypes.c_void_p]
    _lib.tracker_get_position.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ]
    _lib.tracker_update.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ]
    _lib.tracker_predict.argtypes = [
        ctypes.c_void_p, ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ]
    _lib.tracker_last_error.restype = ctypes.c_char_p

def _check_tracker_error() -> None:
    msg = _lib.tracker_last_error()
    if msg is not None:
        error_str = msg.decode("utf-8", errors="replace") if isinstance(msg, bytes) else str(msg)
        if "NotImplementedError" in error_str:
            raise NotImplementedError(error_str)
        raise RuntimeError(error_str)


class KalmanTracker:
    """Python wrapper around the C++ KalmanTracker from task3 (With Python Fallback)."""

    def __init__(self, process_noise: float = 0.05, measurement_noise: float = 10.0) -> None:
        self.q = process_noise
        self.r = measurement_noise
        
        if USE_CPP:
            self._ptr = _lib.tracker_create_with_params(process_noise, measurement_noise)
        else:
            self.initialized = False
            self.last_position = np.zeros(3, dtype=np.float32)
            self.velocity = np.zeros(3, dtype=np.float32)
            self._is_tracking = False

    def __del__(self) -> None:
        if USE_CPP:
            if hasattr(self, '_ptr') and self._ptr is not None:
                _lib.tracker_destroy(self._ptr)

    @property
    def is_tracking(self) -> bool:
        if USE_CPP:
            result = bool(_lib.tracker_is_tracking(self._ptr))
            return result
        else:
            return self._is_tracking

    def reset(self) -> None:
        if USE_CPP:
            _lib.tracker_reset(self._ptr)
        else:
            self.initialized = False
            self._is_tracking = False
            self.velocity = np.zeros(3, dtype=np.float32)

    def get_position(self) -> tuple[float, float, float]:
        if USE_CPP:
            ox, oy, oz = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
            _lib.tracker_get_position(self._ptr, ctypes.byref(ox), ctypes.byref(oy), ctypes.byref(oz))
            _check_tracker_error()
            return (ox.value, oy.value, oz.value)
        else:
            return (float(self.last_position[0]), float(self.last_position[1]), float(self.last_position[2]))

    def update(self, x: float, y: float, z: float, dt: float) -> tuple[float, float, float]:
        if USE_CPP:
            ox, oy, oz = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
            _lib.tracker_update(self._ptr, x, y, z, dt,
                                ctypes.byref(ox), ctypes.byref(oy), ctypes.byref(oz))
            _check_tracker_error()
            return (ox.value, oy.value, oz.value)
        else:
            current_pos = np.array([x, y, z], dtype=np.float32)
            if not self.initialized:
                self.last_position = current_pos
                self.initialized = True
                self._is_tracking = True
                return (float(x), float(y), float(z))
                
            dt = max(0.001, dt)
            measured_velocity = (current_pos - self.last_position) / dt
            
            alpha = self.q / (self.q + self.r) if (self.q + self.r) > 0 else 0.1
            self.velocity = (1.0 - alpha) * self.velocity + alpha * measured_velocity
            self.last_position = current_pos
            self._is_tracking = True
            return (float(x), float(y), float(z))

    def predict(self, dt: float) -> tuple[float, float, float]:
        if USE_CPP:
            ox, oy, oz = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
            _lib.tracker_predict(self._ptr, dt, ctypes.byref(ox), ctypes.byref(oy), ctypes.byref(oz))
            _check_tracker_error()
            return (ox.value, oy.value, oz.value)
        else:
            if not getattr(self, 'initialized', False):
                return (0.0, 0.0, 0.0)
            predicted_pos = self.last_position + self.velocity * dt
            return (float(predicted_pos[0]), float(predicted_pos[1]), float(predicted_pos[2]))