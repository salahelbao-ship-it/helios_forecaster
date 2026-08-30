# utils/helpers.py
import os
import sys
import json
import random
import hashlib
import platform
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

def set_global_seed(seed: int = 42) -> None:
    """Enforces strict reproducibility across NumPy, Python, and TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        pass

def enforce_gpu_memory_growth() -> None:
    """Prevents TensorFlow from pre-allocating 100% of VRAM."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU Setup Error: {e}")

def get_code_hash() -> str:
    """Generates a hash of the current execution state for the artifact manifest."""
    try:
        with open(sys.argv[0], 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        return "unknown_code_hash"

def generate_environment_meta() -> dict:
    """Captures the hardware and software environment for reproducibility tracking."""
    gpus = tf.config.list_physical_devices('GPU')
    det_ops = os.environ.get('TF_DETERMINISTIC_OPS', '0') == '1'
    rep_level = "FULL" if det_ops and not gpus else ("PARTIAL" if det_ops else "NONE")
    
    try:
        import pywt
        has_pywt = True
    except ImportError:
        has_pywt = False

    try:
        import ngboost
        has_ngboost = True
    except ImportError:
        has_ngboost = False

    return {
        "python": sys.version,
        "os": platform.platform(),
        "cpu": platform.processor(),
        "gpu_available": len(gpus) > 0,
        "tf_version": tf.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "tf_deterministic": det_ops,
        "reproducibility_level": rep_level,
        "pywt_available": has_pywt,
        "ngboost_available": has_ngboost
    }

def setup_directories(base_path: str = ".") -> None:
    """Ensures all necessary I/O directories exist."""
    dirs = [
        "cache", "exports", "reports", "db", 
        "predictions", "exports/figures", "exports/production"
    ]
    for d in dirs:
        os.makedirs(os.path.join(base_path, d), exist_ok=True)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to safely serialize NumPy datatypes to JSON/SQLite."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            if np.isnan(obj) or np.isinf(obj): return None
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        return super().default(obj)