import random
from logging import getLogger
from os import environ as env
from pathlib import Path

import numpy as np


def str2bool(s: str) -> bool | None:
    true_vals, false_vals = ("true", "t", "1"), ("false", "f", "0")
    if s.lower() in true_vals:
        return True
    elif s.lower() in false_vals:
        return False
    else:
        return None


def env2bool(name: str, default: str) -> bool:
    var_val = env.get(name, default)
    val = str2bool(var_val)
    if val is None:
        getLogger(__name__).warning(
            f"Unable to match value {var_val} of env var {name} to a boolean. "
            f"Fallback to False."
        )
        val = False
    return val


CACHE_DIR = Path(env.get("XDG_CACHE_HOME", "~/.cache")).expanduser()
HOME = Path(env.get("EMBREG_HOME", "~/.embreg")).expanduser()
CACHE = Path(env.get("EMBREG_CACHE", CACHE_DIR / "embreg")).expanduser()
REPRODUCIBLE = str2bool(env.get("EMBREG_REPRODUCIBLE", "True"))


rng = random.Random()
rng.seed(42 if REPRODUCIBLE else None)
np_rng = np.random.default_rng(rng.getstate()[1][0])


HOME.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)
