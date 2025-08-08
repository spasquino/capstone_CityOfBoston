"""
Global configuration for the Sara–Sasha Capstone project.
Edit these paths to match your environment or set env vars to override.
"""
import os
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(os.getenv("CAPSTONE_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR     = Path(os.getenv("CAPSTONE_DATA_DIR", PROJECT_ROOT / "data"))
FIGURES_DIR  = Path(os.getenv("CAPSTONE_FIGURES_DIR", PROJECT_ROOT / "figures"))
OUTPUT_DIR   = Path(os.getenv("CAPSTONE_OUTPUT_DIR", PROJECT_ROOT / "outputs"))

# Common CRS strings (Massachusetts uses 2249/26986 often)
CRS_WGS84 = "EPSG:4326"
CRS_MA_STATEPLANE = "EPSG:2249"
CRS_UTM19N = "EPSG:32619"

RANDOM_SEED = int(os.getenv("CAPSTONE_RANDOM_SEED", "42"))
