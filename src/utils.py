import logging
from pathlib import Path
import geopandas as gpd

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=level,
    )
    return logging.getLogger("capstone")

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def read_gdf(path: Path, **kwargs) -> gpd.GeoDataFrame:
    return gpd.read_file(path, **kwargs)

def to_gdf(df, geometry=None, crs=None):
    return gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
