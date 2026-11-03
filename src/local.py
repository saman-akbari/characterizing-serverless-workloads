import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = PROJECT_ROOT / "datasets"
DEFAULT_PROCESSED = DEFAULT_RAW / "processed"

raw_data_path = os.environ.get("RAW_DATA_PATH", DEFAULT_RAW)
data_path = os.environ.get("DATA_PATH", DEFAULT_PROCESSED)

raw_data_path = str(Path(raw_data_path).expanduser())
data_path = str(Path(data_path).expanduser())

__all__ = ["data_path", "raw_data_path", "PROJECT_ROOT"]
