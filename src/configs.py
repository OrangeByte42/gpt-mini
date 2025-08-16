import os
from dataclasses import dataclass
from typing import Any


@dataclass
class DatasetsConfigs:
    """Configuration for datasets."""
    # Basic directories
    DATA_DIR: str = os.path.join(".", "data")
    DATASETS_DIR: str = os.path.join(DATA_DIR, "datasets")

    # Download and pre-process directories
    ORI_DATASETS_DIR: str = os.path.join(DATASETS_DIR, "original")
    PREPROCESSED_DATASETS_DIR: str = os.path.join(DATASETS_DIR, "preprocessed")
    COMBINED_DATASETS_DIR: str = os.path.join(DATASETS_DIR, "combined")

    # Create directories if they do not exist
    def __post_init__(self: Any) -> None:
        os.makedirs(self.ORI_DATASETS_DIR, exist_ok=True)
        os.makedirs(self.PREPROCESSED_DATASETS_DIR, exist_ok=True)
        os.makedirs(self.COMBINED_DATASETS_DIR, exist_ok=True)

