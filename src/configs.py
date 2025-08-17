import os
from dataclasses import dataclass
from typing import Any, Tuple


@dataclass
class DatasetsConfigs:
    """Configuration for datasets."""
    # Basic directories
    DATA_DIR: str = os.path.join(".", "data")
    DATASETS_DIR: str = os.path.join(DATA_DIR, "datasets")

    # Tokenizer configuration
    TOKENIZER_NAME: str = "byte_level_bpe"
    TOKENIZER_CACHE_DIR: str = os.path.join(DATA_DIR, "tokenizer")
    MAX_VOCAB_SIZE: int = 32_000

    # Download and pre-process directories
    ORI_DATASETS_DIR: str = os.path.join(DATASETS_DIR, "original")
    PREPROCESSED_DATASETS_DIR: str = os.path.join(DATASETS_DIR, "preprocessed")
    COMBINED_DATASETS_DIR: str = os.path.join(DATASETS_DIR, "combined")

    # Training configuration
    TRAIN_RATIO: float = 0.8
    VALIDATION_RATIO: float = 0.2

    BATCH_SIZE: int = 16
    MAX_LENGTH: int = 512

    # Create directories if they do not exist
    def __post_init__(self: Any) -> None:
        os.makedirs(self.ORI_DATASETS_DIR, exist_ok=True)
        os.makedirs(self.PREPROCESSED_DATASETS_DIR, exist_ok=True)
        os.makedirs(self.COMBINED_DATASETS_DIR, exist_ok=True)
        os.makedirs(self.TOKENIZER_CACHE_DIR, exist_ok=True)



