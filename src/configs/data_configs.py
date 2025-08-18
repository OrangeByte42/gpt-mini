import os
from dataclasses import dataclass
from typing import Any, Tuple


@dataclass
class DatasetConfigs:
    """Configuration for datasets."""
    # Basic directories
    DATA_DIR: str = os.path.join(".", "data")
    DATASET_DIR: str = os.path.join(DATA_DIR, "datasets")

    # Dataset-specific directories
    ORIGINAL_CACHE_DIR: str = os.path.join(DATASET_DIR, "original_cache")
    PROCESSED_CACHE_DIR: str = os.path.join(DATASET_DIR, "processed_cache")
    TRAIN_CACHE_DIR: str = os.path.join(DATASET_DIR, "train_cache")
    TOKENIZED_CACHE_DIR: str = os.path.join(DATASET_DIR, "tokenized_cache")

    # Pre-processing parameters
    LENGTH_RANGE: Tuple[int, int] = (64, 512)
    MAX_LENGTH: int = 512

    # Length distribution showing and processing num of processor
    NUM_BINS: int = 20
    NUM_PROC: int = 20

    # Processing batch size and saving batch size
    PROCESS_BATCH_SIZE: int = 10_000
    SAVE_BATCH_SIZE: int = 1_000_000

    # Spliting parameters
    SEED: int = 42
    TRAIN_RATIO: float = 0.8
    VALID_RATIO: float = 0.2

    # Create directories if they do not exist
    def __post_init__(self: Any) -> None:
        os.makedirs(self.ORIGINAL_CACHE_DIR, exist_ok=True)
        os.makedirs(self.PROCESSED_CACHE_DIR, exist_ok=True)
        os.makedirs(self.TRAIN_CACHE_DIR, exist_ok=True)
        os.makedirs(self.TOKENIZED_CACHE_DIR, exist_ok=True)


@dataclass
class TokenizerConfigs:
    """Configuration for tokenizer."""
    # Basic directories
    DATA_DIR: str = os.path.join(".", "data")
    TOKENIZER_DIR: str = os.path.join(DATA_DIR, "tokenizers")

    # Tokenizer-specific directories
    TOKENIZER_NAME: str = "bpe"
    TOKENIZER_CACHE_DIR: str = os.path.join(TOKENIZER_DIR, TOKENIZER_NAME)
    MIN_FREQ: int = 0
    MAX_VOCAB_SIZE: int = 32_000

    # Create directories if they do not exist
    def __post_init__(self: Any) -> None:
        os.makedirs(self.TOKENIZER_CACHE_DIR, exist_ok=True)





