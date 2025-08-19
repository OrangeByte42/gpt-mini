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
    LENGTH_RANGE: Tuple[int, int] = (64 - 2, 512 - 2)   # - 2 for SOS & EOS tokens
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


@dataclass
class ModelConfigs:
    """Configuration for model."""
    NUM_LAYERS: int = 12
    D_MODEL: int = 768
    NUM_HEADS: int = 12
    D_FF: int = 3072
    DROP_PROB: float = 0.3


@dataclass
class TrainingConfigs:
    """Configuration for training."""
    # Dataset parameters
    BATCH_SIZE: int = 6
    NUM_WORKERS: int = 4

    # Training parameters
    WEIGHT_DECAY: float = 1e-4
    EPOCHES_NUM: int = 100
    WARMUP: int = 8
    INIT_LR: float = 5e-5
    ADAM_EPS: float = 1e-8
    PATIENCE: int = 8
    FACTOR: float = 0.8
    CLIP: float = 0.7

    # Saving parameters
    CHECKPOINT_DIR: str = os.path.join(".", "outs", "checkpoints")
    TRAIN_TRACE_DIR: str = os.path.join(".", "outs", "train_trace")
    SAMPLE_TRACE_DIR: str = os.path.join(".", "outs", "sample_trace")

    # Create directories if they do not exist
    def __post_init__(self: Any) -> None:
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.TRAIN_TRACE_DIR, exist_ok=True)
        os.makedirs(self.SAMPLE_TRACE_DIR, exist_ok=True)



