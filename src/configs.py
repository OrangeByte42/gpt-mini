import os
from dataclasses import dataclass


@dataclass
class DataConfig:
    """Configuration for data parameters."""
    DATASET_NAME: str = ...
    TOKENIZER_NAME: str = ...
    BATCH_SIZE: int = ...
    MAX_SEQ_LEN: int = ...
    DATASET_CACHE_DIR: str = ...
    TOKENIZER_CACHE_DIR: str = ...


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    NUM_LAYERS: int = 12
    D_MODEL: int = 768
    NUM_HEADS: int = 12
    D_FF: int = 3072
    DROP_PROB: float = 0.1


@dataclass
class TrainConfig:
    """Configuration for training parameters."""
    WEIGHT_DECAY: float = 1e-4
    EPOCHS_NUM: int = 100
    WARMUP: int = 8
    INIT_LR: float = 5e-5
    ADAM_LR: float = 1e-8
    PATIENCE: int = 8
    FACTOR: float = 0.8
    CLIP: float = 0.7


@dataclass
class SaveConfig:
    """Configuration for saving."""
    CHECKPOINT_DIR: str = os.path.join(".", "outs", "checkpoints")
    TRAIN_TRACE_DIR: str = os.path.join(".", "outs", "train_trace")

