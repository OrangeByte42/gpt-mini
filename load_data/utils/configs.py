import os


# Global configuration for loading data
DATA_DIR: str = os.path.join(".", "data")
DATASETS_DIR: str = os.path.join(DATA_DIR, "datasets")
ORI_DATASETS_DIR: str = os.path.join(DATASETS_DIR, "original")
PREPROCESSED_DATASETS_DIR: str = os.path.join(DATASETS_DIR, "preprocessed")

os.makedirs(ORI_DATASETS_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_DATASETS_DIR, exist_ok=True)

