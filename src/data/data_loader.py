import os

from datasets import DatasetDict
from datasets import concatenate_datasets

from src.configs import DatasetsConfigs
from src.data.utils.utils import load_parquets


configs: DatasetsConfigs = DatasetsConfigs()

combined_dataset: DatasetDict = load_parquets(configs.COMBINED_DATASETS_DIR, streaming=True)
print(combined_dataset)




