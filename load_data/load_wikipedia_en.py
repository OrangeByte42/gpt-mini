import os
from typing import Any

import datasets

from load_data.utils.configs import *
from load_data.utils.utils import calculate_length
from load_data.utils.utils import show_length_distribution
from load_data.utils.utils import split_by_paragraph
from load_data.utils.utils import save_as_parquets


import numpy as np
from typing import Any
from datasets import Dataset  # 假设你有这个导入


# Set local name
_LOCAL_NAME: str = "wikipedia_en"
_NUM_PROC: int = 16

# Download original dataset
_ORI_SAVE_DIR: str = os.path.join(ORI_DATASETS_DIR, _LOCAL_NAME)
os.makedirs(_ORI_SAVE_DIR, exist_ok=True)

wikipedia_en: Any = datasets.load_dataset("izumi-lab/wikipedia-en-20230720",
                                            cache_dir=_ORI_SAVE_DIR, num_proc=_NUM_PROC)
print(wikipedia_en)
from datasets import DatasetDict
wikipedia_en = DatasetDict({
    "train": wikipedia_en["train"].select(range(1_000))
})


wikipedia_en = wikipedia_en.map(calculate_length, batched=True, batch_size=10_000, num_proc=_NUM_PROC)
print(wikipedia_en)

show_length_distribution(wikipedia_en["train"]["length"])

splited_data = wikipedia_en.map(split_by_paragraph, batched=True, batch_size=10_000, num_proc=_NUM_PROC,
                                remove_columns=wikipedia_en["train"].column_names)
print(splited_data)

splited_data = splited_data.map(calculate_length, batched=True, batch_size=10_000, num_proc=_NUM_PROC)
print(splited_data)
show_length_distribution(splited_data["train"]["length"])

splited_data = splited_data.remove_columns(["length"])
print(splited_data)

splited_data = splited_data["train"]

_PREPROCESSED_SAVE_DIR: str = os.path.join(PREPROCESSED_DATASETS_DIR, _LOCAL_NAME)
os.makedirs(_PREPROCESSED_SAVE_DIR, exist_ok=True)
num_samples_per_file: int = 10_000_000
save_as_parquets(splited_data, num_samples_per_file, _LOCAL_NAME, _PREPROCESSED_SAVE_DIR)




