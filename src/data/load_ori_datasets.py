from typing import Tuple

from src.configs import DatasetsConfigs

from src.data.utils.utils import run_ori_dataset_loading_pipeline
from src.data.utils.utils import run_combine_datasets_pipeline


configs: DatasetsConfigs = DatasetsConfigs()
_LENGTH_RANGE: Tuple[int, int] = (64, 512)

_NUM_BINS: int = 20

_PROCESS_BATCH_SIZE: int = 10_000
_SAVE_BATCH_SIZE: int = 10_000_000

_NUM_PROC: int = 20


wikipedia_en: str = run_ori_dataset_loading_pipeline(
    configs=configs,
    hf_ds_addr="izumi-lab/wikipedia-en-20230720",
    local_name="wikipedia_en",
    column_name="text",
    num_bins=_NUM_BINS,
    length_range=_LENGTH_RANGE,
    process_batch_size=_PROCESS_BATCH_SIZE,
    save_batch_size=_SAVE_BATCH_SIZE,
    num_proc=_NUM_PROC
)

open_text_books: str = run_ori_dataset_loading_pipeline(
    configs=configs,
    hf_ds_addr="izumi-lab/open-text-books",
    local_name="open_text_books",
    column_name="text",
    num_bins=_NUM_BINS,
    length_range=_LENGTH_RANGE,
    process_batch_size=_PROCESS_BATCH_SIZE,
    save_batch_size=_SAVE_BATCH_SIZE,
    num_proc=_NUM_PROC
)

run_combine_datasets_pipeline(
    configs=configs,
    dataset_dirs=[
        wikipedia_en,
        open_text_books,
    ],
    num_bins=_NUM_BINS,
    num_proc=_NUM_PROC,
    save_batch_size=_SAVE_BATCH_SIZE,
)



