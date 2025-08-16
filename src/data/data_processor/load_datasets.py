import datasets
from datasets import Dataset
from typing import Tuple

from src.configs import DatasetsConfigs
from src.data.data_processor.data_processor import DatasetPreprocessor
from src.data.data_processor.data_processor import DatasetsCombiner


if __name__ == "__main__":
    # Initialize global constants
    _DATASETS_CONFIGS: DatasetsConfigs = DatasetsConfigs()
    _NUM_PROC: int = 16
    _LENGTH_RANGE: Tuple[int, int] = (64, 512)
    _NUM_BINS: int = 20
    _BATCH_SIZE: int = 10_000
    _NUM_SAMPLES_PER_FILE: int = 50_000_000

    # Download and pre-process datasets
    ## wikipedia_en dataset
    wikipedia_en: DatasetPreprocessor = DatasetPreprocessor(dataset_name="wikipedia_en",
                                                            configs=_DATASETS_CONFIGS, num_proc=_NUM_PROC)
    dataset: Dataset = datasets.load_dataset("izumi-lab/wikipedia-en-20230720",
                                                cache_dir=wikipedia_en._ori_save_dir, num_proc=_NUM_PROC)
    dataset = dataset["train"].select_columns(["text"])
    wikipedia_en.load_ori_dataset(dataset)
    wikipedia_en.run_pipeline(length_range=_LENGTH_RANGE, num_bins=_NUM_BINS, batch_size=_BATCH_SIZE,
                                num_samples_per_file=_NUM_SAMPLES_PER_FILE)

    ## open_text_books dataset
    open_text_books: DatasetPreprocessor = DatasetPreprocessor(dataset_name="open_text_books",
                                                                configs=_DATASETS_CONFIGS, num_proc=_NUM_PROC)
    dataset: Dataset = datasets.load_dataset("izumi-lab/open-text-books",
                                                cache_dir=open_text_books._ori_save_dir, num_proc=_NUM_PROC)
    dataset = dataset["train"].select_columns(["text"])
    open_text_books.load_ori_dataset(dataset)
    open_text_books.run_pipeline(length_range=_LENGTH_RANGE, num_bins=_NUM_BINS, batch_size=_BATCH_SIZE,
                                    num_samples_per_file=_NUM_SAMPLES_PER_FILE)

    # ## c4 subset dataset
    # c4_subset: DatasetPreprocessor = DatasetPreprocessor(dataset_name="c4_subset",
    #                                                     configs=_DATASETS_CONFIGS, num_proc=_NUM_PROC)
    # dataset: Dataset = datasets.load_dataset("allenai/c4", "en",
    #                                             cache_dir=c4_subset._ori_save_dir, num_proc=_NUM_PROC)
    # dataset = dataset["train"].select_columns(["text"])
    # c4_subset.load_ori_dataset(dataset)
    # c4_subset.run_pipeline(length_range=_LENGTH_RANGE, num_bins=_NUM_BINS, batch_size=_BATCH_SIZE,
    #                         num_samples_per_file=_NUM_SAMPLES_PER_FILE)

    # Combine datasets
    combiner: DatasetsCombiner = DatasetsCombiner(configs=_DATASETS_CONFIGS, num_proc=_NUM_PROC)
    combiner.run_pipeline(datasets_paths=[
        wikipedia_en.save_dir,
        open_text_books.save_dir,
    ], num_bins=_NUM_BINS, batch_size=_BATCH_SIZE, num_samples_per_file=_NUM_SAMPLES_PER_FILE)

