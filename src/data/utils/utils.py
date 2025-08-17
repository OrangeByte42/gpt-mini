import os
from typing import Any, List, Tuple

import numpy as np
from datasets import DatasetDict, Dataset
from datasets import load_dataset, concatenate_datasets

from src.configs import DatasetsConfigs


def show_length_distribution(flatten_dataset: Dataset, column_name: str = "text",
                                num_bins: int = 20, num_proc: int = 20) -> None:

    def _calculate_length4map(batch: Any, column_name: str) -> Any:
        lengths: List[int] = list(map(len, batch[column_name]))
        return {'length': lengths}

    def _show_length_distribution(lengths: np.ndarray, num_bins: int = 20) -> None:
        total_count: int = len(lengths)

        min_length, max_length = lengths.min(), lengths.max()
        bins: np.ndarray = np.linspace(min_length, max_length, num_bins + 1).astype(np.int64)
        hist, _ = np.histogram(lengths, bins=bins)

        print(f"Length distribution (total count: {total_count}):")
        for idx in range(num_bins):
            count: int = hist[idx]
            percentage: float = (count / total_count) * 100
            print(f"Lengths: {bins[idx]:,} - {bins[idx + 1]:,}, Counts: {count:,}, Ratio: {percentage:.6f}%)")

    lengths: Dataset = flatten_dataset.map(
        function=_calculate_length4map,
        fn_kwargs={'column_name': column_name},
        batched=True,
        batch_size=10_000,
        num_proc=num_proc,
        remove_columns=flatten_dataset.column_names
    )

    _show_length_distribution(lengths=np.array(lengths['length']), num_bins=num_bins)
    print(flatten_dataset)

def split_long_texts(flatten_dataset: Dataset, column_name: str = "text",
                        length_range: Tuple[int, int] = (64, 512),
                        batch_size: int = 10_000, num_proc: int = 20) -> Dataset:

    def _split_long_text4map(batch: Any, column_name: str, length_range: Tuple[int, int]) -> Any:
        min_length, max_length = length_range
        splited_texts: List[str] = []

        lengths: List[int] = list(map(len, batch[column_name]))
        for length, text in zip(lengths, batch[column_name]):
            if length < min_length: continue
            elif length <= max_length: splited_texts.append(text)
            else:
                parts: List[str] = text.split("\n")
                parts = list(filter(lambda x: min_length <= len(x) < max_length, parts))
                splited_texts.extend(parts)

        return {'text': splited_texts}

    processed_dataset: Dataset = flatten_dataset.map(
        function=_split_long_text4map,
        fn_kwargs={'column_name': column_name, 'length_range': length_range},
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=flatten_dataset.column_names
    )

    return processed_dataset

def save_as_parquets(dataset_dict: DatasetDict,
                        save_dir: str, num_samples_per_files: int) -> None:
    for split, dataset in dataset_dict.items():
        split_dir: str = os.path.join(save_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        num_shards: int = (len(dataset) + num_samples_per_files - 1) // num_samples_per_files
        for shard_idx in range(num_shards):
            shard: Dataset = dataset.shard(num_shards=num_shards, index=shard_idx)
            shard_path: str = os.path.join(split_dir, f"{split}_shard_{shard_idx}.parquet")
            shard.to_parquet(shard_path)
            print(f"Saved shard {shard_idx + 1}/{num_shards} to {shard_path}")

def load_parquets(save_dir: str, streaming: bool = False) -> DatasetDict:
    dataset_dict: DatasetDict = DatasetDict()

    for split in os.listdir(save_dir):
        split_dir: str = os.path.join(save_dir, split)
        if not os.path.isdir(split_dir): continue
        parquet_files: List[str] = [os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith('.parquet')]
        if parquet_files:
            dataset_dict[split] = load_dataset(
                "parquet", data_files=parquet_files,
                streaming=streaming
            )

    return dataset_dict

def run_ori_dataset_loading_pipeline(configs: DatasetsConfigs, hf_ds_addr: str, local_name: str,
                                column_name: str, num_bins: int,
                                length_range: Tuple[int, int], process_batch_size: int,
                                save_batch_size: int, num_proc: int) -> str:

    print("Starting dataset loading pipeline...")
    print(f"================================================")

    # Initialize necessary directories
    ori_save_dir: str = os.path.join(configs.ORI_DATASETS_DIR, local_name)
    processed_save_dir: str = os.path.join(configs.PREPROCESSED_DATASETS_DIR, local_name)

    os.makedirs(ori_save_dir, exist_ok=True)
    os.makedirs(processed_save_dir, exist_ok=True)

    # Download the original datasets
    print(f"Loading dataset...")
    print(f"================================================")
    dataset_dict: DatasetDict = load_dataset(
        path=hf_ds_addr,
        cache_dir=ori_save_dir,
        num_proc=num_proc,
    )
    print(f"")

    # Flatten the dataset
    print(f"Flattening dataset...")
    print(f"================================================")
    flatten_dataset: Dataset = concatenate_datasets(dsets=dataset_dict.values())
    print(f"")

    # Show original length distribution
    print(f"Showing original length distribution...")
    print(f"================================================")
    show_length_distribution(
        flatten_dataset=flatten_dataset,
        column_name=column_name,
        num_bins=num_bins,
        num_proc=num_proc
    )

    # Try to load the dataset from the cache
    print(f"Trying to load dataset from cache...")
    print(f"================================================")
    try:
        dataset_dict: DatasetDict = load_parquets(save_dir=processed_save_dir, streaming=False)
        if len(dataset_dict) == 0: raise ValueError("No dataset found in cache.")
    except Exception as e:
        print(f"Failed to load dataset from cache: {e}")
        print(f"Proceeding with processing the dataset...")
        print(f"")
    else:
        print(f"Dataset loaded from cache successfully!")
        print(f"Showing processed length distribution...")
        print(f"================================================")
        flatten_dataset: Dataset = concatenate_datasets(dsets=dataset_dict.values())
        show_length_distribution(
            flatten_dataset=flatten_dataset,
            column_name=column_name,
            num_bins=num_bins,
            num_proc=num_proc
        )
        print(f"")
        return processed_save_dir

    # Pre-process the dataset
    print(f"Processing dataset...")
    print(f"================================================")
    flatten_dataset = split_long_texts(
        flatten_dataset=flatten_dataset,
        column_name=column_name,
        length_range=length_range,
        batch_size=process_batch_size,
        num_proc=num_proc
    )
    print(f"")

    # Show processed length distribution
    print(f"Showing processed length distribution...")
    print(f"================================================")
    show_length_distribution(
        flatten_dataset=flatten_dataset,
        column_name=column_name,
        num_bins=num_bins,
        num_proc=num_proc
    )
    print(f"")

    # Save the processed dataset as parquet files
    print(f"Saving processed dataset...")
    print(f"================================================")
    dataset_dict = DatasetDict({"train": flatten_dataset})

    save_as_parquets(
        dataset_dict=dataset_dict,
        save_dir=processed_save_dir,
        num_samples_per_files=save_batch_size
    )
    print(f"")

def run_combine_datasets_pipeline(configs: DatasetsConfigs, dataset_dirs: List[str],
                                    num_bins: int, num_proc: int,
                                    save_batch_size: int) -> None:

    print("Starting dataset combining pipeline...")
    print(f"================================================")

    # Initialize necessary directories
    combined_save_dir: str = configs.COMBINED_DATASETS_DIR
    os.makedirs(combined_save_dir, exist_ok=True)
    print(f"")

    # Combine datasets
    print(f"Combining datasets...")
    print(f"================================================")
    processed_datasets: List[Dataset] = []

    for dataset_dir in dataset_dirs:
        ori_dataset: Any = load_parquets(save_dir=dataset_dir)
        ori_dataset = concatenate_datasets(dsets=ori_dataset.values())
        processed_datasets.append(ori_dataset)

    combined_dataset: Dataset = concatenate_datasets(dsets=processed_datasets)
    print(f"")

    # Show combined length distribution
    show_length_distribution(
        flatten_dataset=combined_dataset,
        column_name="text",
        num_bins=num_bins,
        num_proc=num_proc
    )

    # Split the combined dataset into train and test sets
    print(f"Splitting combined dataset into train and test sets...")
    print(f"================================================")
    dataset_dict: DatasetDict = combined_dataset.train_test_split(
        train_size=configs.TRAIN_RATIO,
        test_size=configs.VALIDATION_RATIO,
        seed=42
    )
    print(f"")

    # Save the combined dataset
    print(f"Saving combined dataset...")
    print(f"================================================")

    save_as_parquets(
        dataset_dict=dataset_dict,
        save_dir=combined_save_dir,
        num_samples_per_files=save_batch_size
    )
    print(f"")

