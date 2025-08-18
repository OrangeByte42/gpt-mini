import os
from typing import Any, List

import numpy as np
from datasets import DatasetDict, Dataset
from datasets import load_dataset, concatenate_datasets


def show_length_distribution(flatten_dataset: Dataset, column_name: str, num_bins: int,
                                process_batch_size: int, num_proc: int) -> None:
    """Show the length distribution of a specific column in the dataset.
    @param flatten_dataset: The dataset to analyze.
    @param column_name: The name of the column to analyze.
    @param num_bins: The number of bins for the histogram.
    @param process_batch_size: The batch size for processing the dataset.
    @param num_proc: The number of processes to use for parallel computation.
    """

    def _calculate_length4map(batch: Any, column_name: str) -> List[int]:
        """Calculate the length of the specified column in each example.
        @param batch: A batch of data from the dataset.
        @param column_name: The name of the column to analyze.
        @return: A list of lengths for the specified column.
        """
        lengths: List[int] = list(map(len, batch[column_name]))
        return {'length': lengths}

    def _show_length_distribution(lengths: np.ndarray, num_bins: int) -> None:
        """Show the length distribution as a histogram.
        @param lengths: An array of lengths.
        @param num_bins: The number of bins for the histogram.
        """
        # Calculate histogram
        min_length, max_length = lengths.min(), lengths.max()
        bins: np.ndarray = np.linspace(min_length, max_length, num_bins + 1).astype(np.int64)
        hist, _ = np.histogram(lengths, bins=bins)

        # Show length distribution
        total_count: int = len(lengths)
        print(f"Length distribution (total count: {total_count}):")

        for idx in range(num_bins):
            count: int = hist[idx]
            percentage: float = (count / total_count) * 100
            print(f"Length: {bins[idx]:,} - {bins[idx + 1]:,}, Counts: {count:,}, Percentage: {percentage:.6f}%")

    # Calculate lengths for the specified column
    lengths: Dataset = flatten_dataset.map(
        function=_calculate_length4map,
        fn_kwargs={'column_name': column_name},
        batched=True,
        batch_size=process_batch_size,
        num_proc=num_proc,
        remove_columns=flatten_dataset.column_names
    )

    # Show the length distribution
    _show_length_distribution(np.array(lengths['length']), num_bins=num_bins)
    print(flatten_dataset)

def save_as_parquets(dataset_dict: DatasetDict, save_dir: str, save_batch_size: int) -> None:
    """Save the dataset as Parquet files.
    @param dataset_dict: The dataset to save.
    @param save_dir: The directory where the Parquet files will be saved.
    @param save_batch_size: The batch size for saving the dataset.
    """
    # Iterate through each split in the dataset
    for split_name, dataset in dataset_dict.items():
        # Create the directory for the current split
        split_save_dir: str = os.path.join(save_dir, split_name)
        os.makedirs(split_save_dir, exist_ok=True)

        # Split the dataset into shards and save each shard as a Parquet file
        num_shards: int = (len(dataset) + save_batch_size - 1) // save_batch_size
        for shard_idx in range(num_shards):
            shard: Dataset = dataset.shard(num_shards=num_shards, index=shard_idx)
            shard_path: str = os.path.join(split_save_dir, f"{split_name}_shard_{shard_idx}.parquet")
            shard.to_parquet(shard_path)
            print(f"Saved shard {shard_idx + 1}/{num_shards} to {shard_path}")

def load_parquets(save_dir: str, streaming: bool) -> DatasetDict:
    """Save the dataset as Parquet files.
    @param save_dir: The directory where the Parquet files are saved.
    @param streaming: Whether to load the dataset in streaming mode.
    @return: A DatasetDict containing the loaded datasets.
    """
    # Create a DatasetDict to hold the loaded datasets
    dataset_dict: DatasetDict = DatasetDict()

    # Iterate through each split in the save directory
    for split_name in os.listdir(save_dir):
        split_dir: str = os.path.join(save_dir, split_name)
        parquet_files: List[str] = [os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith('.parquet')]
        if parquet_files:
            dataset_dict[split_name] = load_dataset("parquet", data_files=parquet_files, streaming=streaming)["train"]

    # Return the DatasetDict containing the loaded datasets
    return dataset_dict

