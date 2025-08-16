import os
from typing import Any, Optional, Iterable, List, Tuple

from datasets import Dataset
from datasets import load_dataset, concatenate_datasets
import numpy as np

from src.configs import DatasetsConfigs


class DatasetProcessor:
    """A base class for data processors."""

    def __init__(self: Any, configs: DatasetsConfigs, num_proc: int = 16) -> None:
        """Initialize the DatasetProcessor.
        @param configs: Configuration for datasets
        @param num_proc: Number of processes to use for parallel processing
        """
        # Configuration attributes
        self.configs: DatasetsConfigs = configs
        self.num_proc: int = num_proc

        # Dataset attributes
        self.dataset_name: Optional[str] = None
        self.dataset: Optional[Dataset] = None

        # Saving attributes
        self._save_dir: Optional[str] = None

    def _create_save_dir(self: Any, save_dir: str) -> str:
        """Create specified directory if it does not exist.
        @param save_dir: Directory to create
        @return: The path to the created directory
        """
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def _calculate_length(self: Any, batch: Any) -> Any:
        """Calculate the length of each text in the batch.
        @param batch: A batch of data
        @return: The batch with an additional 'length' field
        """
        lengths: List[int] = [len(text) for text in batch['text']]
        batch['length'] = lengths
        return batch

    def _show_length_distribution(self: Any, lengths: Iterable[int], num_bins: int) -> None:
        """Display the distribution of text lengths in the dataset.
        @param lengths: Iterable of text lengths
        @param num_bins: Number of bins for the histogram
        """
        # Convert lengths to a numpy array
        lengths: np.ndarray = np.array(lengths)
        total_count: int = len(lengths)

        # Calculate the histogram
        min_length, max_length = lengths.min(), lengths.max()
        bins: np.ndarray = np.linspace(min_length, max_length, num_bins + 1)
        bins = bins.astype(np.int64)
        hist, _ = np.histogram(lengths, bins=bins)

        # Print the histogram
        for idx in range(num_bins):
            count: int = hist[idx]
            percentage: float = (count / total_count) * 100
            print(f"{bins[idx]:,} - {bins[idx + 1]:,}: ({count:,}, {percentage:.6f}%)")

    def show_length_distribution(self: Any, num_bins: int = 20, batch_size: int = 10_000) -> None:
        """Display the distribution of text lengths in the dataset.
        @param num_bins: Number of bins for the histogram
        @param batch_size: Batch size for processing
        """
        # Calculate lengths in batches
        self.dataset = self.dataset.map(self._calculate_length, batched=True, batch_size=batch_size, num_proc=self.num_proc)

        # Show the length distribution
        self._show_length_distribution(self.dataset['length'], num_bins=num_bins)

        # Remove the 'length' column after displaying the distribution
        self.dataset = self.dataset.remove_columns(['length'])

        # Print the dataset schema after processing
        print(self.dataset)

    def save_as_parquets(self: Any, num_samples_per_file: int = 10_000_000) -> None:
        """Save the dataset as Parquet files.
        @param num_samples_per_file: Number of samples per Parquet file
        """
        # Split the dataset into smaller files
        num_shards: int = (len(self.dataset) + num_samples_per_file - 1) // num_samples_per_file

        # Save each shard as a Parquet file
        for shard_idx in range(num_shards):
            shard: Dataset = self.dataset.shard(num_shards=num_shards, index=shard_idx)
            shard_path: str = os.path.join(self._save_dir, f"{self.dataset_name}_shard_{shard_idx}.parquet")
            shard.to_parquet(shard_path)
            print(f"Saved shard {shard_idx + 1}/{num_shards} to {shard_path}")


class DatasetPreprocessor(DatasetProcessor):
    """A class to download and pre-process datasets for training."""

    def __init__(self: Any, dataset_name: str, configs: DatasetsConfigs, num_proc: int = 16) -> None:
        """Initialize the DatasetPreprocessor.
        @param dataset_name: Name of the dataset
        @param configs: Configuration for the datasets
        @param num_proc: Number of processes to use for parallel processing
        """
        super().__init__(configs=configs, num_proc=num_proc)

        # Saving attributes
        self._ori_save_dir: str = self._create_save_dir(os.path.join(configs.ORI_DATASETS_DIR, dataset_name))
        self._save_dir: str = self._create_save_dir(os.path.join(configs.PREPROCESSED_DATASETS_DIR, dataset_name))

        # Dataset attributes
        self.dataset_name: str = dataset_name

    def load_ori_dataset(self: Any, dataset: Dataset) -> None:
        """Load the original dataset.
        @param dataset: The original dataset to load
        """
        # Load the dataset
        self.dataset = dataset

    @property
    def ori_save_dir(self: Any) -> str:
        """Get the original save directory."""
        return self._ori_save_dir

    @property
    def save_dir(self: Any) -> str:
        """Get the preprocessed save directory."""
        return self._save_dir

    def _split_by_paragraph(self: Any, batch: Any, length_range: Tuple[int, int]) -> Any:
        """Split the text into paragraphs based on specified length range.
        @param batch: A batch of data
        @param length_range: A tuple specifying the minimum and maximum length of paragraphs
        @return: The batch with a new 'text' field containing the split paragraphs
        """
        # Get necessary variables
        min_length, max_length = length_range
        splited_texts: List[str] = []

        # Iterate through each text in the batch
        for text in batch['text']:
            if len(text) < min_length: continue
            elif len(text) <= max_length: splited_texts.append(text)
            else:
                parts: List[str] = text.split('\n')
                parts = list(filter(lambda x: min_length <= len(x) <= max_length, parts))
                splited_texts.extend(parts)

        # Return the batch with the new 'text' field
        return {'text': splited_texts}

    def split_by_paragraph(self: Any, length_range: Tuple[int, int] = (64, 512), batch_size: int = 10_000) -> None:
        """Split the text into paragraphs based on specified length range.
        @param length_range: A tuple specifying the minimum and maximum length of paragraphs
        @param batch_size: Batch size for processing
        """
        # Split the text into paragraphs
        self.dataset = self.dataset.map(self._split_by_paragraph, fn_kwargs={'length_range': length_range},
                                        batched=True, batch_size=batch_size, num_proc=self.num_proc)

    def run_pipeline(self: Any, length_range: Tuple[int, int], num_bins: int, batch_size: int,
                        num_samples_per_file: int) -> None:
        """Run the entire data preprocessing pipeline."""
        print("Starting data preprocessing pipeline...")
        print(f"==================================================")
        print(f"Show length distribution before splitting by paragraphs:")
        print(f"==================================================")
        self.show_length_distribution(num_bins=num_bins, batch_size=batch_size)
        print(f"")
        print(f"==================================================")
        print(f"Splitting dataset by paragraphs with length range {length_range}...")
        self.split_by_paragraph(length_range=length_range, batch_size=batch_size)
        print(f"")
        print(f"==================================================")
        print(f"Show length distribution after splitting by paragraphs:")
        self.show_length_distribution(num_bins=num_bins, batch_size=batch_size)
        print(f"")
        print(f"==================================================")
        print(f"Saving dataset as Parquet files...")
        self.save_as_parquets(num_samples_per_file=num_samples_per_file)
        print(f"Data preprocessing pipeline completed successfully.")


class DatasetsCombiner(DatasetProcessor):
    """A class to combine multiple datasets into one."""

    def __init__(self: Any, configs: DatasetsConfigs, num_proc: int = 16) -> None:
        """Initialize the DatasetsCombiner.
        @param configs: Configuration for datasets
        @param num_proc: Number of processes to use for parallel processing
        """
        super().__init__(configs=configs, num_proc=num_proc)

        # Saving attributes
        self._save_dir: str = self._create_save_dir(configs.COMBINED_DATASETS_DIR)

        # Dataset attributes
        self.dataset_name: str = "combined_dataset"
        self.datasets: List[Dataset] = []

    def load_datasets(self: Any, datasets_paths: List[str]) -> None:
        """Load datasets based on the provided configurations."""
        for dataset_path in datasets_paths:
            dataset: Dataset = load_dataset(
                "parquet", data_files=os.path.join(dataset_path, f"*.parquet"),
                num_proc=self.num_proc, split='train'
            )
            self.datasets.append(dataset)
            print(f"Loaded dataset from {dataset_path} with {len(dataset)} examples.")

    def combine_datasets(self: Any) -> None:
        """Combine multiple datasets into one."""
        self.dataset = concatenate_datasets(self.datasets)

    def run_pipeline(self: Any, datasets_paths: List[str], num_bins: int, batch_size: int,
                        num_samples_per_file: int = 10_000_000) -> None:
        """Run the entire datasets combining pipeline."""
        print("Starting datasets combining pipeline...")
        print(f"==================================================")
        print(f"Loading datasets...")
        self.load_datasets(datasets_paths=datasets_paths)
        print(f"Loaded {len(self.datasets)} datasets.")
        print(f"")
        print(f"==================================================")
        print(f"Combining datasets...")
        self.combine_datasets()
        print(f"Combined dataset has {len(self.dataset)} examples.")
        print(f"")
        print(f"==================================================")
        print(f"Show length distribution of the combined dataset:")
        self.show_length_distribution(num_bins=num_bins, batch_size=batch_size)
        print(f"")
        print(f"==================================================")
        print(f"Saving combined dataset as Parquet files...")
        self.save_as_parquets(num_samples_per_file=num_samples_per_file)
        print(f"Datasets combining pipeline completed successfully.")


