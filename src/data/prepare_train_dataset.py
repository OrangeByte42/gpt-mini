import os
from typing import Any, List, Tuple, Dict

from datasets import DatasetDict, Dataset
from datasets import load_dataset, concatenate_datasets

from src.configs.data_configs import DatasetConfigs
from src.data.utils import show_length_distribution
from src.data.utils import load_parquets, save_as_parquets


def split_long_texts(flatten_dataset: Dataset, column_name: str, length_range: Tuple[int, int],
                        process_batch_size: int, num_proc: int) -> Dataset:
    """Split long texts in a specific column of the dataset into smaller chunks.
    @param flatten_dataset: The dataset to process.
    @param column_name: The name of the column containing the texts to split.
    @param length_range: A tuple specifying the minimum and maximum length of the chunks.
    @param process_batch_size: The batch size for processing the dataset.
    @param num_proc: The number of processes to use for parallel computation.
    @return: A new dataset with the long texts split into smaller chunks.
    """

    def _split_long_texts4map(batch: Any, column_name: str, length_range: Tuple[int, int]) -> Dict[str, List[str]]:
        """Split long texts into smaller chunks.
        @param batch: A batch of data from the dataset.
        @param column_name: The name of the column containing the texts to split.
        @param length_range: A tuple specifying the minimum and maximum length of the chunks.
        @return: A dictionary with the column name as key and a list of split texts as value.
        """
        # Prepare necessary variables
        min_length, max_length = length_range
        splited_texts: List[str] = []

        # Iterate through each text in the batch
        lengths: List[int] = list(map(len, batch[column_name]))
        for length, text in zip(lengths, batch[column_name]):
            if length < min_length: continue
            elif length <= max_length: splited_texts.append(text)
            else:
                parts: List[str] = text.split("\n")
                parts = list(filter(lambda x: min_length <= len(x) <= max_length, parts))
                splited_texts.extend(parts)

        # Return the split texts as a dictionary
        return {column_name: splited_texts}

    # Split long texts in the specified column
    splited_dataset: Dataset = flatten_dataset.map(
        function=_split_long_texts4map,
        fn_kwargs={'column_name': column_name, 'length_range': length_range},
        batched=True,
        batch_size=process_batch_size,
        num_proc=num_proc,
        remove_columns=flatten_dataset.column_names
    )

    # Return the new dataset with split texts
    return splited_dataset

def run_ori_dataset_loading_pipeline(configs: DatasetConfigs, hf_ds_addr: str, local_ds_name: str,
                                        column_name: str) -> str:
    """Load original datasets from Hugging Face or local files and return the dataset name.
    @param configs: Dataset configurations.
    @param hf_ds_addr: The address of the dataset on Hugging Face.
    @param local_ds_name: The name of the local dataset to save.
    @param column_name: The name of the column containing the texts to process.
    @return: The directory where the processed dataset is saved.
    """

    # Starting pipeline for loading original datasets
    print("Starting pipeline for loading original datasets...")
    print(f"=================================================================")

    # Initialize necessary directories
    print(f"Creating directories for original and processed datasets...")
    print(f"==================================================================")
    ori_cache_dir: str = os.path.join(configs.ORIGINAL_CACHE_DIR, local_ds_name)
    processed_cache_dir: str = os.path.join(configs.PROCESSED_CACHE_DIR, local_ds_name)

    os.makedirs(ori_cache_dir, exist_ok=True)
    os.makedirs(processed_cache_dir, exist_ok=True)

    # Try to load the processed dataset from cache
    print(f"Trying to load the processed dataset from cache...")
    print(f"==================================================================")
    try:
        # Attempt to load the processed dataset from the cache directory
        dataset_dict: DatasetDict = load_parquets(save_dir=processed_cache_dir, streaming=False)
        if len(dataset_dict) == 0: raise ValueError("No datasets found in the processed cache.")
    except Exception as e:
        # If loading fails, process the original dataset
        print(f"Failed to load processed dataset from cache: {e}")
        print(f"Processing the original dataset...")
        print(f"")
    else:
        # If loading is successful, return the processed cache directory
        print(f"Successfully loaded processed dataset from cache.")
        print(f"")

        # Show processed dataset's length distribution
        print(f"Showing processed dataset's length distribution...")
        print(f"==================================================================")
        flatten_dataset: Dataset = concatenate_datasets(dsets=dataset_dict.values())
        show_length_distribution(
            flatten_dataset=flatten_dataset,
            column_name=column_name,
            num_bins=configs.NUM_BINS,
            process_batch_size=configs.PROCESS_BATCH_SIZE,
            num_proc=configs.NUM_PROC,
        )
        print(f"")

        # Return the processed cache directory
        return processed_cache_dir

    # Download and load datasets
    print(f"Downloading and loading datasets from {hf_ds_addr}...")
    print(f"==================================================================")
    dataset_dict: DatasetDict = load_dataset(
        path=hf_ds_addr,
        cache_dir=ori_cache_dir,
        num_proc=configs.NUM_PROC,
    )
    print(f"")

    # Flatten the dataset
    print(f"Flattening the dataset...")
    print(f"==================================================================")
    flatten_dataset: Dataset = concatenate_datasets(dsets=dataset_dict.values())
    print(f"")

    # Show original dataset's length distribution
    print(f"Showing original dataset's length distribution...")
    print(f"==================================================================")
    show_length_distribution(
        flatten_dataset=flatten_dataset,
        column_name=column_name,
        num_bins=configs.NUM_BINS,
        process_batch_size=configs.PROCESS_BATCH_SIZE,
        num_proc=configs.NUM_PROC,
    )
    print(f"")

    # If loading from cache failed, process the original dataset
    print(f"Processing the original dataset...")
    print(f"==================================================================")
    flatten_dataset = split_long_texts(
        flatten_dataset=flatten_dataset,
        column_name=column_name,
        length_range=configs.LENGTH_RANGE,
        process_batch_size=configs.PROCESS_BATCH_SIZE,
        num_proc=configs.NUM_PROC,
    )
    print(f"")

    # Show the processed dataset's length distribution
    print(f"Showing processed dataset's length distribution...")
    print(f"==================================================================")
    show_length_distribution(
        flatten_dataset=flatten_dataset,
        column_name=column_name,
        num_bins=configs.NUM_BINS,
        process_batch_size=configs.PROCESS_BATCH_SIZE,
        num_proc=configs.NUM_PROC,
    )
    print(f"")

    # Save the processed dataset to cache
    print(f"Saving the processed dataset to cache...")
    print(f"==================================================================")
    dataset_dict: DatasetDict = DatasetDict({"train": flatten_dataset})
    save_as_parquets(
        dataset_dict=dataset_dict,
        save_dir=processed_cache_dir,
        save_batch_size= configs.SAVE_BATCH_SIZE,
    )
    print(f"Successfully saved the processed dataset to cache.")
    print(f"")

def run_combine_datasets_pipeline(configs: DatasetConfigs, dataset_dirs: List[str]) -> None:
    """Combine multiple datasets into a single dataset and save it to cache.
    @param configs: Dataset configurations.
    @param dataset_dirs: List of directories containing the datasets to combine.
    """

    # Starting pipeline for combining datasets
    print("Starting pipeline for combining datasets...")
    print(f"=================================================================")

    # Combine datasets
    print(f"Combining datasets...")
    print(f"==================================================================")
    processed_datasets: List[Dataset] = []
    for dataset_dir in dataset_dirs:
        processed_dataset: DatasetDict = load_parquets(save_dir=dataset_dir, streaming=False)
        processed_dataset = concatenate_datasets(dsets=processed_dataset.values())
        processed_datasets.append(processed_dataset)
    combined_dataset: Dataset = concatenate_datasets(dsets=processed_datasets)
    print(f"Successfully combined {len(processed_datasets)} datasets.")
    print(f"")

    # Show combined dataset's length distribution
    print(f"Showing combined dataset's length distribution...")
    print(f"==================================================================")
    show_length_distribution(
        flatten_dataset=combined_dataset,
        column_name="text",
        num_bins=configs.NUM_BINS,
        process_batch_size=configs.PROCESS_BATCH_SIZE,
        num_proc=configs.NUM_PROC,
    )

    # Split the combined dataset into train and test sets
    print(f"Splitting the combined dataset into train and test sets...")
    print(f"==================================================================")
    combined_dataset = combined_dataset.train_test_split(
        train_size=configs.TRAIN_RATIO,
        test_size=configs.VALID_RATIO,
        seed=configs.SEED,
        shuffle=True,
    )
    print(f"Successfully split the combined dataset into train and test sets.")
    print(f"")

    # Save the combined dataset to cache
    print(f"Saving the combined dataset to cache...")
    print(f"==================================================================")
    save_as_parquets(
        dataset_dict=combined_dataset,
        save_dir=configs.TRAIN_CACHE_DIR,
        save_batch_size=configs.SAVE_BATCH_SIZE,
    )
    print(f"Successfully saved the combined dataset to cache.")
    print(f"")


if __name__ == "__main__":
    """Main function to run the dataset loading pipeline."""
    # Load dataset configurations
    configs: DatasetConfigs = DatasetConfigs()

    # Load original datasets from Hugging Face or local files
    ## wikipedia-en
    wikipedia_en: str = run_ori_dataset_loading_pipeline(
        configs=configs,
        hf_ds_addr="izumi-lab/wikipedia-en-20230720",
        local_ds_name="wikipedia_en",
        column_name="text"
    )

    ## open-text-books
    open_text_books: str = run_ori_dataset_loading_pipeline(
        configs=configs,
        hf_ds_addr="izumi-lab/open-text-books",
        local_ds_name="open_text_books",
        column_name="text"
    )

    # Combine datasets into a single dataset
    run_combine_datasets_pipeline(
        configs=configs,
        dataset_dirs=[wikipedia_en, open_text_books]
    )

