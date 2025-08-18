import os
from typing import Any, List

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from datasets import DatasetDict, Dataset
from datasets import load_dataset, concatenate_datasets
import pandas as pd

from src.configs.data_configs import DatasetConfigs, TokenizerConfigs
from src.data.utils import load_parquets


def run_tokenizer_generation_pipeline(
    tokenizer_configs: TokenizerConfigs,
    dataset_configs: DatasetConfigs,
    tokenizer_save_path: str,
) -> None:
    """Train a tokenizer using the provided configurations and dataset.
    @param tokenizer_configs: Configuration for the tokenizer.
    @param dataset_configs: Configuration for the dataset.
    @param tokenizer_save_path: Path where the trained tokenizer will be saved.
    @return: Save path of the trained tokenizer.
    """

    # Starting tokenizer generation pipeline
    print("Starting tokenizer generation pipeline...")
    print(f"=================================================================")

    # Generate training data for the tokenizer training
    print(f"Generating training data for tokenizer...")
    print(f"=================================================================")
    tokenizer_train_data_path: str = os.path.join(tokenizer_configs.TOKENIZER_CACHE_DIR, "train_data.txt")
    combined_dataset_dict: DatasetDict = load_parquets(dataset_configs.TRAIN_CACHE_DIR, streaming=True)

    if os.path.exists(tokenizer_train_data_path):
        print(f"Training data already exists at {tokenizer_train_data_path}.")
        print(f"If you want to regenerate it, please delete the file first and then run the script again.")
        print(f"")
    else:
        print(f"Writing tokenizer training texts to {tokenizer_train_data_path}...")
        with open(tokenizer_train_data_path, "w", encoding="utf-8") as f:
            for split_name, split_dataset in combined_dataset_dict.items():
                for batch in split_dataset.batch(batch_size=dataset_configs.PROCESS_BATCH_SIZE):
                    texts: List[str] = batch["text"]
                    f.write("\n".join(texts)  + "\n")
        print(f"Successfully wrote tokenizer training data to {tokenizer_train_data_path}.")
        print(f"")

    if os.path.exists(tokenizer_save_path):
        print(f"Tokenizer already exists at {tokenizer_save_path}.")
        print(f"If you want to regenerate it, please delete the file first and then run the script again.")
        print(f"")
        return
    else:
        # Initialize the tokenizer
        print(f"Initializing the tokenizer...")
        print(f"=================================================================")
        tokenizer: Tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        # Train the tokenizer
        print(f"Training the tokenizer...")
        print(f"=================================================================")
        trainer: BpeTrainer = BpeTrainer(
            vocab_size=tokenizer_configs.MAX_VOCAB_SIZE,
            min_frequency=tokenizer_configs.MIN_FREQ,
            special_tokens=[
                "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]",
                "[BOS]", "[EOS]"
            ]
        )
        tokenizer.train(files=[tokenizer_train_data_path], trainer=trainer)

        # Save the tokenizer
        print(f"Saving the tokenizer...")
        print(f"=================================================================")
        tokenizer.save(tokenizer_save_path)
        print(f"Tokenizer saved to {tokenizer_save_path}.")
        print(f"")

def run_tokenize_dataset_pipeline(
    dataset_configs: DatasetConfigs,
    tokenizer_file: str,
) -> None:
    """Tokenize the dataset using the trained tokenizer.
    @param dataset_configs: Configuration for the dataset.
    @param tokenizer_file: Path to the trained tokenizer.
    """

    def _tokenize_function(batch: Any, column_name: str, max_length: int, tokenizer: PreTrainedTokenizerFast) -> Any:
        """Tokenize a batch of data.
        @param batch: A batch of data to tokenize.
        @param column_name: The name of the column to tokenize.
        @param max_length: The maximum length of the tokenized sequences.
        @param tokenizer: The tokenizer to use for tokenization.
        """
        return tokenizer(batch[column_name], truncation=True, padding=True, max_length=max_length,
                            add_special_tokens=True)

    # Starting dataset tokenization pipeline
    print("Starting dataset tokenization pipeline...")
    print(f"=================================================================")

    # Load the tokenizer
    print(f"Loading the tokenizer from {tokenizer_file}...")
    print(f"=================================================================")
    fast_tokenizer: PreTrainedTokenizerFast = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
    fast_tokenizer.pad_token = "[PAD]"
    fast_tokenizer.unk_token = "[UNK]"
    fast_tokenizer.cls_token = "[CLS]"
    fast_tokenizer.sep_token = "[SEP]"
    fast_tokenizer.mask_token = "[MASK]"
    fast_tokenizer.bos_token = "[BOS]"
    fast_tokenizer.eos_token = "[EOS]"

    # Tokenize the dataset
    print(f"Tokenizing the dataset...")
    print(f"=================================================================")

    # Get directory for Input and Output
    combined_dataset_dir: str = dataset_configs.TRAIN_CACHE_DIR
    tokenized_dataset_dir: str = dataset_configs.TOKENIZED_CACHE_DIR

    # Iterate through each split in the combined dataset
    for split_name in os.listdir(combined_dataset_dir):
        # Create a directory for the tokenized split
        tokenized_split_path: str = os.path.join(tokenized_dataset_dir, split_name)
        os.makedirs(tokenized_split_path, exist_ok=True)
        for shard_name in os.listdir(os.path.join(combined_dataset_dir, split_name)):
            input_shard_path: str = os.path.join(combined_dataset_dir, split_name, shard_name)
            tokenized_shard_path: str = os.path.join(tokenized_split_path, shard_name)
            # Load the shard file
            input_dataset: Dataset = load_dataset("parquet", data_files=input_shard_path, streaming=False)["train"]
            # Tokenize the dataset
            tokenized_dataset: Dataset = input_dataset.map(
                function=_tokenize_function,
                fn_kwargs={"column_name": "text", "max_length": dataset_configs.MAX_LENGTH, "tokenizer": fast_tokenizer},
                batched=True,
                batch_size=dataset_configs.PROCESS_BATCH_SIZE,
                remove_columns=input_dataset.column_names,
                num_proc=dataset_configs.NUM_PROC,
            )
            # Save the tokenized dataset
            tokenized_dataset.to_parquet(tokenized_shard_path)

    print(f"Successfully tokenized the dataset and saved to {tokenized_dataset_dir}.")
    print(f"")

if __name__ == "__main__":
    """Run the tokenizer generation and dataset tokenization pipelines."""

    # Load configurations
    tokenizer_configs: TokenizerConfigs = TokenizerConfigs()
    dataset_configs: DatasetConfigs = DatasetConfigs()

    tokenizer_save_path: str = os.path.join(tokenizer_configs.TOKENIZER_CACHE_DIR, "tokenizer.json")

    # Generate the tokenizer
    if os.path.exists(tokenizer_save_path):
        print(f"Tokenizer already exists at {tokenizer_save_path}.")
        print(f"If you want to regenerate it, please delete the file first and then run the script again.")
        print(f"")
    else:
        run_tokenizer_generation_pipeline(
            tokenizer_configs=tokenizer_configs,
            dataset_configs=dataset_configs,
            tokenizer_save_path=tokenizer_save_path,
        )

    # Tokenize the dataset
    run_tokenize_dataset_pipeline(
        dataset_configs=dataset_configs,
        tokenizer_file=tokenizer_save_path,
    )





