import os
from typing import Any, List, Optional, Dict

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from datasets import DatasetDict, Dataset
from datasets import load_dataset

from src.configs.configs import DatasetConfigs, TokenizerConfigs
from src.data.utils import load_parquets


class BpeTokenizer:
    """BPE Tokenizer class for handling BPE tokenization."""

    def __init__(self: Any, tokenizer_configs: TokenizerConfigs,
                    dataset_configs: DatasetConfigs) -> None:
        """Initialize the BPE tokenizer with the given configurations.
        @param tokenizer_configs: Configuration for the tokenizer.
        @param dataset_configs: Configuration for the dataset.
        """

        # Set the tokenizer configurations
        self.tokenizer_configs: TokenizerConfigs = tokenizer_configs
        self.dataset_configs: DatasetConfigs = dataset_configs

        # Create necessary directories
        self.cache_dir: str = os.path.join(tokenizer_configs.TOKENIZER_CACHE_DIR,
                                                        tokenizer_configs.TOKENIZER_NAME)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.tokenizer_path: str = os.path.join(self.cache_dir, "tokenizer.json")
        self.train_data_path: str = os.path.join(self.cache_dir, "train_data.txt")

        # Tokenizer
        self.tokenizer: Optional[PreTrainedTokenizerFast] = None

        # Special tokens
        self._unk_token: str = "[UNK]"
        self._pad_token: str = "[PAD]"
        self._sos_token: str = "[SOS]"
        self._eos_token: str = "[EOS]"

    @property
    def unk_token(self: Any) -> str:
        """Return the unknown token."""
        return self._unk_token

    @property
    def pad_token(self: Any) -> str:
        """Return the padding token."""
        return self._pad_token

    @property
    def sos_token(self: Any) -> str:
        """Return the start of sequence token."""
        return self._sos_token

    @property
    def eos_token(self: Any) -> str:
        """Return the end of sequence token."""
        return self._eos_token

    def run_tokenizer_generation_pipeline(self: Any) -> None:
        """Run the tokenizer generation pipeline."""

        # Check if cached tokenizer exists
        if os.path.exists(self.tokenizer_path):
            print(f"Tokenizer already exists at {self.tokenizer_path}.")
            print(f"If you want to regenerate it, please delete the file first then run the script again.")
            print(f"")
            return

        # Starting tokenizer generation pipeline
        print("Starting tokenizer generation pipeline...")
        print(f"=================================================================")

        # Generate training data for the tokenizer training
        print(f"Generating training data for tokenizer...")
        print(f"=================================================================")
        combine_dataset_dict: DatasetDict = load_parquets(self.dataset_configs.TRAIN_CACHE_DIR, streaming=True)

        print(f"Writing tokenizer training texts to {self.train_data_path}...")
        with open(self.train_data_path, "w", encoding="utf-8") as f:
            for split_name, split_dataset in combine_dataset_dict.items():
                for batch in split_dataset.batch(batch_size=self.dataset_configs.PROCESS_BATCH_SIZE):
                    texts: List[str] = batch["text"]
                    f.write("\n".join(texts) + "\n")
        print(f"Successfully wrote tokenizer training texts to {self.train_data_path}.")
        print(f"")

        # Initialize the tokenizer
        print(f"Initializing the tokenizer...")
        print(f"=================================================================")
        tokenizer: Tokenizer = Tokenizer(BPE(unk_token=self.unk_token))
        tokenizer.pre_tokenizer = Whitespace()

        # Train the tokenizer
        print(f"Training the tokenizer...")
        print(f"=================================================================")
        trainer: BpeTrainer = BpeTrainer(
            vocab_size=self.tokenizer_configs.MAX_VOCAB_SIZE,
            min_frequency=self.tokenizer_configs.MIN_FREQ,
            special_tokens=[
                self._unk_token,
                self._pad_token,
                self._sos_token,
                self._eos_token,
            ]
        )

        tokenizer.train(files=[self.train_data_path], trainer=trainer)

        # Save the tokenizer
        print(f"Saving the tokenizer to {self.tokenizer_path}...")
        print(f"=================================================================")
        tokenizer.save(self.tokenizer_path)
        print(f"Successfully saved the tokenizer to {self.tokenizer_path}.")
        print(f"")

    def load_tokenizer(self: Any) -> PreTrainedTokenizerFast:
        """Load the tokenizer from the cache directory."""

        print(f"Loading the tokenizer from {self.tokenizer_path}...")
        print(f"=================================================================")
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=self.tokenizer_path)
        self.tokenizer.pad_token = self._pad_token
        self.tokenizer.unk_token = self._unk_token
        self.tokenizer.sos_token = self._sos_token
        self.tokenizer.eos_token = self._eos_token

    def _tokenize(self: Any, batch: Any) -> Any:
        """Tokenize a batch of data.
        @param batch: A batch of data to tokenize.
        @return: A batch of tokenized data.
        """
        # Add start and end of sequence tokens to the text
        column_name: str = "text"
        batch[column_name] = [
            self.sos_token + text + self.eos_token for text in batch[column_name]
        ]

        # Tokenize the batch
        token_ids_batch: Dict[str, List[int]] = self.tokenizer(batch[column_name], truncation=True, padding=True,
                                                                max_length=self.dataset_configs.MAX_LENGTH,
                                                                add_special_tokens=True)

        # Return the tokenized batch
        return token_ids_batch

    def run_dataset_tokenization_pipeline(self: Any) -> None:
        """Tokenize the dataset using the BPE tokenizer."""

        # Tokenize the dataset
        print(f"Starting dataset tokenization pipeline...")
        print(f"=================================================================")

        print(f"Tokenizing the dataset...")
        print(f"=================================================================")

        # Get directory of Input and Output
        input_dir: str = self.dataset_configs.TRAIN_CACHE_DIR
        output_dir: str = self.dataset_configs.TOKENIZED_CACHE_DIR

        # Iterate through each split in the input dataset directory
        for split_name in os.listdir(input_dir):
            # Create a directory for the tokenized split
            tokenized_split_dir: str = os.path.join(output_dir, split_name)
            os.makedirs(tokenized_split_dir, exist_ok=True)

            # Iterate through each shard in the split directory
            for shard_name in os.listdir(os.path.join(input_dir, split_name)):
                input_shard_path: str = os.path.join(input_dir, split_name, shard_name)
                output_shard_path: str = os.path.join(tokenized_split_dir, shard_name)
                # Load the dataset shard
                input_dataset: Dataset = load_dataset("parquet", data_files=input_shard_path, streaming=False)["train"]
                # Tokenize the dataset shard
                output_dataset: Dataset = input_dataset.map(
                    function=self._tokenize,
                    batched=True,
                    batch_size=self.dataset_configs.PROCESS_BATCH_SIZE,
                    num_proc=self.dataset_configs.NUM_PROC,
                    remove_columns=input_dataset.column_names,
                )
                # Save the tokenized dataset shard
                output_dataset.to_parquet(output_shard_path)

        print(f"Successfully tokenized the dataset and saved it to {output_dir}.")
        print(f"")



if __name__ == "__main__":
    """Main function to run the BPE tokenizer generation pipeline."""

    # Initialize the BPE tokenizer with configurations
    bpe_tokenizer: BpeTokenizer = BpeTokenizer(
        tokenizer_configs=TokenizerConfigs(),
        dataset_configs=DatasetConfigs(),
    )

    # Run the tokenizer generation pipeline
    bpe_tokenizer.run_tokenizer_generation_pipeline()
    bpe_tokenizer.load_tokenizer()
    bpe_tokenizer.run_dataset_tokenization_pipeline()

