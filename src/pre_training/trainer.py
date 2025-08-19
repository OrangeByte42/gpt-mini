import os
import time
import datetime
import math
from typing import Any, Optional, List, Dict, Union

import torch
from torch import nn
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, DistributedSampler
from datasets import DatasetDict, Dataset

from src.configs.configs import DatasetConfigs, TokenizerConfigs, ModelConfigs, TrainingConfigs
from src.data.utils import load_parquets
from src.data.bpe_tokenizer import BpeTokenizer
from src.models.gpt_mini import GPTMini
from src.utils.utils import count_parameters, epoch_time, cleanup, save_obj_by_pickle
from src.generator.autoregressive_gpt_mini import AutoregressiveGPTMini



class Trainer:
    """Trainer class for pre-training."""

    def __init__(self: Any,
                    dataset_configs: DatasetConfigs,
                    tokenizer_configs: TokenizerConfigs,
                    model_configs: ModelConfigs,
                    training_configs: TrainingConfigs,
                    ddp: bool,
                    samples: List[str]) -> None:
        """Initialize the Trainer with configurations.
        @param dataset_configs: Dataset configurations.
        @param tokenizer_configs: Tokenizer configurations.
        @param model_configs: Model configurations.
        @param training_configs: Training configurations.
        @param ddp: Whether to use Distributed Data Parallel (DDP).
        @param samples: Sample texts for inference after each epoch.
        """
        # Configurations
        self.dataset_configs: DatasetConfigs = dataset_configs
        self.tokenizer_configs: TokenizerConfigs = tokenizer_configs
        self.model_configs: ModelConfigs = model_configs
        self.training_configs: TrainingConfigs = training_configs

        # Dataset & Tokenizer & Model
        self.train_dataloader: Optional[DataLoader] = None
        self.valid_dataloader: Optional[DataLoader] = None
        self.tokenizer: Optional[BpeTokenizer] = None
        self.model: Optional[GPTMini] = None

        # Training components
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[ReduceLROnPlateau] = None
        self.criterion: Optional[nn.CrossEntropyLoss] = None

        self.ddp: bool = ddp

        # Training trace record
        self.best_loss: float = float("inf")
        self.train_losses: List[float] = list()
        self.valid_losses: List[float] = list()

        self.sample_trace: List[Dict[str, List[str]]] = [{"prompt": sample, "generated": []} for sample in samples]

    def _load_dataset(self: Any) -> None:
        """Load the dataset."""
        # Load the dataset from the tokenized cache directory
        dataset_save_dir: str = self.dataset_configs.TOKENIZED_CACHE_DIR
        dataset_dict: DatasetDict = load_parquets(dataset_save_dir, streaming=True)

        # Convert the dataset dict to pytorch DataLoaders
        train_dataset: Dataset = dataset_dict["train"]
        valid_dataset: Dataset = dataset_dict["test"]

        if self.ddp:
            # Create distributed samplers for training and validation datasets
            train_sampler: DistributedSampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
            valid_sampler: DistributedSampler = DistributedSampler(valid_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())

            # Create DataLoaders with distributed samplers
            self.train_dataloader = DataLoader(train_dataset, batch_size=self.training_configs.BATCH_SIZE,
                                                            sampler=train_sampler, num_workers=self.training_configs.NUM_WORKERS)
            self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.training_configs.BATCH_SIZE,
                                                            sampler=valid_sampler, num_workers=self.training_configs.NUM_WORKERS)
        else:
            # Create DataLoaders without distributed samplers
            self.train_dataloader = DataLoader(train_dataset, batch_size=self.training_configs.BATCH_SIZE, num_workers=self.training_configs.NUM_WORKERS)
            self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.training_configs.BATCH_SIZE, num_workers=self.training_configs.NUM_WORKERS)

        # Echo the dataset information
        if not self.ddp or (self.ddp and dist.is_initialized() and dist.get_rank() == 0):
            print(f"[Loading Dataset]:")
            print(f"Load dataset from: {dataset_save_dir}")
            print(f"Train dataloaer: {self.train_dataloader.dataset}")
            print(f"Valid dataloader: {self.valid_dataloader.dataset}")
            print(f"Loaded dataset successfully.", end="\n\n")

    def _load_tokenizer(self: Any) -> BpeTokenizer:
        """Load the tokenizer."""
        # Load the tokenizer from the cache directory
        tokenizer: BpeTokenizer = BpeTokenizer(self.tokenizer_configs, self.dataset_configs)
        tokenizer.load_tokenizer()

        # Set the tokenizer to the instance variable
        self.tokenizer = tokenizer

        # Echo the tokenizer information
        if not self.ddp or (self.ddp and dist.is_initialized() and dist.get_rank() == 0):
            print(f"[Loading Tokenizer]:")
            print(f"Load tokenizer from: {self.tokenizer_configs.TOKENIZER_CACHE_DIR}")
            print(f"Tokenizer vocabulary size: {self.tokenizer.vocab_size}")
            print(f"Tokenizer pad ID: {self.tokenizer.pad_id}")
            print(f"Loaded tokenizer successfully.", end="\n\n")

    def _load_model(self: Any, device: torch.device) -> GPTMini:
        """Load the model."""
        # Instantiate the GPTMini model with configurations
        gpt_mini: GPTMini = GPTMini(
            vocab_size=self.tokenizer_configs.MAX_VOCAB_SIZE,
            max_seq_len=self.dataset_configs.MAX_LENGTH,
            pad_id=self.tokenizer.pad_id,
            num_layers=self.model_configs.NUM_LAYERS,
            d_model=self.model_configs.D_MODEL,
            num_heads=self.model_configs.NUM_HEADS,
            d_ff=self.model_configs.D_FF,
            drop_prob=self.model_configs.DROP_PROB,
            device=device,
        )

        # Initialize the model weights
        gpt_mini.init_weights()

        # Set the model to the instance variable
        self.model = gpt_mini

        # Echo the model parameters
        if not self.ddp or (self.ddp and dist.is_initialized() and dist.get_rank() == 0):
            total_params, trainable_params = count_parameters(gpt_mini, self.ddp)
            print(f"[Instantiatin Model]:")
            print(f"Model Parameters: {total_params:,} ({total_params / 1e9:.4f} B)")
            print(f"Trainable Parameters: {trainable_params:,} ({trainable_params / 1e9:.4f} B)")
            print(f"Instantiated GPTMini model successfully.", end="\n\n")

    def _init_training_components(self: Any, device: torch.device) -> None:
        """Initialize the training components."""
        # Ensure the model is initialized
        assert self.model is not None, "Model must be initialized before training components."

        # Create the optimizer
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.training_configs.INIT_LR,
            weight_decay=self.training_configs.WEIGHT_DECAY,
            eps=self.training_configs.ADAM_EPS,
        )

        # Create the learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            factor=self.training_configs.FACTOR,
            patience=self.training_configs.PATIENCE,
        )

        # Create the loss criterion
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id).to(device)

    def _epoch_train(self: Any, model: Any, device: torch.device, ddp: bool) -> float:
        """Train the model for one epoch."""
        # Set the model to training mode and create necessary variables
        model.train()

        epoch_loss: float = 0.0
        total_batches: int = 0

        # iterate over the training dataloader
        for batch in self.train_dataloader:
            # Get batch data and move to device
            X: torch.Tensor = torch.stack(batch["input_ids"]).transpose(0, 1).to(device)

            # Inference the model output
            # Teacher forcing: use X[:, :-1] as input and X[:, 1:] as target
            output: torch.Tensor = model(X[:, :-1])
            reshaped_output: torch.Tensor = output.contiguous().view(-1, output.size(-1))
            reshaped_target: torch.Tensor = X[:, 1:].contiguous().view(-1)

            # Calculate the loss
            loss: torch.Tensor = self.criterion(reshaped_output, reshaped_target)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.training_configs.CLIP)
            self.optimizer.step()

            # Update epoch loss and total batches
            epoch_loss += loss.item()
            total_batches += 1

        # Convert training loss and batch count to tensor
        local_loss: torch.Tensor = torch.tensor(epoch_loss, device=device)
        local_batch_count: torch.Tensor = torch.tensor(total_batches, device=device)

        # If using DDP, reduce the loss and batch count across all processes
        if ddp == True:
            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_batch_count, op=dist.ReduceOp.SUM)

        # Calculate the average loss across all processes
        global_avg_loss: float = local_loss.item() / local_batch_count.item()

        # Return the average loss for the epoch
        return global_avg_loss

    def _epoch_valid(self: Any, model: Any, device: torch.device, ddp: bool) -> float:
        """Validate the model for one epoch."""
        # Set the model to evaluation mode and create necessary variables
        model.eval()

        epoch_loss: float = 0.0
        total_batches: int = 0

        # Iterate over the validation dataloader
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_dataloader):
                # Get batch data and move to device
                X: torch.Tensor = torch.stack(batch["input_ids"]).transpose(0, 1).to(device)

                # Inference the model output
                # Teacher forcing: use X[:, :-1] as input and X[:, 1:] as target
                output: torch.Tensor = model(X[:, :-1])
                reshaped_output: torch.Tensor = output.contiguous().view(-1, output.size(-1))
                reshaped_target: torch.Tensor = X[:, 1:].contiguous().view(-1)

                # Calculate the loss
                loss: torch.Tensor = self.criterion(reshaped_output, reshaped_target)

                # Update epoch loss and total batches
                epoch_loss += loss.item()
                total_batches += 1

        # Convert validation loss and batch count to tensor
        local_loss: torch.Tensor = torch.tensor(epoch_loss, device=device)
        local_batch_count: torch.Tensor = torch.tensor(total_batches, device=device)

        # If using DDP, reduce the loss and batch count across all processes
        if ddp == True:
            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_batch_count, op=dist.ReduceOp.SUM)

        # Calculate the average loss across all processes
        global_avg_loss: float = local_loss.item() / local_batch_count.item()

        # Autoregressive inference for samples
        if not self.ddp or (self.ddp and dist.is_initialized() and dist.get_rank() == 0):
            # Intialize the autoregressive generator
            generator: AutoregressiveGPTMini = AutoregressiveGPTMini(
                model=model,
                tokenizer=self.tokenizer,
                device=device,
            )
            # Prepare the inputs
            sample_inputs: Any = [sample["prompt"] for sample in self.sample_trace]
            sample_inputs = [[self.tokenizer.sos_id] + self.tokenizer.encode(sample, add_special_tokens=False) for sample in sample_inputs]
            sample_inputs = torch.tensor(sample_inputs, device=device)
            # Generate text for each sample
            generated_ids: torch.Tensor = generator.generate(sample_inputs, max_seq_len=self.dataset_configs.MAX_LENGTH)
            for idx, generated in enumerate(generated_ids):
                generated_text: str = self.tokenizer.decode(generated.tolist(), skip_special_tokens=True)
                self.sample_trace[idx]["generated"].append(generated_text)

        # Return the average loss for the epoch
        return global_avg_loss

    def _train(self: Any, model: Any, device: torch.device, ddp: bool) -> None:
        """Train the model."""

        # Necessary training configurations
        max_seq_len: int = self.dataset_configs.MAX_LENGTH
        epoches_num: int = self.training_configs.EPOCHES_NUM
        clip: float = self.training_configs.CLIP
        warmup: int = self.training_configs.WARMUP

        # Get actual model (unwrap DDP if necessary)
        actual_model: Optional[GPTMini] = None
        if ddp == True and isinstance(model, DDP):
            actual_model = model.module
        else:
            actual_model = model

        # Training loop
        for epoch in range(self.training_configs.EPOCHES_NUM):
            # Train for one epoch
            start_time: float = time.time()
            if ddp == True: self.train_dataloader.sampler.set_epoch(epoch)
            train_loss: float = self._epoch_train(model=model, device=device, ddp=ddp)
            valid_loss: float = self._epoch_valid(model=model, device=device, ddp=ddp)
            end_time: float = time.time()

            # Step the scheduler
            if epoch > warmup: self.scheduler.step(valid_loss)

            # Echo the training and validation loss
            if ddp == False or (ddp == True and dist.get_rank() == 0):
                elapsed_mins, elapsed_secs = epoch_time(start_time, end_time)
                print(f"[Training Trace] >>>"
                        f"Epoch: {epoch + 1:0>{len(str(epoches_num))}}/{epoches_num}, {elapsed_mins}m {elapsed_secs}s ::"
                        f"Train Loss: {train_loss:<8.4f}, Train PPL: {math.exp(train_loss):<8.4f} | "
                        f"Valid Loss: {valid_loss:<8.4f}, Valid PPL: {math.exp(valid_loss):<8.4f}")

            # Record the training and validation loss
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)

            # Save the best model based on validation loss
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss

                save_dir: str = self.training_configs.CHECKPOINT_DIR
                save_filename: str = f"epoch-{epoch:0>{len(str(epoches_num))}}-valid_loss-{valid_loss:0>7.4f}.pt"
                save_path: str = os.path.join(save_dir, save_filename)
                os.makedirs(save_dir, exist_ok=True)

                checkpoint: Dict[str, Any] = {
                    "epoch_idx": epoch,
                    "model_state_dict": actual_model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "dataset_configs": self.dataset_configs,
                    "tokenizer_configs": self.tokenizer_configs,
                    "model_configs": self.model_configs,
                    "training_configs": self.training_configs,
                }

                torch.save(checkpoint, save_path)

            if ddp == True: dist.barrier()

    def _train_with_ddp(self: Any) -> None:
        """Train the model with Distributed Data Parallel (DDP)."""
        assert self.ddp == True, "DDP must be enabled for this method."
        assert torch.cuda.is_available(), "DDP flag must be set to True for DDP training."
        assert torch.cuda.device_count() > 1, "This code requires at least two GPUs for DDP."

        # Setup devices and distributed environment
        world_size: int = int(os.environ["WORLD_SIZE"])
        rank: int = int(os.environ["RANK"])
        local_rank: int = int(os.environ["LOCAL_RANK"])

        torch.backends.cudnn.benchmark = True   # Enable benchmark mode for faster training
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size,
                                rank=rank, timeout=datetime.timedelta(seconds=3_000))
        torch.cuda.set_device(rank)
        dist.barrier()

        # Load dataset, tokenizer, and model
        self._load_dataset()
        self._load_tokenizer()
        self._load_model(device=torch.device(local_rank))

        # Wrap the model with DDP
        self.model = DDP(self.model, device_ids=[local_rank], output_device=torch.device(local_rank))

        # Initialize training components
        self._init_training_components(device=torch.device(local_rank))

        # Start training
        self._train(
            model=self.model,
            device=torch.device(local_rank),
            ddp=self.ddp,
        )

        # Cleanup distributed environment
        dist.barrier()      # Ensure all processes reach this point before cleanup
        cleanup()

    def _train_without_ddp(self: Any) -> None:
        """Train the model without Distributed Data Parallel (DDP)."""
        assert self.ddp == False, "DDP flag must be set to False for non-DDP training."
        assert torch.cuda.is_available(), "This code requires a GPU with CUDA support."

        # Setup device
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load Dataset, Tokenizer, and Model
        self._load_dataset()
        self._load_tokenizer()
        self._load_model(device=device)

        # Initialize training components
        self._init_training_components(device=device)

        # Start training
        self._train(
            model=self.model,
            device=device,
            ddp=self.ddp,
        )

    def _save_traces_data(self: Any) -> None:
        """Save training trace and sample trace data."""
        # Save training trace data
        save_dir: str = self.training_configs.TRAIN_TRACE_DIR
        os.makedirs(save_dir, exist_ok=True)

        train_trace: Dict[Optional[List[float]]] = {
            "train_losses": self.train_losses,
            "valid_losses": self.valid_losses,
            "sample_trace": self.sample_trace,
        }

        save_obj_by_pickle(save_dir, "train_trace.pkl", train_trace)

    def train(self: Any) -> None:
        """Train the model based on the DDP flag."""
        # Train the model
        if self.ddp == True: self._train_with_ddp()
        else: self._train_without_ddp()

        # Save training trace data
        self._save_traces_data()


if __name__ == "__main__":
    """Example usage of the Trainer class."""
    # Remove warnings
    import warnings
    warnings.filterwarnings("ignore")

    # initialize configurations
    dataset_configs: DatasetConfigs = DatasetConfigs()
    tokenizer_configs: TokenizerConfigs = TokenizerConfigs()
    model_configs: ModelConfigs = ModelConfigs()
    training_configs: TrainingConfigs = TrainingConfigs()

    # Create a Trainer instance
    trainer: Trainer = Trainer(
        dataset_configs=dataset_configs,
        tokenizer_configs=tokenizer_configs,
        model_configs=model_configs,
        training_configs=training_configs,
        ddp=False,  # Set to True for DDP training
        samples=["Hello, how are you?", "What is the meaning of life?"]
    )

    # Start training
    trainer.train()







