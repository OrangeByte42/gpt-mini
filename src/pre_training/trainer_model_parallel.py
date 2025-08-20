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
from src.models.gpt_mini_model_parallel import GPTMiniModelParallel
from src.utils.utils import count_parameters, epoch_time, cleanup, save_obj_by_pickle
from src.generator.autoregressive_gpt_mini_model_parallel import AutoregressiveGPTMiniModelParallel


class TrainerModelParallel:
    """Trainer class for pre-training with Model Parallelism support."""

    def __init__(self: Any,
                    dataset_configs: DatasetConfigs,
                    tokenizer_configs: TokenizerConfigs,
                    model_configs: ModelConfigs,
                    training_configs: TrainingConfigs,
                    device_map: Dict[str, torch.device],
                    ddp: bool,
                    samples: List[str]) -> None:
        """Initialize the Model Parallel Trainer with configurations.
        @param dataset_configs: Dataset configurations.
        @param tokenizer_configs: Tokenizer configurations.
        @param model_configs: Model configurations.
        @param training_configs: Training configurations.
        @param device_map: Dictionary mapping components to devices
        @param ddp: Whether to use Distributed Data Parallel (DDP).
        @param samples: Sample texts for inference after each epoch.
        """
        # Configurations
        self.dataset_configs: DatasetConfigs = dataset_configs
        self.tokenizer_configs: TokenizerConfigs = tokenizer_configs
        self.model_configs: ModelConfigs = model_configs
        self.training_configs: TrainingConfigs = training_configs

        # Model Parallel setup
        self.device_map: Dict[str, torch.device] = device_map
        self.primary_device: torch.device = list(device_map.values())[0]  # First device for loss computation

        # Dataset & Tokenizer & Model
        self.train_dataloader: Optional[DataLoader] = None
        self.valid_dataloader: Optional[DataLoader] = None
        self.tokenizer: Optional[BpeTokenizer] = None
        self.model: Optional[GPTMiniModelParallel] = None

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
            self.train_dataloader = DataLoader(train_dataset, batch_size=self.training_configs.BATCH_SIZE,
                                                            num_workers=self.training_configs.NUM_WORKERS)
            self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.training_configs.BATCH_SIZE,
                                                            num_workers=self.training_configs.NUM_WORKERS)

    def _load_tokenizer(self: Any) -> BpeTokenizer:
        """Load the tokenizer."""
        # Load the tokenizer from the tokenizer cache directory
        tokenizer: BpeTokenizer = BpeTokenizer(self.tokenizer_configs, self.dataset_configs)
        tokenizer.load_tokenizer()

        # Set the tokenizer to the instance variable
        self.tokenizer = tokenizer

        # Return the tokenizer
        return tokenizer

    def _load_model(self: Any) -> GPTMiniModelParallel:
        """Load the model with model parallelism."""
        # Instantiate the GPTMiniModelParallel model with configurations
        gpt_mini: GPTMiniModelParallel = GPTMiniModelParallel(
            vocab_size=self.tokenizer_configs.MAX_VOCAB_SIZE,
            max_seq_len=self.dataset_configs.MAX_LENGTH,
            pad_id=self.tokenizer.pad_id,
            num_layers=self.model_configs.NUM_LAYERS,
            d_model=self.model_configs.D_MODEL,
            num_heads=self.model_configs.NUM_HEADS,
            d_ff=self.model_configs.D_FF,
            drop_prob=self.model_configs.DROP_PROB,
            device_map=self.device_map,
        )

        # Initialize the model weights
        gpt_mini.init_weights()

        # Set the model to the instance variable
        self.model = gpt_mini

        # Echo the model parameters and device assignments
        if not self.ddp or (self.ddp and dist.is_initialized() and dist.get_rank() == 0):
            # Count parameters (simplified for model parallel case)
            total_params = sum(p.numel() for p in gpt_mini.parameters())
            trainable_params = sum(p.numel() for p in gpt_mini.parameters() if p.requires_grad)
            
            print(f"[Instantiating Model Parallel Model]:")
            print(f"Model Parameters: {total_params:,} ({total_params / 1e9:.4f} B)")
            print(f"Trainable Parameters: {trainable_params:,} ({trainable_params / 1e9:.4f} B)")
            
            # Print device assignments
            print(f"Device Assignments:")
            device_assignments = gpt_mini.arch.get_device_assignments()
            for component, device in device_assignments.items():
                print(f"  {component}: {device}")
            
            print(f"Instantiated GPTMiniModelParallel model successfully.", end="\n\n")

        return gpt_mini

    def _init_training_components(self: Any) -> None:
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

        # Create the loss criterion (place on primary device for loss computation)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id).to(self.primary_device)

    def _epoch_train(self: Any, model: Any, ddp: bool) -> float:
        """Train the model for one epoch with model parallelism."""
        # Set the model to training mode and create necessary variables
        model.train()

        epoch_loss: float = 0.0
        total_batches: int = 0

        # iterate over the training dataloader
        for batch in self.train_dataloader:
            # Get batch data and move to the embedding device (first device in pipeline)
            X: torch.Tensor = torch.stack(batch["input_ids"]).transpose(0, 1)
            X = model.move_batch_to_devices(X)

            # Inference the model output
            # Teacher forcing: use X[:, :-1] as input and X[:, 1:] as target
            output: torch.Tensor = model(X[:, :-1])
            
            # Move output to primary device for loss computation
            if output.device != self.primary_device:
                output = output.to(self.primary_device)
            
            # Prepare target and move to primary device
            target = X[:, 1:].to(self.primary_device)
            
            # Reshape for loss computation
            reshaped_output: torch.Tensor = output.contiguous().view(-1, output.size(-1))
            reshaped_target: torch.Tensor = target.contiguous().view(-1)

            # Calculate the loss
            loss: torch.Tensor = self.criterion(reshaped_output, reshaped_target)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping across all devices
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.training_configs.CLIP)
            self.optimizer.step()

            # Update epoch loss and total batches
            epoch_loss += loss.item()
            total_batches += 1

        # Convert training loss and batch count to tensor
        local_loss: torch.Tensor = torch.tensor(epoch_loss, device=self.primary_device)
        local_batch_count: torch.Tensor = torch.tensor(total_batches, device=self.primary_device)

        # If using DDP, reduce the loss and batch count across all processes
        if ddp == True:
            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_batch_count, op=dist.ReduceOp.SUM)

        # Calculate the average loss across all processes
        global_avg_loss: float = local_loss.item() / local_batch_count.item()

        # Return the average loss for the epoch
        return global_avg_loss

    def _epoch_valid(self: Any, model: Any, ddp: bool) -> float:
        """Validate the model for one epoch with model parallelism."""
        # Set the model to evaluation mode and create necessary variables
        model.eval()

        epoch_loss: float = 0.0
        total_batches: int = 0

        # Iterate over the validation dataloader
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_dataloader):
                # Get batch data and move to embedding device
                X: torch.Tensor = torch.stack(batch["input_ids"]).transpose(0, 1)
                X = model.move_batch_to_devices(X)

                # Inference the model output
                # Teacher forcing: use X[:, :-1] as input and X[:, 1:] as target
                output: torch.Tensor = model(X[:, :-1])
                
                # Move output to primary device for loss computation
                if output.device != self.primary_device:
                    output = output.to(self.primary_device)
                
                # Prepare target and move to primary device
                target = X[:, 1:].to(self.primary_device)
                
                # Reshape for loss computation
                reshaped_output: torch.Tensor = output.contiguous().view(-1, output.size(-1))
                reshaped_target: torch.Tensor = target.contiguous().view(-1)

                # Calculate the loss
                loss: torch.Tensor = self.criterion(reshaped_output, reshaped_target)

                # Update epoch loss and total batches
                epoch_loss += loss.item()
                total_batches += 1

        # Convert validation loss and batch count to tensor
        local_loss: torch.Tensor = torch.tensor(epoch_loss, device=self.primary_device)
        local_batch_count: torch.Tensor = torch.tensor(total_batches, device=self.primary_device)

        # If using DDP, reduce the loss and batch count across all processes
        if ddp == True:
            dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_batch_count, op=dist.ReduceOp.SUM)

        # Calculate the average loss across all processes
        global_avg_loss: float = local_loss.item() / local_batch_count.item()

        # Autoregressive inference for samples (only on rank 0 for DDP)
        if not self.ddp or (self.ddp and dist.is_initialized() and dist.get_rank() == 0):
            # Initialize the autoregressive generator
            generator: AutoregressiveGPTMiniModelParallel = AutoregressiveGPTMiniModelParallel(
                model=model,
                tokenizer=self.tokenizer,
                device_map=self.device_map,
            )
            # Prepare the inputs
            sample_inputs: Any = [sample["prompt"] for sample in self.sample_trace]
            sample_inputs = [[self.tokenizer.sos_id] + self.tokenizer.encode(sample, add_special_tokens=False) for sample in sample_inputs]
            sample_inputs = torch.tensor(sample_inputs, device=self.primary_device)
            # Generate text for each sample
            generated_ids: torch.Tensor = generator.generate(sample_inputs, max_seq_len=self.dataset_configs.MAX_LENGTH)

            # Decode and store generated text
            for idx, generated in enumerate(generated_ids):
                generated_text: str = self.tokenizer.decode(generated.tolist(), skip_special_tokens=True)
                self.sample_trace[idx]["generated"].append(generated_text)

        # Return the average loss for the epoch
        return global_avg_loss

    def _train(self: Any, model: Any, ddp: bool) -> None:
        """Train the model with model parallelism."""

        # Necessary training configurations
        max_seq_len: int = self.dataset_configs.MAX_LENGTH
        epoches_num: int = self.training_configs.EPOCHES_NUM
        clip: float = self.training_configs.CLIP
        warmup: int = self.training_configs.WARMUP

        # Training loop
        for epoch in range(self.training_configs.EPOCHES_NUM):
            # Train for one epoch
            start_time: float = time.time()
            if ddp == True: self.train_dataloader.sampler.set_epoch(epoch)
            train_loss: float = self._epoch_train(model=model, ddp=ddp)
            valid_loss: float = self._epoch_valid(model=model, ddp=ddp)
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
                
                # Print memory usage
                memory_info = model.get_memory_usage()
                for device_name, usage in memory_info.items():
                    print(f"  {device_name}: {usage['allocated']:.2f}GB allocated, {usage['cached']:.2f}GB cached")

            # Record the training and validation loss
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)

            # Save the best model based on validation loss
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss

                save_dir: str = self.training_configs.CHECKPOINT_DIR
                save_filename: str = f"epoch-{epoch:0>{len(str(epoches_num))}}-valid_loss-{valid_loss:0>7.4f}-model_parallel.pt"
                save_path: str = os.path.join(save_dir, save_filename)
                os.makedirs(save_dir, exist_ok=True)

                checkpoint: Dict[str, Any] = {
                    "epoch_idx": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "dataset_configs": self.dataset_configs,
                    "tokenizer_configs": self.tokenizer_configs,
                    "model_configs": self.model_configs,
                    "training_configs": self.training_configs,
                    "device_map": self.device_map,  # Save device mapping
                }

                torch.save(checkpoint, save_path)

            if ddp == True: dist.barrier()

        # Save training trace data
        self._save_traces_data()

    def _train_without_ddp(self: Any) -> None:
        """Train the model without Distributed Data Parallel (DDP) using Model Parallelism."""
        assert self.ddp == False, "DDP flag must be set to False for non-DDP training."
        assert torch.cuda.is_available(), "This code requires GPUs with CUDA support."

        # Load Dataset, Tokenizer, and Model
        self._load_dataset()
        self._load_tokenizer()
        self._load_model()

        # Initialize training components
        self._init_training_components()

        # Start training
        self._train(
            model=self.model,
            ddp=self.ddp,
        )

    def _save_traces_data(self: Any) -> None:
        """Save training trace and sample trace data."""
        # Save training trace data
        train_trace_save_dir: str = self.training_configs.TRAIN_TRACE_DIR
        sample_trace_save_dir: str = self.training_configs.SAMPLE_TRACE_DIR

        os.makedirs(train_trace_save_dir, exist_ok=True)
        os.makedirs(sample_trace_save_dir, exist_ok=True)

        # Save training losses
        train_trace_path: str = os.path.join(train_trace_save_dir, "train_losses.pkl")
        valid_trace_path: str = os.path.join(train_trace_save_dir, "valid_losses.pkl")
        sample_trace_path: str = os.path.join(sample_trace_save_dir, "sample_traces.pkl")

        save_obj_by_pickle(self.train_losses, train_trace_path)
        save_obj_by_pickle(self.valid_losses, valid_trace_path)
        save_obj_by_pickle(self.sample_trace, sample_trace_path)

    def train(self: Any) -> None:
        """Train the model based on the DDP flag."""
        if self.ddp:
            self._train_with_ddp()
        else:
            self._train_without_ddp()

    def _train_with_ddp(self: Any) -> None:
        """Train the model with Distributed Data Parallel (DDP) and Model Parallelism."""
        # Note: DDP + Model Parallelism is complex and requires careful setup
        # For now, we recommend using either DDP OR Model Parallelism, not both
        raise NotImplementedError("DDP + Model Parallelism is not yet implemented. Please use model_parallel=True and ddp=False for now.")


def create_device_map_auto(num_layers: int, available_devices: List[torch.device]) -> Dict[str, torch.device]:
    """Automatically create a device map for model parallelism
    @param num_layers: Number of transformer layers
    @param available_devices: List of available CUDA devices
    @return: Dictionary mapping component names to devices
    """
    if len(available_devices) == 0:
        raise ValueError("No available devices provided")
    
    device_map = {}
    
    # Embedding goes to first device
    device_map['embedding'] = available_devices[0]
    
    # Distribute layers across devices
    layers_per_device = num_layers // len(available_devices)
    remainder = num_layers % len(available_devices)
    
    layer_idx = 0
    for device_idx, device in enumerate(available_devices):
        # Calculate how many layers for this device
        num_layers_this_device = layers_per_device
        if device_idx < remainder:
            num_layers_this_device += 1
        
        # Assign layers to this device
        for _ in range(num_layers_this_device):
            device_map[f'layer_{layer_idx}'] = device
            layer_idx += 1
    
    # Output layer goes to last device
    device_map['output'] = available_devices[-1]
    
    return device_map


if __name__ == "__main__":
    """Example usage of the Model Parallel Trainer class."""
    # Remove warnings
    import warnings
    warnings.filterwarnings("ignore")

    # Initialize configurations
    dataset_configs: DatasetConfigs = DatasetConfigs()
    tokenizer_configs: TokenizerConfigs = TokenizerConfigs()
    model_configs: ModelConfigs = ModelConfigs()
    training_configs: TrainingConfigs = TrainingConfigs()

    # Setup model parallelism device map
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Found {num_gpus} GPU(s) available")
        
        # Create list of available devices
        available_devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)]
        
        # Create device map
        device_map = create_device_map_auto(
            num_layers=model_configs.NUM_LAYERS,
            available_devices=available_devices
        )
        
        print("Device Map:")
        for component, device in device_map.items():
            print(f"  {component}: {device}")
    else:
        print("CUDA not available, using CPU")
        device_map = {
            'embedding': torch.device('cpu'),
            'output': torch.device('cpu')
        }
        for i in range(model_configs.NUM_LAYERS):
            device_map[f'layer_{i}'] = torch.device('cpu')

    # Create a Model Parallel Trainer instance
    trainer: TrainerModelParallel = TrainerModelParallel(
        dataset_configs=dataset_configs,
        tokenizer_configs=tokenizer_configs,
        model_configs=model_configs,
        training_configs=training_configs,
        device_map=device_map,
        ddp=False,  # Model parallelism without DDP for now
        samples=["Hello, how are you?", "What is the meaning of life?"]
    )

    # Start training
    trainer.train()
