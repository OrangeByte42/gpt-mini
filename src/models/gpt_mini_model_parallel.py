import torch
from torch import nn
from typing import Any, List, Dict, Optional
from torch.nn.parallel import DistributedDataParallel as DDP

from src.models.components.decoder_only_arch_model_parallel import DecoderOnlyArchModelParallel


class GPTMiniModelParallel(nn.Module):
    """GPT mini with Model Parallel support for larger batch sizes"""
    
    def __init__(self: Any, vocab_size: int, max_seq_len: int, pad_id: int,
                    num_layers: int, d_model: int, num_heads: int, d_ff: int,
                    drop_prob: float, device_map: Dict[str, torch.device]) -> None:
        """GPT mini model constructor with model parallel support
        @param vocab_size: Size of the vocabulary
        @param max_seq_len: Maximum sequence length
        @param pad_id: Padding token ID
        @param num_layers: Number of transformer layers
        @param d_model: Dimension of the model
        @param num_heads: Number of attention heads
        @param d_ff: Dimension of the feed-forward network
        @param drop_prob: Dropout probability
        @param device_map: Dictionary mapping component names to devices
        """
        super(GPTMiniModelParallel, self).__init__()

        self.pad_id: int = pad_id
        self.device_map: Dict[str, torch.device] = device_map
        self.num_layers: int = num_layers
        
        # Create model parallel architecture
        self.arch: DecoderOnlyArchModelParallel = DecoderOnlyArchModelParallel(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            pad_id=pad_id,
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            drop_prob=drop_prob,
            device_map=device_map
        )

    def make_mask(self: Any, X: torch.Tensor) -> torch.Tensor:
        """Create attention mask for the input tensor (combines padding mask and causal mask)
        @param X: Input tensor of shape (batch_size, seq_len)
        @return: Attention mask of shape (batch_size, 1, seq_len, seq_len)
        """
        # Create padding mask
        padding_mask: torch.Tensor = (X != self.pad_id).unsqueeze(1).unsqueeze(1)

        # Create subsequent mask (causal mask)
        seq_len: int = X.shape[1]
        causal_mask: torch.Tensor = torch.tril(torch.ones(seq_len, seq_len, device=X.device, dtype=torch.bool))

        # Combine masks
        mask: torch.Tensor = (padding_mask & causal_mask)

        return mask

    def forward(self: Any, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GPT-2 mini model with model parallelism
        @param X: Input tensor of shape (batch_size, seq_len)
        @return: Output tensor of shape (batch_size, seq_len, vocab_size)
        """
        # Create attention mask
        mask: torch.Tensor = self.make_mask(X)

        # Pass through the model parallel architecture
        output: torch.Tensor = self.arch(X, mask)

        return output

    def init_weights(self: Any) -> None:
        """Initialize model weights of the GPT-2 mini model"""
        # Define a function to initialize weights
        def init_weights(module: nn.Module) -> None:
            """Initialize weights for a module"""
            if hasattr(module, 'weight') and module.weight.dim() > 1:
                nn.init.kaiming_uniform_(module.weight.data)

        # Apply the initialization function to all modules in the model
        self.apply(init_weights)

    def get_memory_usage(self: Any) -> Dict[str, Dict[str, float]]:
        """Get memory usage for each device
        @return: Dictionary with memory usage information for each device
        """
        memory_info = {}
        for device_name, device in self.device_map.items():
            if device.type == 'cuda':
                memory_info[device_name] = {
                    'allocated': torch.cuda.memory_allocated(device) / 1024**3,  # GB
                    'cached': torch.cuda.memory_reserved(device) / 1024**3,      # GB
                }
        return memory_info

    def move_batch_to_devices(self: Any, batch: torch.Tensor) -> torch.Tensor:
        """Move input batch to the appropriate device for embedding layer
        @param batch: Input batch tensor
        @return: Batch tensor moved to embedding device
        """
        embedding_device = self.device_map.get('embedding', list(self.device_map.values())[0])
        return batch.to(embedding_device)
