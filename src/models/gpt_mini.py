import torch
from torch import nn
from typing import Any

from src.models.components.decoder_only_arch import DecoderOnlyArch


class GPTMini(nn.Module):
    """GPT mini"""
    def __init__(self: Any, vocab_size: int, max_seq_len: int, pad_id: int,
                    num_layers: int, d_model: int, num_heads: int, d_ff: int,
                    drop_prob: float, device: torch.device) -> None:
        """GPT mini model constructor
        @param vocab_size: Size of the vocabulary
        @param max_seq_len: Maximum sequence length
        @param pad_id: Padding token ID
        @param num_layers: Number of transformer layers
        @param d_model: Dimension of the model
        @param num_heads: Number of attention heads
        @param d_ff: Dimension of the feed-forward network
        @param drop_prob: Dropout probability
        @param device: Device to run the model on
        """
        super(GPTMini, self).__init__()

        self.pad_id: int = pad_id

        self.device: torch.device = device

        self.arch: DecoderOnlyArch = DecoderOnlyArch(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            pad_id=pad_id,
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            drop_prob=drop_prob,
            device=device
        )

        self.to(device)

    def make_mask(self: Any, X: torch.Tensor) -> torch.Tensor:
        """Create attention mask for the input tensor (combines padding mask and causal mask)
        @param X: Input tensor of shape (batch_size, seq_len)
        @return: Attention mask of shape (batch_size, 1, seq_len, seq_len), shape[1] == 1 for mha_heads broadcast
        """
        # Create padding mask
        padding_mask: torch.Tensor = (X != self.pad_id).unsqueeze(1).unsqueeze(1)

        # Create subsequent mask (causal mask)
        seq_len: int = X.shape[1]
        causal_mask: torch.Tensor = torch.tril(torch.ones(seq_len, seq_len, device=X.device, dtype=torch.bool))

        # Combine masks
        mask: torch.Tensor = (padding_mask & causal_mask)

        # Move to device
        mask = mask.to(self.device)

        # Return the mask
        return mask

    def forward(self: Any, X: torch.Tensor) -> torch.Tensor:
        """Forward pass of the GPT-2 mini model
        @param X: Input tensor of shape (batch_size, seq_len)
        @return: Output tensor of shape (batch_size, seq_len, vocab_size)
        """
        # Create attention mask
        mask: torch.Tensor = self.make_mask(X)

        # Pass through the architecture
        output: torch.Tensor = self.arch(X, mask)

        # Return the output
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

