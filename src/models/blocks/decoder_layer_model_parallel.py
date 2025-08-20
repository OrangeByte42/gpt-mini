import torch
from torch import nn
from typing import Any, Dict

from src.models.layers.layernorm import LayerNorm
from src.models.layers.multihead_attention import MultiHeadAttention
from src.models.layers.position_wise_feed_forward import PositionWiseFeedForward


class DecoderLayerModelParallel(nn.Module):
    """Decoder Layer with Model Parallelism Support"""
    
    def __init__(self: Any, d_model: int, num_heads: int, d_ff: int, drop_prob: float, device: torch.device) -> None:
        """constructor
        @param d_model: dimension of the model
        @param num_heads: number of attention heads
        @param d_ff: dimension of the feed-forward network
        @param drop_prob: dropout probability
        @param device: device to place this layer on
        @return: None
        """
        super(DecoderLayerModelParallel, self).__init__()

        self.device: torch.device = device

        # Sublayer-01: Multi-Head Attention
        self.mha: MultiHeadAttention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.ln1: LayerNorm = LayerNorm(d_model=d_model)
        self.dropout1: nn.Dropout = nn.Dropout(p=drop_prob)

        # Sublayer-02: Position-wise Feed-Forward Network
        self.ffn: PositionWiseFeedForward = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, drop_prob=drop_prob)
        self.ln2: LayerNorm = LayerNorm(d_model=d_model)
        self.dropout2: nn.Dropout = nn.Dropout(p=drop_prob)

        # Move all components to the specified device
        self.to(device)

    def forward(self: Any, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """apply decoder layer with automatic device handling
        @param X: input tensor of shape (batch_size, seq_len, d_model)
        @param mask: mask tensor for the target sequence (self-attention) of shape (batch_size, 1, seq_len, seq_len)
        @return: output tensor of shape (batch_size, seq_len, d_model)
        """
        # Ensure inputs are on the correct device
        X = X.to(self.device)
        mask = mask.to(self.device)

        # Sublayer 1: Self-attention with residual connection and layer norm
        residual_X: torch.Tensor = X
        X: torch.Tensor = self.mha(Q=X, K=X, V=X, mask=mask)
        X = self.dropout1(X)
        X = self.ln1(X + residual_X)

        # Sublayer 2: Feed-forward with residual connection and layer norm
        residual_X = X
        X = self.ffn(X)
        X = self.dropout2(X)
        X = self.ln2(X + residual_X)

        return X

    def get_device(self: Any) -> torch.device:
        """Get the device this layer is on
        @return: The device this layer is placed on
        """
        return self.device

    def memory_usage(self: Any) -> Dict[str, float]:
        """Get memory usage of this layer
        @return: Dictionary with memory usage information
        """
        if self.device.type == 'cuda':
            return {
                'allocated_MB': torch.cuda.memory_allocated(self.device) / 1024**2,
                'reserved_MB': torch.cuda.memory_reserved(self.device) / 1024**2
            }
        return {'allocated_MB': 0.0, 'reserved_MB': 0.0}
