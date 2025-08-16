import torch
from torch import nn
from typing import Any

from src.models.blocks.decoder_layer import DecoderLayer
from src.models.layers.embedding import TransformerEmbedding


class DecoderOnlyArch(nn.Module):
    """Decoder-Only Architecture"""
    def __init__(self: Any, vocab_size: int, max_seq_len: int, padding_id: int,
                num_layers: int, d_model: int, num_heads: int, d_ff: int,
                drop_prob: float, device: torch.device) -> None:
        """constructor
        @param vocab_size: size of the decoder vocabulary
        @param max_seq_len: maximum sequence length
        @param padding_id: index of the padding token which decided by tokenizer
        @param num_layers: number of decoder layers
        @param d_model: dimension of the model
        @param num_heads: number of attention heads
        @param d_ff: dimension of the feed-forward network
        @param drop_prob: dropout probability
        @param device: device to use for the decoder
        """
        super(DecoderOnlyArch, self).__init__()

        self.embedding: TransformerEmbedding = TransformerEmbedding(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            padding_id=padding_id,
            d_model=d_model,
            drop_prob=drop_prob,
            device=device
        )

        self.layers: nn.ModuleList = nn.ModuleList([DecoderLayer(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            drop_prob=drop_prob
        ) for _ in range(num_layers)])

        self.fc: nn.Linear = nn.Linear(d_model, vocab_size)

        self.to(device)     # Move entire decoder to device

    def forward(self: Any, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """apply decoder
        @param X: input tensor of shape (batch_size, seq_len)
        @param encoder_output_X: output tensor from the encoder of shape (batch_size, seq_len, d_model)
        @param src_mask: mask tensor for the source sequence of shape (batch_size, 1, 1, seq_len)
        @param mask: mask tensor for the target sequence of shape (batch_size, 1, seq_len, seq_len)
        @return: output tensor of shape (batch_size, seq_len, vocab_size)
        """
        # Embedding and Positional Encoding
        X = self.embedding(X)
        # Pass through each decoder layer
        for layer in self.layers:
            X = layer(X, mask)
        output: torch.Tensor = self.fc(X)
        # Return logits
        return output

