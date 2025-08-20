import torch
from torch import nn
from typing import Any, Dict, List

from src.models.blocks.decoder_layer_model_parallel import DecoderLayerModelParallel
from src.models.layers.embedding import TransformerEmbedding


class DecoderOnlyArchModelParallel(nn.Module):
    """Decoder-Only Architecture with Model Parallelism"""
    
    def __init__(self: Any, vocab_size: int, max_seq_len: int, pad_id: int,
                num_layers: int, d_model: int, num_heads: int, d_ff: int,
                drop_prob: float, device_map: Dict[str, torch.device]) -> None:
        """constructor
        @param vocab_size: size of the decoder vocabulary
        @param max_seq_len: maximum sequence length
        @param pad_id: index of the padding token which decided by tokenizer
        @param num_layers: number of decoder layers
        @param d_model: dimension of the model
        @param num_heads: number of attention heads
        @param d_ff: dimension of the feed-forward network
        @param drop_prob: dropout probability
        @param device_map: Dictionary mapping layer names to devices
        """
        super(DecoderOnlyArchModelParallel, self).__init__()
        
        self.device_map: Dict[str, torch.device] = device_map
        self.num_layers: int = num_layers
        
        # Embedding layer (usually on first GPU)
        embedding_device = device_map.get('embedding', list(device_map.values())[0])
        self.embedding: TransformerEmbedding = TransformerEmbedding(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            pad_id=pad_id,
            d_model=d_model,
            drop_prob=drop_prob,
            device=embedding_device
        )
        
        # Distribute decoder layers across GPUs
        self.layers: nn.ModuleList = nn.ModuleList()
        for layer_idx in range(num_layers):
            # Calculate which device this layer should be on
            device_key = f'layer_{layer_idx}'
            if device_key in device_map:
                layer_device = device_map[device_key]
            else:
                # Distribute layers evenly across available devices
                available_devices = [d for k, d in device_map.items() if k != 'embedding' and k != 'output']
                if available_devices:
                    layer_device = available_devices[layer_idx % len(available_devices)]
                else:
                    layer_device = embedding_device
            
            layer = DecoderLayerModelParallel(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                drop_prob=drop_prob,
                device=layer_device
            )
            self.layers.append(layer)
        
        # Output projection layer (usually on last GPU)
        output_device = device_map.get('output', list(device_map.values())[-1])
        self.fc: nn.Linear = nn.Linear(d_model, vocab_size).to(output_device)
        
        # Store device assignments for debugging
        self.layer_devices: List[torch.device] = [layer.device for layer in self.layers]

    def forward(self: Any, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """apply decoder with model parallelism
        @param X: input tensor of shape (batch_size, seq_len)
        @param mask: mask tensor for the target sequence of shape (batch_size, 1, seq_len, seq_len)
        @return: output tensor of shape (batch_size, seq_len, vocab_size)
        """
        # Embedding and Positional Encoding (first GPU)
        X = self.embedding(X)
        
        # Move mask to the same device as current X for each layer
        current_device = X.device
        
        # Pass through each decoder layer with device transfers
        for layer_idx, layer in enumerate(self.layers):
            target_device = self.layer_devices[layer_idx]
            
            # Move data to target device if necessary
            if X.device != target_device:
                X = X.to(target_device)
                mask = mask.to(target_device)
            
            # Apply the layer
            X = layer(X, mask)
        
        # Move to output device for final projection
        output_device = next(self.fc.parameters()).device
        if X.device != output_device:
            X = X.to(output_device)
        
        # Apply final linear layer
        output: torch.Tensor = self.fc(X)
        
        return output

    def get_device_assignments(self: Any) -> Dict[str, str]:
        """Get device assignments for all components
        @return: Dictionary mapping component names to device names
        """
        assignments = {
            'embedding': str(next(self.embedding.parameters()).device),
            'output': str(next(self.fc.parameters()).device)
        }
        
        for i, device in enumerate(self.layer_devices):
            assignments[f'layer_{i}'] = str(device)
        
        return assignments

    def memory_summary(self: Any) -> Dict[str, Dict[str, float]]:
        """Get memory usage summary for each device
        @return: Memory usage information
        """
        memory_info = {}
        for device_name, device in self.device_map.items():
            if device.type == 'cuda':
                memory_info[device_name] = {
                    'allocated_GB': torch.cuda.memory_allocated(device) / 1024**3,
                    'reserved_GB': torch.cuda.memory_reserved(device) / 1024**3
                }
        return memory_info
