def create_balanced_memory_device_map(num_layers: int,
                                     available_devices: List[torch.device],
                                     layer_memory_weights: List[float] = None) -> Dict[str, torch.device]:
    """Create a balanced memory device map with improved distribution strategy."""
    if not torch.cuda.is_available():
        return create_balanced_device_map(num_layers, available_devices)
    
    num_devices = len(available_devices)
    device_map = {}
    device_loads = [0.0] * num_devices
    
    # Get memory information for each device
    device_memory = []
    for device in available_devices:
        if device.type == 'cuda':
            props = torch.cuda.get_device_properties(device)
            device_memory.append(props.total_memory)
        else:
            device_memory.append(float('inf'))
    
    if layer_memory_weights is None:
        layer_memory_weights = [1.0] * num_layers
    
    # Memory weights based on actual GPT model components:
    # Embedding: vocab_size * hidden_dim parameters + gradients
    # Each transformer layer: ~4 * hidden_dim^2 parameters + activations
    # Output: hidden_dim * vocab_size (shared with embedding in some models)
    
    # For GPT-mini (vocab=50K, hidden=768):
    # Embedding: ~38M parameters → high memory
    # Each transformer layer: ~2.3M parameters
    # Output layer: ~38M parameters (if not shared)
    
    embedding_weight = 2.5  # Reduced from 3.0
    output_weight = 2.2     # Reduced from 2.5
    
    # Strategy: Better distribution
    # 1. Assign embedding to GPU with most available memory initially
    # 2. Assign output to a different GPU
    # 3. Distribute transformer layers to balance total load
    
    # Embedding on least loaded device initially  
    embedding_device_idx = 0  # Start with GPU 0
    device_map['embedding'] = available_devices[embedding_device_idx]
    device_loads[embedding_device_idx] += embedding_weight
    
    # Output on GPU 2 (middle) for better balance
    if num_devices >= 3:
        output_device_idx = 2
    elif num_devices >= 2:
        output_device_idx = 1
    else:
        output_device_idx = 0
    
    device_map['output'] = available_devices[output_device_idx]
    device_loads[output_device_idx] += output_weight
    
    # Distribute transformer layers to balance loads
    for layer_idx in range(num_layers):
        # Find device with minimum relative load
        relative_loads = [load / memory for load, memory in zip(device_loads, device_memory)]
        min_device_idx = relative_loads.index(min(relative_loads))
        
        device_map[f'layer_{layer_idx}'] = available_devices[min_device_idx]
        device_loads[min_device_idx] += layer_memory_weights[layer_idx]
    
    return device_map
