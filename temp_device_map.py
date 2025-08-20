def create_balanced_memory_device_map(num_layers: int,
                                     available_devices: List[torch.device],
                                     layer_memory_weights: List[float] = None) -> Dict[str, torch.device]:
    """Create a balanced memory device map with better distribution strategy."""
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
    
    # Embedding layer has high memory usage - put on device with most memory
    embedding_weight = 3.0
    output_weight = 2.5
    
    # Distribute embedding and output to different devices
    embedding_device_idx = device_memory.index(max(device_memory))
    device_map['embedding'] = available_devices[embedding_device_idx]  
    device_loads[embedding_device_idx] += embedding_weight
    
    # Put output on different device if possible
    if num_devices > 1:
        output_device_idx = (embedding_device_idx + num_devices // 2) % num_devices
    else:
        output_device_idx = 0
    device_map['output'] = available_devices[output_device_idx]
    device_loads[output_device_idx] += output_weight
    
    # Distribute layers evenly
    for layer_idx in range(num_layers):
        relative_loads = [load / memory for load, memory in zip(device_loads, device_memory)]
        min_device_idx = relative_loads.index(min(relative_loads))
        
        device_map[f'layer_{layer_idx}'] = available_devices[min_device_idx]
        device_loads[min_device_idx] += layer_memory_weights[layer_idx]
    
    return device_map
