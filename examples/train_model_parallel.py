#!/usr/bin/env python3
"""
Example script for training GPT-mini with Model Parallelism.

This script demonstrates how to:
1. Set up model parallelism across multiple GPUs
2. Configure device mapping for optimal memory usage
3. Train with larger batch sizes using model parallelism
4. Monitor memory usage across devices
"""

import sys
import os
import warnings
import torch

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.configs.configs import DatasetConfigs, TokenizerConfigs, ModelConfigs, TrainingConfigs
from src.configs.model_parallel_configs import (
    ModelParallelConfigs, 
    create_balanced_device_map, 
    create_memory_optimized_device_map,
    print_device_map,
    get_recommended_batch_size
)
from src.pre_training.trainer_model_parallel import TrainerModelParallel


def main():
    """Main function for model parallel training."""
    # Remove warnings for cleaner output
    warnings.filterwarnings("ignore")
    
    print("=" * 80)
    print("GPT-MINI MODEL PARALLEL TRAINING")
    print("=" * 80)
    
    # Initialize configurations
    dataset_configs = DatasetConfigs()
    tokenizer_configs = TokenizerConfigs()
    model_configs = ModelConfigs()
    training_configs = TrainingConfigs()
    model_parallel_configs = ModelParallelConfigs()
    
    # Print system information
    print(f"\nSystem Information:")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    else:
        print("Warning: CUDA not available. Model parallelism will be limited.")
    
    # Get device information
    device_info = model_parallel_configs.get_device_info()
    available_devices = model_parallel_configs.get_available_devices()
    
    print(f"\nAvailable Devices for Model Parallelism:")
    for device in available_devices:
        print(f"  - {device}")
    
    # Create device map
    print(f"\nCreating Device Map...")
    print(f"Model has {model_configs.NUM_LAYERS} transformer layers")
    
    # Choose device mapping strategy
    if len(available_devices) > 1:
        print("Using memory-optimized device mapping")
        device_map = create_memory_optimized_device_map(
            num_layers=model_configs.NUM_LAYERS,
            available_devices=available_devices
        )
    else:
        print("Using balanced device mapping (single device)")
        device_map = create_balanced_device_map(
            num_layers=model_configs.NUM_LAYERS,
            available_devices=available_devices
        )
    
    # Print device assignments
    print_device_map(device_map, model_parallel_configs)
    
    # Adjust batch size for model parallelism
    original_batch_size = training_configs.BATCH_SIZE
    recommended_batch_size = get_recommended_batch_size(device_map, model_size_gb=1.0)
    
    print(f"\nBatch Size Configuration:")
    print(f"  Original batch size: {original_batch_size}")
    print(f"  Recommended batch size: {recommended_batch_size}")
    
    # Use larger batch size for model parallelism (can handle more due to distributed memory)
    model_parallel_batch_size = max(original_batch_size * 4, recommended_batch_size)
    training_configs.BATCH_SIZE = model_parallel_batch_size
    
    print(f"  Selected batch size: {training_configs.BATCH_SIZE}")
    
    # Create trainer
    print(f"\nInitializing Model Parallel Trainer...")
    
    # Sample prompts for generation testing
    sample_prompts = [
        "Hello, how are you?",
        "What is the meaning of life?",
        "In a distant galaxy,",
        "The future of artificial intelligence",
        "Once upon a time in a magical forest,"
    ]
    
    trainer = TrainerModelParallel(
        dataset_configs=dataset_configs,
        tokenizer_configs=tokenizer_configs,
        model_configs=model_configs,
        training_configs=training_configs,
        device_map=device_map,
        ddp=False,  # Model parallelism without DDP
        samples=sample_prompts
    )
    
    # Print training configuration
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {training_configs.EPOCHES_NUM}")
    print(f"  Learning Rate: {training_configs.INIT_LR}")
    print(f"  Batch Size: {training_configs.BATCH_SIZE}")
    print(f"  Model Parallel: Enabled")
    print(f"  DDP: Disabled (using Model Parallelism instead)")
    
    print(f"\nModel Configuration:")
    print(f"  Layers: {model_configs.NUM_LAYERS}")
    print(f"  Model Dimension: {model_configs.D_MODEL}")
    print(f"  Attention Heads: {model_configs.NUM_HEADS}")
    print(f"  Feed-Forward Dimension: {model_configs.D_FF}")
    
    # Start training
    print(f"\n" + "=" * 80)
    print("STARTING MODEL PARALLEL TRAINING")
    print("=" * 80)
    
    try:
        trainer.train()
        print(f"\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        print("This might be due to:")
        print("1. Insufficient GPU memory")
        print("2. Missing training data")
        print("3. Configuration issues")
        print("\nTry reducing the batch size or model size.")
        raise
    
    # Print final memory usage if available
    if hasattr(trainer, 'model') and trainer.model:
        print(f"\nFinal Memory Usage:")
        memory_usage = trainer.model.get_memory_usage()
        for device_name, usage in memory_usage.items():
            print(f"  {device_name}: {usage['allocated']:.2f}GB allocated, {usage['cached']:.2f}GB cached")


if __name__ == "__main__":
    main()
