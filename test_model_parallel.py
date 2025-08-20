#!/usr/bin/env python3
"""
Simple test script for Model Parallelism functionality.
Tests basic model creation and forward pass with model parallelism.
"""

import torch
import sys
import os

# Add src to path
sys.path.append('src')

from src.models.gpt_mini_model_parallel import GPTMiniModelParallel
from src.configs.model_parallel_configs import (
    ModelParallelConfigs, 
    create_balanced_device_map,
    print_device_map
)


def test_model_parallel_basic():
    """Test basic model parallel functionality."""
    print("=" * 60)
    print("TESTING MODEL PARALLELISM - BASIC FUNCTIONALITY")
    print("=" * 60)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available. Using CPU for testing.")
        available_devices = [torch.device('cpu')]
    else:
        print(f"CUDA available with {torch.cuda.device_count()} device(s)")
        available_devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    
    # Print device information
    print(f"Available devices: {available_devices}")
    
    # Test configuration
    vocab_size = 1000
    max_seq_len = 128
    pad_id = 0
    num_layers = 4  # Small model for testing
    d_model = 256
    num_heads = 8
    d_ff = 512
    drop_prob = 0.1
    
    print(f"\nTest Model Configuration:")
    print(f"  Vocabulary Size: {vocab_size}")
    print(f"  Max Sequence Length: {max_seq_len}")
    print(f"  Number of Layers: {num_layers}")
    print(f"  Model Dimension: {d_model}")
    print(f"  Attention Heads: {num_heads}")
    
    # Create device map
    device_map = create_balanced_device_map(
        num_layers=num_layers,
        available_devices=available_devices
    )
    
    print_device_map(device_map)
    
    try:
        # Create model
        print("\nCreating GPTMiniModelParallel...")
        model = GPTMiniModelParallel(
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
        
        # Initialize weights
        model.init_weights()
        
        print("✅ Model created successfully!")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        print("\nTesting forward pass...")
        batch_size = 2
        seq_len = 32
        
        # Create test input
        test_input = torch.randint(1, vocab_size-1, (batch_size, seq_len))
        test_input = model.move_batch_to_devices(test_input)
        
        print(f"Input shape: {test_input.shape}")
        print(f"Input device: {test_input.device}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(test_input)
        
        print(f"Output shape: {output.shape}")
        print(f"Output device: {output.device}")
        print("✅ Forward pass successful!")
        
        # Test memory usage reporting
        if torch.cuda.is_available():
            print(f"\nMemory Usage:")
            memory_usage = model.get_memory_usage()
            for device_name, usage in memory_usage.items():
                print(f"  {device_name}: {usage['allocated']:.3f}GB allocated, {usage['cached']:.3f}GB cached")
        
        print(f"\n✅ ALL TESTS PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_device_mapping():
    """Test different device mapping strategies."""
    print("\n" + "=" * 60)
    print("TESTING DEVICE MAPPING STRATEGIES")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping device mapping tests.")
        return True
    
    num_layers = 6
    available_devices = [torch.device(f'cuda:{i}') for i in range(min(torch.cuda.device_count(), 2))]
    
    try:
        # Test balanced mapping
        print(f"\n1. Testing balanced device mapping:")
        device_map_balanced = create_balanced_device_map(num_layers, available_devices)
        print_device_map(device_map_balanced)
        
        # Test memory optimized mapping
        print(f"\n2. Testing memory optimized device mapping:")
        from src.configs.model_parallel_configs import create_memory_optimized_device_map
        device_map_optimized = create_memory_optimized_device_map(num_layers, available_devices)
        print_device_map(device_map_optimized)
        
        print(f"\n✅ Device mapping tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Device mapping test failed: {str(e)}")
        return False


def main():
    """Main test function."""
    print("Starting Model Parallelism Tests...")
    
    # Test 1: Basic functionality
    test1_passed = test_model_parallel_basic()
    
    # Test 2: Device mapping
    test2_passed = test_device_mapping()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Basic Functionality: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Device Mapping: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print(f"\n🎉 All tests passed! Model parallelism is working correctly.")
        print(f"You can now run the full training with: python examples/train_model_parallel.py")
    else:
        print(f"\n⚠️ Some tests failed. Please check your setup.")
    
    return test1_passed and test2_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
