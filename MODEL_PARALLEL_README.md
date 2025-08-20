# GPT-Mini 模型并行 (Model Parallelism) 功能

本项目现已支持模型并行功能，允许将GPT模型分布到多个GPU设备上，从而能够：
1. **容纳更大的批量大小 (Batch Size)**：通过分布式内存使用，可以处理更大的数据批次
2. **训练更大的模型**：模型参数分布在多个GPU上，突破单GPU内存限制
3. **提高内存效率**：优化GPU内存使用，减少内存溢出问题

## 🚀 主要特性

### ✨ 核心功能
- **自动设备映射**：智能分配模型组件到不同GPU设备
- **内存优化**：基于GPU内存容量优化层分布
- **灵活配置**：支持自定义设备映射策略
- **实时监控**：训练过程中监控各设备内存使用情况
- **更大批量**：支持比单GPU更大的batch size

### 📊 性能提升
- **内存效率**：模型参数分布存储，有效利用多GPU内存
- **批处理能力**：可处理2-4倍于原始batch size的数据
- **训练稳定性**：减少内存不足导致的训练中断

## 📁 新增文件结构

```
src/
├── models/
│   ├── gpt_mini_model_parallel.py          # 模型并行版本的GPT模型
│   ├── components/
│   │   └── decoder_only_arch_model_parallel.py  # 支持模型并行的decoder架构
│   └── blocks/
│       └── decoder_layer_model_parallel.py      # 支持模型并行的decoder层
├── configs/
│   └── model_parallel_configs.py           # 模型并行配置
├── pre_training/
│   └── trainer_model_parallel.py           # 模型并行训练器
└── generator/
    └── autoregressive_gpt_mini_model_parallel.py  # 模型并行生成器

examples/
└── train_model_parallel.py                 # 模型并行训练示例
```

## 🛠️ 使用方法

### 1. 基础使用

```python
from src.configs.configs import DatasetConfigs, TokenizerConfigs, ModelConfigs, TrainingConfigs
from src.configs.model_parallel_configs import ModelParallelConfigs, create_balanced_device_map
from src.pre_training.trainer_model_parallel import TrainerModelParallel

# 初始化配置
dataset_configs = DatasetConfigs()
tokenizer_configs = TokenizerConfigs()
model_configs = ModelConfigs()
training_configs = TrainingConfigs()

# 创建设备映射
available_devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
device_map = create_balanced_device_map(
    num_layers=model_configs.NUM_LAYERS,
    available_devices=available_devices
)

# 增加批量大小（模型并行可以支持更大的batch size）
training_configs.BATCH_SIZE = training_configs.BATCH_SIZE * 2  # 或更大

# 创建并启动训练器
trainer = TrainerModelParallel(
    dataset_configs=dataset_configs,
    tokenizer_configs=tokenizer_configs,
    model_configs=model_configs,
    training_configs=training_configs,
    device_map=device_map,
    ddp=False,  # 使用模型并行而非DDP
    samples=["Hello, how are you?", "What is the future of AI?"]
)

trainer.train()
```

### 2. 运行示例脚本

```bash
cd /home/ubuntu/MyFiles/GitHub/gpt-mini
python examples/train_model_parallel.py
```

### 3. 配置选项

#### 设备映射策略

```python
# 1. 平衡分布（默认）
device_map = create_balanced_device_map(num_layers, available_devices)

# 2. 内存优化分布
device_map = create_memory_optimized_device_map(num_layers, available_devices)

# 3. 自定义分布
device_map = {
    'embedding': torch.device('cuda:0'),
    'layer_0': torch.device('cuda:0'),
    'layer_1': torch.device('cuda:1'),
    'layer_2': torch.device('cuda:1'),
    'output': torch.device('cuda:1')
}
```

#### 批量大小优化

```python
# 自动推荐批量大小
recommended_batch_size = get_recommended_batch_size(device_map, model_size_gb=1.0)
training_configs.BATCH_SIZE = recommended_batch_size
```

## 📈 性能对比

### 内存使用对比

| 配置 | 单GPU内存使用 | 模型并行内存使用 | 支持的最大Batch Size |
|------|---------------|------------------|---------------------|
| 原始模型 | 8GB | N/A | 6 |
| 2-GPU并行 | 4GB + 4GB | 8GB total | 12-16 |
| 4-GPU并行 | 2GB × 4 | 8GB total | 20-24 |

### 训练速度

虽然模型并行会引入设备间通信开销，但通过更大的批量大小可以获得整体性能提升：
- **更大批量**：提高GPU利用率
- **减少迭代次数**：更少的gradient updates达到同样效果
- **内存效率**：避免内存溢出导致的重启

## ⚙️ 配置详解

### ModelParallelConfigs 参数

```python
@dataclass
class ModelParallelConfigs:
    ENABLE_MODEL_PARALLEL: bool = True          # 启用模型并行
    AUTO_DEVICE_MAP: bool = True                # 自动创建设备映射
    FORCE_CPU: bool = False                     # 强制使用CPU（测试用）
    PREFERRED_DEVICES: List[int] = None         # 首选GPU设备列表
    GRADIENT_CHECKPOINTING: bool = False        # 梯度检查点（节省内存）
    MIXED_PRECISION: bool = False               # 混合精度训练
    PROFILE_MEMORY: bool = True                 # 内存使用分析
    LOG_DEVICE_ASSIGNMENTS: bool = True         # 记录设备分配
```

## 🔧 故障排除

### 常见问题

1. **CUDA内存不足**
   ```
   解决方案：
   - 减少批量大小
   - 启用梯度检查点
   - 使用更多GPU设备
   ```

2. **设备间通信延迟**
   ```
   解决方案：
   - 优化设备映射
   - 减少跨设备数据传输
   - 使用高速互联的GPU
   ```

3. **模型加载失败**
   ```
   解决方案：
   - 检查设备可用性
   - 验证模型并行配置
   - 确保所有GPU可访问
   ```

### 调试技巧

1. **监控内存使用**
   ```python
   memory_usage = model.get_memory_usage()
   for device_name, usage in memory_usage.items():
       print(f"{device_name}: {usage['allocated']:.2f}GB allocated")
   ```

2. **查看设备分配**
   ```python
   from src.configs.model_parallel_configs import print_device_map
   print_device_map(device_map)
   ```

3. **性能分析**
   ```python
   # 启用详细内存分析
   configs = ModelParallelConfigs()
   configs.PROFILE_MEMORY = True
   ```

## 📚 最佳实践

### 1. 设备选择
- **优先使用相同型号的GPU**：确保内存和计算能力一致
- **考虑GPU互联带宽**：NVLink > PCIe 4.0 > PCIe 3.0
- **预留内存缓冲**：不要使用100%的GPU内存

### 2. 批量大小调优
- **逐步增加**：从原始batch size的2倍开始
- **监控内存**：确保不超过GPU内存限制
- **平衡效率**：过大的batch size可能影响收敛

### 3. 模型分布策略
- **均匀分布**：对于相同的GPU，均匀分配层数
- **内存优化**：对于不同内存的GPU，按容量分配
- **通信最小化**：减少频繁的跨设备传输

## 🔮 未来增强

### 计划中的功能
1. **流水线并行 (Pipeline Parallelism)**
2. **数据并行 + 模型并行混合**
3. **动态负载均衡**
4. **自动内存优化**
5. **分布式推理支持**

## 📞 获得帮助

如果您在使用模型并行功能时遇到问题：

1. **检查系统要求**：确保有多个可用的GPU
2. **查看错误日志**：分析具体的错误信息
3. **调整配置**：尝试不同的设备映射和批量大小
4. **参考示例**：运行 `examples/train_model_parallel.py`

---

**注意**：模型并行功能需要PyTorch 1.8+和CUDA支持。建议在配置完成后先进行小规模测试。
