# SparseFlow: Adaptive Sparse Model Offloading

**Efficient large language model inference through intelligent neuron-level weight management**

SparseFlow enables running large language models on resource-constrained GPUs by exploiting the natural activation sparsity in neural networks. Through dynamic weight offloading and LRU caching, SparseFlow achieves significant speedups compared to naive offloading approaches while maintaining competitive accuracy.

## 🌟 Key Features

- **Adaptive Weight Management**: Dynamically loads only the neurons predicted to be active
- **LRU Caching**: Intelligent cache eviction keeps frequently used neurons on GPU
- **Async Transfer**: Overlaps weight loading with computation for minimal overhead
- **Memory Efficient**: Run models 2-3x larger than GPU memory capacity
- **Fast Inference**: ~15x faster than naive offloading, ~7x faster than dense partial offloading

## 📊 Performance Highlights

With SparseFlow on a single GPU:
- **Throughput**: 6-7 tokens/second for OPT-6.7B (13.4 GB model)
- **Memory Usage**: ~8.7 GB peak GPU memory (58% of model size)
- **Cache Hit Rate**: 90%+ for early layers, 60-65% for deep layers

### Basic Usage

```python
import torch
from transformers import AutoTokenizer
from sparseflow import SparseFlowConfig, OffloadStrategy, DiskWeightStore, SparseOPTForCausalLM

# Setup model
model_name = 'facebook/opt-1.3b'
hf_config = AutoConfig.from_pretrained(model_name)

# Configure offloading strategy
strategy = OffloadStrategy(
    gpu_cache_ratio=0.3,        # Keep 30% of neurons on GPU
    use_lru_eviction=True,      # Use LRU cache policy
    async_transfer=True,        # Enable async weight loading
)

config = SparseFlowConfig.from_transformers_config(hf_config, strategy=strategy)
weight_store = DiskWeightStore(model_name)
weight_store.initialize_from_pretrained(hf_config)

# Create model
model = SparseOPTForCausalLM(config, weight_store)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Generate text
prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors='pt')
output = model.generate(inputs.input_ids.to('cuda'), max_new_tokens=100)
print(tokenizer.decode(output[0]))
```

### Run Demo

```bash
python -m sparseflow.examples.demo_generation
```

## 🔧 How It Works

SparseFlow exploits **activation sparsity** in neural networks, particularly in MLP (feedforward) layers:

### 1. Natural Sparsity
- With ReLU activation, many neurons output zero for a given input
- Early layers: >99% of neurons inactive
- Deep layers: ~90% of neurons inactive
- Adjacent tokens share 60-90% of active neurons

### 2. Adaptive Loading Pipeline

```
┌─────────────┐
│  Input      │
│  Token      │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│   Predictor     │  ← Predicts active neurons
│   Network       │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  LRU Cache      │  ← Check what's already on GPU
│  Lookup         │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Async Weight   │  ← Load missing neurons from CPU
│  Transfer       │     (overlapped with attention computation)
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  MLP Forward    │  ← Compute with sparse weights
│  Pass           │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Cache Update   │  ← Update LRU cache
└─────────────────┘
```

### 3. Key Innovations

**Neuron-Level Granularity**: Unlike layer-level offloading, SparseFlow operates at neuron granularity, minimizing data transfer.

**Predictive Loading**: Uses lightweight predictor networks to forecast which neurons will activate, enabling proactive loading.

**Async Execution**: Weight transfers happen concurrently with attention computation, hiding transfer latency.

**LRU Caching**: Frequently accessed neurons stay on GPU, reducing redundant transfers across tokens.

## 📈 Architecture

```
sparseflow/
├── core/                      # Core functionality
│   ├── config.py             # Configuration management
│   └── weight_manager.py     # Weight storage and caching
├── orchestration/            # Coordination layer
│   ├── neuron_predictor.py  # Activation prediction
│   └── transfer_coordinator.py  # Transfer planning
├── models/                   # Model implementations
│   └── opt_sparse.py        # Sparse OPT model
├── utils/                    # Utilities
│   └── tensor_ops.py        # Tensor operations
└── examples/                 # Example scripts
    └── demo_generation.py   # Demo application
```

## ⚙️ Configuration Options

### OffloadStrategy Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gpu_cache_ratio` | float | 0.3 | Percentage of neurons to cache on GPU |
| `cpu_cache_ratio` | float | 1.0 | Percentage to keep in CPU memory |
| `aggressive_offload` | bool | False | Fully offload after each layer |
| `activation_threshold` | float | 1.0 | Minimum activation score threshold |
| `use_lru_eviction` | bool | True | Use LRU vs FIFO eviction |
| `async_transfer` | bool | True | Enable asynchronous transfers |

### Example Configurations

**Maximum Speed** (requires more GPU memory):
```python
strategy = OffloadStrategy(gpu_cache_ratio=0.5, use_lru_eviction=True)
```

**Minimum Memory** (slower):
```python
strategy = OffloadStrategy(gpu_cache_ratio=0.2, aggressive_offload=True)
```

**Balanced** (recommended):
```python
strategy = OffloadStrategy(gpu_cache_ratio=0.3, use_lru_eviction=True)
```

## 🎯 Use Cases

- **Research**: Experiment with large models on consumer GPUs
- **Prototyping**: Rapid iteration without expensive hardware
- **Edge Deployment**: Run large models on resource-constrained devices
- **Cost Optimization**: Reduce cloud GPU costs for inference

## ⚠️ Limitations

1. **Approximate Inference**: Predictors aren't perfect; slight accuracy degradation possible
2. **ReLU Dependency**: Optimized for ReLU-based architectures (OPT, some BERT variants)
3. **Batch Size 1**: Current implementation optimized for single-sample inference
4. **Predictor Training**: Pre-trained predictors required for best performance

## 🔬 Technical Details

### Memory Layout

Weights are stored in optimized formats for fast indexing:
- **FC1 weights**: `[ffn_dim, hidden_dim]` - row-major for neuron indexing
- **FC2 weights**: `[ffn_dim, hidden_dim]` - pre-transposed for efficient indexing
- **Attention weights**: Concatenated by head for potential head-level pruning

### Cache Management

Uses a circular ring buffer with LRU tracking:
1. New neurons overwrite LRU entries
2. Cache hit: Move entry to MRU position
3. Cache miss: Evict LRU, insert new neuron

### Predictor Networks

Lightweight 2-layer MLPs predict neuron activations:
- Input: Last token hidden state `[hidden_dim]`
- Hidden: Compressed representation
- Output: Activation scores `[ffn_dim]`

## 🤝 Contributing

Contributions welcome! Areas for improvement:

1. **More Models**: Extend beyond OPT (BERT, GPT-2, LLaMA)
2. **Attention Sparsity**: Add head-level pruning for attention layers
3. **Custom Kernels**: Optimize non-contiguous indexing with CUDA
4. **Adaptive Ratios**: Per-layer cache size based on sparsity profiles
5. **Evaluation Suite**: Comprehensive accuracy benchmarks

## 📚 References

This work builds on research in sparse neural networks and efficient inference:

- [Deja Vu: Contextual Sparsity for Efficient LLMs](https://proceedings.mlr.press/v202/liu23am/liu23am.pdf)
- [LLM in a Flash: Efficient Large Language Model Inference](https://arxiv.org/abs/2312.11514)
- Activation sparsity in ReLU networks

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

Built with PyTorch and HuggingFace Transformers.

---

**Note**: SparseFlow is a research project focused on exploring efficient inference techniques. For production use, thorough testing and validation is recommended.

