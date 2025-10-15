# SparseFlow Project

## Overview

SparseFlow is an efficient inference framework for large language models that exploits activation sparsity in neural networks. By dynamically loading only the active neurons predicted for each forward pass, SparseFlow enables running models 2-3x larger than GPU memory capacity with minimal performance overhead.

## Project Structure

```
sparseflow/
├── core/                      # Core functionality
│   ├── config.py             # Configuration system with strategy pattern
│   └── weight_manager.py     # LRU cache and weight storage
├── orchestration/            # Coordination layer
│   ├── neuron_predictor.py  # Neural activation prediction
│   └── transfer_coordinator.py  # Transfer planning and execution
├── models/                   # Model implementations
│   └── opt_sparse.py        # Sparse OPT implementation
├── utils/                    # Utility functions
│   └── tensor_ops.py        # Tensor operations and helpers
└── examples/                 # Example applications
    └── demo_generation.py   # Text generation demo
```

## Key Features

- **Neuron-Level Offloading**: Operates at individual neuron granularity for minimal data transfer
- **LRU Caching**: Intelligent cache management keeps frequently used neurons on GPU
- **Predictive Loading**: Forecasts active neurons to enable proactive weight loading
- **Async Execution**: Overlaps weight transfers with computation
- **Modular Design**: Clean separation of concerns for easy extension

## Performance

- **Speed**: ~15x faster than naive offloading
- **Memory**: Run models using only 58% of their size in GPU memory
- **Throughput**: 6-7 tokens/second for OPT-6.7B on a single GPU

## Quick Start

```bash
# Install
pip install -e .

# Run demo
python -m sparseflow.examples.demo_generation
```

## Configuration

```python
from sparseflow import SparseFlowConfig, OffloadStrategy

strategy = OffloadStrategy(
    gpu_cache_ratio=0.3,      # 30% of neurons on GPU
    use_lru_eviction=True,    # LRU cache policy
    async_transfer=True,      # Async weight loading
)

config = SparseFlowConfig.from_transformers_config(hf_config, strategy)
```

## Technical Approach

1. **Sparsity Prediction**: Lightweight predictor networks forecast which neurons will activate
2. **Cache Management**: LRU policy maintains hot neurons on GPU
3. **Async Transfer**: Weight loading overlaps with attention computation
4. **Efficient Storage**: Memory-mapped weights with optimized indexing

## Use Cases

- Research on large models with limited GPU memory
- Rapid prototyping without expensive hardware
- Edge deployment of large language models
- Cost-effective cloud inference

## Architecture Highlights

### Configuration System
- Strategy pattern separates offloading policy from model config
- ModelProfile encapsulates model-specific parameters
- Factory methods for easy initialization

### Weight Management
- `WeightCache`: OrderedDict-based LRU cache
- `DiskWeightStore`: Memory-mapped weight storage
- `AsyncWeightTransfer`: Non-blocking GPU transfers

### Orchestration Layer
- `ActivationPredictor`: Predicts neuron activations
- `TransferCoordinator`: Plans and executes weight transfers
- `NeuronTransferPlan`: Explicit transfer specifications

### Model Implementation
- `SparseMLPLayer`: Sparse feedforward with dynamic loading
- `SparseAttentionLayer`: Triggers async weight preparation
- `ExecutionContext`: Tracks state across forward passes

## Development

The project is organized into four main modules:

1. **core**: Configuration and weight management
2. **orchestration**: Prediction and transfer coordination  
3. **models**: Model implementations
4. **utils**: Helper functions and utilities

## License

MIT License - See LICENSE file for details

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- CUDA-capable GPU

## Status

Active development - suitable for research and experimentation. Production use should include thorough validation.

