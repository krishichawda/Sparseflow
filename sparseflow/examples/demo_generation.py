"""
SparseFlow Demo: Text Generation with Adaptive Offloading

This demonstrates using SparseFlow for efficient large model inference
with automatic sparse weight management.
"""

import torch
from transformers import AutoTokenizer, AutoConfig
from sparseflow import (
    SparseFlowConfig,
    OffloadStrategy,
    DiskWeightStore,
    SparseOPTForCausalLM,
    measure_memory_usage,
    reset_memory_stats,
)


def setup_model(model_name: str, gpu_cache_ratio: float = 0.3):
    """
    Initialize SparseFlow model with adaptive offloading
    
    Args:
        model_name: HuggingFace model identifier
        gpu_cache_ratio: Percentage of neurons to cache on GPU (0.0-1.0)
    
    Returns:
        Tuple of (model, tokenizer, config)
    """
    print(f"\n{'='*60}")
    print(f"Setting up SparseFlow for {model_name}")
    print(f"{'='*60}\n")
    
    # Load model configuration
    hf_config = AutoConfig.from_pretrained(model_name)
    
    # Create offload strategy
    strategy = OffloadStrategy(
        gpu_cache_ratio=gpu_cache_ratio,
        cpu_cache_ratio=1.0,
        aggressive_offload=False,
        activation_threshold=1.0,
        use_lru_eviction=True,
        async_transfer=True,
    )
    
    # Create SparseFlow configuration
    sf_config = SparseFlowConfig.from_transformers_config(
        hf_config,
        strategy=strategy
    )
    
    print(f"Model Configuration:")
    print(f"  - Hidden dim: {sf_config.profile.hidden_dimension}")
    print(f"  - FFN dim: {sf_config.profile.ffn_dimension}")
    print(f"  - Layers: {sf_config.profile.num_layers}")
    print(f"  - GPU cache: {gpu_cache_ratio*100:.0f}% per layer")
    print()
    
    # Initialize weight store
    weight_store = DiskWeightStore(model_name)
    weight_store.initialize_from_pretrained(hf_config)
    
    # Create sparse model
    model = SparseOPTForCausalLM(sf_config, weight_store)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer, sf_config


def generate_text(model, tokenizer, prompt: str, max_tokens: int = 100,
                  temperature: float = 0.9):
    """
    Generate text using SparseFlow model
    
    Args:
        model: SparseFlow model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Generated text
    """
    print(f"Prompt: {prompt}")
    print(f"\nGenerating (max {max_tokens} tokens)...")
    print("-" * 60)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs.input_ids.to('cuda')
    
    # Reset memory tracking
    reset_memory_stats()
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens
        )
    
    # Decode output
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return generated_text


def print_performance_stats(model, sf_config):
    """Print performance and memory statistics"""
    context = model.context
    
    print("\n" + "="*60)
    print("Performance Statistics")
    print("="*60)
    
    # Latency stats
    avg_latency = context.get_avg_decode_latency()
    if avg_latency > 0:
        tokens_per_sec = 1.0 / avg_latency
        print(f"Average decode latency: {avg_latency*1000:.2f} ms/token")
        print(f"Throughput: {tokens_per_sec:.2f} tokens/sec")
    
    # Memory stats
    current_mem, peak_mem = measure_memory_usage()
    print(f"\nMemory Usage:")
    print(f"  - Current GPU: {current_mem:.2f} GB")
    print(f"  - Peak GPU: {peak_mem:.2f} GB")
    
    # Cache efficiency
    print(f"\nCache Configuration:")
    print(f"  - GPU cache ratio: {sf_config.strategy.gpu_cache_ratio*100:.0f}%")
    print(f"  - Eviction policy: {'LRU' if sf_config.strategy.use_lru_eviction else 'FIFO'}")
    
    print("="*60 + "\n")


def main():
    """Main demo function"""
    # Configuration
    MODEL_NAME = 'facebook/opt-1.3b'  # Start with smaller model for demo
    GPU_CACHE_RATIO = 0.3  # Keep 30% of neurons on GPU
    
    # Setup
    model, tokenizer, config = setup_model(MODEL_NAME, GPU_CACHE_RATIO)
    
    # Example prompts
    prompts = [
        "The key to making great coffee is",
        "In the field of artificial intelligence, the most important breakthrough was",
        "To build a neural network from scratch, you need to",
    ]
    
    # Generate for each prompt
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'#'*60}")
        print(f"Example {i}/{len(prompts)}")
        print(f"{'#'*60}\n")
        
        generated = generate_text(
            model,
            tokenizer,
            prompt,
            max_tokens=80,
        )
        
        print("\nGenerated Text:")
        print("-" * 60)
        print(generated)
        print("-" * 60)
    
    # Print statistics
    print_performance_stats(model, config)


if __name__ == '__main__':
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This demo requires a GPU.")
        exit(1)
    
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    
    # Run demo
    main()

