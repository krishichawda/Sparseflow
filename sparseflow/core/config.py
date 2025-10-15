"""
SparseFlow Configuration Module
Manages configuration for adaptive sparse model offloading
"""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class OffloadStrategy:
    """Configuration for offloading strategy"""
    
    # GPU cache size as percentage of full layer (e.g., 0.3 = 30%)
    gpu_cache_ratio: float = 0.3
    
    # CPU cache size as percentage (1.0 = full layer)
    cpu_cache_ratio: float = 1.0
    
    # Whether to completely offload layers after forward pass
    aggressive_offload: bool = False
    
    # Prediction threshold for neuron activation
    activation_threshold: float = 1.0
    
    # LRU cache eviction policy vs FIFO
    use_lru_eviction: bool = True
    
    # Enable asynchronous weight transfer
    async_transfer: bool = True


@dataclass
class ModelProfile:
    """Model-specific configuration profile"""
    
    model_identifier: str
    hidden_dimension: int
    ffn_dimension: int
    num_layers: int
    num_attention_heads: int
    max_sequence_length: int
    vocab_size: int
    
    # Optional: Layer-specific sparsity levels
    layer_sparsity_map: Optional[dict] = None


@dataclass
class SparseFlowConfig:
    """Main configuration for SparseFlow system"""
    
    # Model profile
    profile: ModelProfile
    
    # Offloading strategy
    strategy: OffloadStrategy = field(default_factory=OffloadStrategy)
    
    # Computation precision
    compute_dtype: torch.dtype = torch.float16
    
    # Device specifications
    primary_device: str = 'cuda'
    secondary_device: str = 'cpu'
    
    # Performance tracking
    enable_profiling: bool = True
    
    # Predictor model path (optional)
    predictor_weights_path: Optional[str] = None
    
    def get_gpu_neuron_count(self, layer_idx: Optional[int] = None) -> int:
        """Calculate number of neurons to cache on GPU"""
        if layer_idx is not None and self.strategy.use_lru_eviction:
            # Can vary per layer if sparsity map provided
            if self.profile.layer_sparsity_map:
                ratio = self.profile.layer_sparsity_map.get(layer_idx, self.strategy.gpu_cache_ratio)
            else:
                ratio = self.strategy.gpu_cache_ratio
        else:
            ratio = self.strategy.gpu_cache_ratio
        
        return int(self.profile.ffn_dimension * ratio)
    
    def get_head_dimension(self) -> int:
        """Calculate attention head dimension"""
        return self.profile.hidden_dimension // self.profile.num_attention_heads
    
    @classmethod
    def from_transformers_config(cls, hf_config, strategy: Optional[OffloadStrategy] = None):
        """Create SparseFlowConfig from HuggingFace config"""
        profile = ModelProfile(
            model_identifier=hf_config.name_or_path if hasattr(hf_config, 'name_or_path') else 'unknown',
            hidden_dimension=hf_config.hidden_size,
            ffn_dimension=hf_config.ffn_dim if hasattr(hf_config, 'ffn_dim') else hf_config.intermediate_size,
            num_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            max_sequence_length=hf_config.max_position_embeddings,
            vocab_size=hf_config.vocab_size,
        )
        
        return cls(
            profile=profile,
            strategy=strategy or OffloadStrategy(),
        )

