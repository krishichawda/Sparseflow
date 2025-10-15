"""
Sparse OPT Model Implementation
Custom implementation with dynamic neuron offloading
"""

from typing import Optional, Tuple, List
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.models.opt.modeling_opt import (
    OPTAttention, OPTDecoder, OPTModel, OPTForCausalLM
)
from transformers.generation import GenerationConfig

from ..core.config import SparseFlowConfig
from ..core.weight_manager import DiskWeightStore, AsyncWeightTransfer
from ..orchestration.neuron_predictor import PredictorFactory
from ..orchestration.transfer_coordinator import TransferCoordinator


class ExecutionContext:
    """Tracks execution state across forward passes"""
    
    def __init__(self):
        self.current_layer = 0
        self.is_prefill_phase = True
        self.forward_latencies: List[float] = []
        self.transfer_stats: dict = {}
    
    def reset_layer_counter(self):
        """Reset layer counter for new forward pass"""
        self.current_layer = 0
    
    def record_latency(self, latency: float):
        """Record forward pass latency"""
        self.forward_latencies.append(latency)
    
    def get_avg_decode_latency(self) -> float:
        """Get average decoding latency (excluding first token)"""
        if len(self.forward_latencies) <= 1:
            return 0.0
        return sum(self.forward_latencies[1:]) / len(self.forward_latencies[1:])


class SparseMLPLayer(nn.Module):
    """
    Sparse MLP layer with dynamic weight loading
    """
    
    def __init__(self, layer_idx: int, config: SparseFlowConfig,
                 coordinator: TransferCoordinator, predictor_factory: PredictorFactory,
                 weight_store: DiskWeightStore, context: ExecutionContext):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.coordinator = coordinator
        self.predictor_factory = predictor_factory
        self.weight_store = weight_store
        self.context = context
        
        # Weight tensors (dynamically loaded)
        self.fc1_weight_base: Optional[torch.Tensor] = None
        self.fc1_bias_base: Optional[torch.Tensor] = None
        self.fc2_weight_base: Optional[torch.Tensor] = None
        
        # Delta weights for newly loaded neurons
        self.fc1_weight_delta: Optional[torch.Tensor] = None
        self.fc1_bias_delta: Optional[torch.Tensor] = None
        self.fc2_weight_delta: Optional[torch.Tensor] = None
        
        # Activation function
        self.activation_fn = ACT2FN[self.config.profile.model_identifier.split('-')[0] 
                                    if '-' in self.config.profile.model_identifier else 'relu']
        
        # Ring buffer position for cache updates
        self.cache_write_position = 0
    
    def prepare_weights(self, predicted_neurons: torch.Tensor):
        """
        Prepare weights based on predicted active neurons
        
        Args:
            predicted_neurons: Indices of predicted active neurons
        """
        # Create transfer plan
        plan = self.coordinator.plan_transfer(self.layer_idx, predicted_neurons)
        
        if self.context.is_prefill_phase:
            # Load full weights during prefill
            self._load_full_weights()
            # Initialize cache with predicted neurons
            self.coordinator.get_layer_cache(self.layer_idx).bulk_insert(
                predicted_neurons.tolist()
            )
        else:
            # Load only missing neurons
            if not plan.is_empty():
                self._load_delta_weights(plan)
            else:
                # All neurons cached, no transfer needed
                self._clear_delta_weights()
    
    def _load_full_weights(self):
        """Load complete layer weights"""
        fc1_shape = (self.config.profile.ffn_dimension, self.config.profile.hidden_dimension)
        fc2_shape = (self.config.profile.ffn_dimension, self.config.profile.hidden_dimension)
        
        self.fc1_weight_base, self.fc1_bias_base, self.fc2_weight_base = \
            self.coordinator.warmup_layer(
                self.layer_idx,
                f'decoder.layers.{self.layer_idx}.fc1.weight',
                f'decoder.layers.{self.layer_idx}.fc1.bias',
                f'decoder.layers.{self.layer_idx}.fc2.weight',
                fc1_shape,
                fc2_shape
            )
        
        # Register shapes
        self.weight_store.register_weight_shape(
            f'decoder.layers.{self.layer_idx}.fc1.weight', fc1_shape
        )
        self.weight_store.register_weight_shape(
            f'decoder.layers.{self.layer_idx}.fc1.bias', (fc1_shape[0],)
        )
        self.weight_store.register_weight_shape(
            f'decoder.layers.{self.layer_idx}.fc2.weight', fc2_shape
        )
        
        self._clear_delta_weights()
    
    def _load_delta_weights(self, plan):
        """Load weights for newly required neurons"""
        fc1_shape = (self.config.profile.ffn_dimension, self.config.profile.hidden_dimension)
        fc2_shape = (self.config.profile.ffn_dimension, self.config.profile.hidden_dimension)
        
        self.fc1_weight_delta, self.fc1_bias_delta, self.fc2_weight_delta = \
            self.coordinator.execute_transfer(
                self.layer_idx,
                plan,
                f'decoder.layers.{self.layer_idx}.fc1.weight',
                f'decoder.layers.{self.layer_idx}.fc1.bias',
                f'decoder.layers.{self.layer_idx}.fc2.weight',
                fc1_shape,
                fc2_shape
            )
    
    def _clear_delta_weights(self):
        """Clear delta weight tensors"""
        empty_tensor = torch.empty(
            0, self.config.profile.hidden_dimension,
            dtype=self.config.compute_dtype,
            device=self.config.primary_device
        )
        self.fc1_weight_delta = empty_tensor
        self.fc2_weight_delta = empty_tensor
        self.fc1_bias_delta = torch.empty(
            0, dtype=self.config.compute_dtype,
            device=self.config.primary_device
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP layer
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            
        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        # Combine base and delta weights
        fc1_weight = torch.cat([self.fc1_weight_base, self.fc1_weight_delta], dim=0)
        fc1_bias = torch.cat([self.fc1_bias_base, self.fc1_bias_delta], dim=0)
        fc2_weight = torch.cat([self.fc2_weight_base, self.fc2_weight_delta], dim=0)
        
        # Standard MLP computation
        # Note: Weight ordering doesn't matter for final result
        x = F.linear(hidden_states, fc1_weight, fc1_bias)
        x = self.activation_fn(x)
        x = F.linear(x, fc2_weight.t())  # fc2_weight is pre-transposed
        
        return x
    
    def update_cache(self):
        """Update ring buffer cache with delta weights"""
        if self.config.strategy.aggressive_offload:
            # Clear everything
            self.fc1_weight_base = None
            self.fc1_bias_base = None
            self.fc2_weight_base = None
            return
        
        if self.context.is_prefill_phase:
            # After prefill, keep only cached subset
            cached_indices = self.coordinator.get_cached_neuron_indices(self.layer_idx)
            if cached_indices.numel() > 0:
                gpu_indices = cached_indices.to(self.config.primary_device)
                self.fc1_weight_base = self.fc1_weight_base[gpu_indices]
                self.fc1_bias_base = self.fc1_bias_base[gpu_indices]
                self.fc2_weight_base = self.fc2_weight_base[gpu_indices]
            return
        
        # Update ring buffer with new weights
        delta_size = self.fc1_weight_delta.size(0)
        if delta_size == 0:
            return
        
        cache_size = self.fc1_weight_base.size(0)
        next_position = (self.cache_write_position + delta_size) % cache_size
        
        if next_position > self.cache_write_position:
            # Contiguous update
            self.fc1_weight_base[self.cache_write_position:next_position] = self.fc1_weight_delta
            self.fc1_bias_base[self.cache_write_position:next_position] = self.fc1_bias_delta
            self.fc2_weight_base[self.cache_write_position:next_position] = self.fc2_weight_delta
        elif delta_size > 0:
            # Wrap around
            split = cache_size - self.cache_write_position
            self.fc1_weight_base[self.cache_write_position:] = self.fc1_weight_delta[:split]
            self.fc1_weight_base[:next_position] = self.fc1_weight_delta[split:]
            self.fc1_bias_base[self.cache_write_position:] = self.fc1_bias_delta[:split]
            self.fc1_bias_base[:next_position] = self.fc1_bias_delta[split:]
            self.fc2_weight_base[self.cache_write_position:] = self.fc2_weight_delta[:split]
            self.fc2_weight_base[:next_position] = self.fc2_weight_delta[split:]
        
        self.cache_write_position = next_position


class SparseAttentionLayer(nn.Module):
    """
    Attention layer that triggers MLP weight loading
    """
    
    def __init__(self, layer_idx: int, config: SparseFlowConfig,
                 predictor_factory: PredictorFactory, next_mlp_layer,
                 weight_store: DiskWeightStore, async_transfer: AsyncWeightTransfer):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.predictor_factory = predictor_factory
        self.next_mlp_layer = next_mlp_layer
        self.weight_store = weight_store
        self.async_transfer = async_transfer
        
        # Use standard OPT attention (weights loaded separately)
        self.attention = None
        self.layer_norm = None
    
    def load_attention_weights(self):
        """Load attention weights to GPU"""
        # Load full attention weights
        head_dim = self.config.get_head_dimension()
        hidden_dim = self.config.profile.hidden_dimension
        
        # Load concatenated attention weights
        qkv_out_shape = (self.config.profile.num_attention_heads, head_dim * 4, hidden_dim)
        qkv_out_weight = self.weight_store.load_weight(
            f'decoder.layers.{self.layer_idx}.self_attn.qkv_out_proj.weight',
            qkv_out_shape
        )
        
        # Transfer to GPU
        gpu_weights = self.async_transfer.schedule_transfer(qkv_out_weight)
        return gpu_weights
    
    def forward(self, hidden_states: torch.Tensor, **kwargs) -> Tuple:
        """Forward through attention with async MLP preparation"""
        # Predict next layer's MLP neurons based on current hidden states
        predictor = self.predictor_factory.get_predictor(self.layer_idx)
        k = self.config.get_gpu_neuron_count(self.layer_idx)
        predicted_neurons = predictor.select_top_k(
            hidden_states,
            k,
            self.config.strategy.activation_threshold
        )
        
        # Asynchronously prepare MLP weights while computing attention
        self.next_mlp_layer.prepare_weights(predicted_neurons)
        
        # Compute attention (using standard implementation)
        normed = self.layer_norm(hidden_states)
        attn_output = self.attention(normed, **kwargs)
        
        return attn_output


class SparseOPTDecoderLayer(nn.Module):
    """Complete decoder layer with sparse MLP"""
    
    def __init__(self, layer_idx: int, config: SparseFlowConfig,
                 coordinator: TransferCoordinator, predictor_factory: PredictorFactory,
                 weight_store: DiskWeightStore, async_transfer: AsyncWeightTransfer,
                 context: ExecutionContext, next_attention_layer=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.context = context
        
        # MLP layer
        self.mlp = SparseMLPLayer(
            layer_idx, config, coordinator, predictor_factory,
            weight_store, context
        )
        
        # Attention layer (loads next layer's MLP weights)
        self.attention_wrapper = SparseAttentionLayer(
            layer_idx, config, predictor_factory, self.mlp,
            weight_store, async_transfer
        )
        
        # Next layer attention (loaded during MLP computation)
        self.next_attention_layer = next_attention_layer
        
        # Layer norms and other components
        self.final_layer_norm = None
    
    def forward(self, hidden_states: torch.Tensor, **kwargs):
        """Forward pass through decoder layer"""
        # Attention forward (prepares MLP weights async)
        residual = hidden_states
        attn_output = self.attention_wrapper.forward(hidden_states, **kwargs)
        hidden_states = residual + attn_output[0]
        
        # Synchronize to ensure MLP weights are ready
        torch.cuda.synchronize()
        
        # Load next layer's attention weights while computing MLP
        if self.next_attention_layer is not None:
            # Async load next attention
            pass
        
        # MLP forward
        residual = hidden_states
        normed = self.final_layer_norm(hidden_states)
        mlp_output = self.mlp(normed)
        hidden_states = residual + mlp_output
        
        # Update MLP cache after computation
        self.mlp.update_cache()
        
        self.context.current_layer += 1
        
        return (hidden_states,) + attn_output[1:]


class SparseOPTForCausalLM(nn.Module):
    """
    Sparse OPT model with dynamic offloading
    """
    
    def __init__(self, config: SparseFlowConfig, weight_store: DiskWeightStore):
        super().__init__()
        self.config = config
        self.weight_store = weight_store
        
        # Execution context
        self.context = ExecutionContext()
        
        # Transfer infrastructure
        self.async_transfer = AsyncWeightTransfer()
        self.coordinator = TransferCoordinator(
            config, weight_store, self.async_transfer
        )
        
        # Predictor factory
        self.predictor_factory = PredictorFactory(
            config, config.predictor_weights_path
        )
        
        # Build layers
        self.layers: List[SparseOPTDecoderLayer] = []
        self._build_layers()
        
        # Embeddings
        self.embed_tokens_weight = None
        self.embed_positions_weight = None
        
        # Output head
        self.lm_head_weight = None
        self.final_norm_weight = None
        self.final_norm_bias = None
    
    def _build_layers(self):
        """Build all decoder layers"""
        for i in range(self.config.profile.num_layers):
            next_attn = self.layers[i - 1].attention_wrapper if i > 0 else None
            
            layer = SparseOPTDecoderLayer(
                i, self.config, self.coordinator, self.predictor_factory,
                self.weight_store, self.async_transfer, self.context,
                next_attention_layer=next_attn
            )
            self.layers.append(layer)
    
    def load_embeddings(self):
        """Load embedding weights"""
        if self.embed_tokens_weight is None:
            emb_shape = (self.config.profile.vocab_size, self.config.profile.hidden_dimension)
            pos_shape = (self.config.profile.max_sequence_length + 2, self.config.profile.hidden_dimension)
            
            emb_weight = self.weight_store.load_weight('decoder.embed_tokens.weight', emb_shape)
            pos_weight = self.weight_store.load_weight('decoder.embed_positions.weight', pos_shape)
            
            self.embed_tokens_weight, self.embed_positions_weight = \
                self.async_transfer.schedule_batch_transfer([emb_weight, pos_weight])
    
    def forward(self, input_ids: torch.Tensor, **kwargs):
        """Forward pass"""
        start_time = time.time()
        
        # Ensure embeddings loaded
        torch.cuda.synchronize()
        
        # Embedding lookup
        hidden_states = F.embedding(input_ids, self.embed_tokens_weight)
        
        # Position embeddings
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0) + 2
        position_embeds = F.embedding(positions, self.embed_positions_weight)
        hidden_states = hidden_states + position_embeds
        
        # Through layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, **kwargs)[0]
        
        # Final norm and LM head
        hidden_states = F.layer_norm(
            hidden_states,
            (self.config.profile.hidden_dimension,),
            self.final_norm_weight,
            self.final_norm_bias
        )
        logits = F.linear(hidden_states, self.lm_head_weight)
        
        # Record latency
        torch.cuda.synchronize()
        self.context.record_latency(time.time() - start_time)
        self.context.reset_layer_counter()
        
        # After first pass, switch to decode mode
        if self.context.is_prefill_phase:
            self.context.is_prefill_phase = False
        
        return type('Output', (), {'logits': logits})()
    
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100, **kwargs):
        """Simple greedy generation"""
        self.load_embeddings()
        torch.cuda.synchronize()
        
        generated = input_ids
        
        for _ in range(max_new_tokens):
            outputs = self.forward(generated)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            
            if next_token.item() == 2:  # EOS
                break
        
        return generated

