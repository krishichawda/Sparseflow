"""
Transfer Coordinator
Orchestrates the transfer of neuron weights between CPU and GPU
"""

from typing import List, Tuple, Set
import torch
from ..core.weight_manager import WeightCache, AsyncWeightTransfer


class NeuronTransferPlan:
    """Plan for transferring neurons between devices"""
    
    def __init__(self):
        self.neurons_to_load: List[int] = []
        self.neurons_to_evict: List[int] = []
        self.neurons_already_cached: List[int] = []
        self.transfer_positions: dict[int, int] = {}
    
    def get_total_transfer_count(self) -> int:
        """Get number of neurons that need to be transferred"""
        return len(self.neurons_to_load)
    
    def is_empty(self) -> bool:
        """Check if there are no transfers needed"""
        return len(self.neurons_to_load) == 0


class TransferCoordinator:
    """Coordinates neuron weight transfers between CPU and GPU"""
    
    def __init__(self, config, weight_store, async_transfer: AsyncWeightTransfer):
        self.config = config
        self.weight_store = weight_store
        self.async_transfer = async_transfer
        
        # Per-layer caches
        self.layer_caches: dict[int, WeightCache] = {}
    
    def initialize_layer_cache(self, layer_idx: int):
        """Initialize weight cache for a specific layer"""
        cache_size = self.config.get_gpu_neuron_count(layer_idx)
        self.layer_caches[layer_idx] = WeightCache(
            capacity=cache_size,
            device=self.config.primary_device
        )
    
    def get_layer_cache(self, layer_idx: int) -> WeightCache:
        """Get cache for a layer, creating if necessary"""
        if layer_idx not in self.layer_caches:
            self.initialize_layer_cache(layer_idx)
        return self.layer_caches[layer_idx]
    
    def plan_transfer(self, layer_idx: int, 
                     predicted_neurons: torch.Tensor) -> NeuronTransferPlan:
        """
        Create a transfer plan for predicted active neurons
        
        Args:
            layer_idx: Layer index
            predicted_neurons: Tensor of predicted active neuron indices
            
        Returns:
            NeuronTransferPlan with details of what to transfer
        """
        plan = NeuronTransferPlan()
        cache = self.get_layer_cache(layer_idx)
        
        # Find which neurons are missing from cache
        predicted_set = set(predicted_neurons.tolist())
        cached_set = set(cache.get_cached_indices().tolist())
        
        plan.neurons_already_cached = list(predicted_set & cached_set)
        plan.neurons_to_load = list(predicted_set - cached_set)
        
        # Determine positions for new neurons
        for neuron_idx in plan.neurons_to_load:
            position, evicted = cache.insert(neuron_idx)
            plan.transfer_positions[neuron_idx] = position
            if evicted is not None:
                plan.neurons_to_evict.append(evicted)
        
        return plan
    
    def execute_transfer(self, layer_idx: int, plan: NeuronTransferPlan,
                        fc1_weight_key: str, fc1_bias_key: str, 
                        fc2_weight_key: str,
                        fc1_shape: Tuple, fc2_shape: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Execute the transfer plan and load weights to GPU
        
        Args:
            layer_idx: Layer index
            plan: Transfer plan
            fc1_weight_key: Key for first FC layer weights
            fc1_bias_key: Key for FC1 bias
            fc2_weight_key: Key for second FC layer weights (transposed)
            fc1_shape: Shape of FC1 weights
            fc2_shape: Shape of FC2 weights (transposed)
            
        Returns:
            Tuple of (fc1_weights, fc1_bias, fc2_weights) on GPU
        """
        if plan.is_empty():
            # No transfers needed, return empty tensors
            return (
                torch.empty(0, fc1_shape[1], dtype=self.config.compute_dtype, device=self.config.primary_device),
                torch.empty(0, dtype=self.config.compute_dtype, device=self.config.primary_device),
                torch.empty(0, fc2_shape[1], dtype=self.config.compute_dtype, device=self.config.primary_device)
            )
        
        # Load indexed weights from disk
        neuron_indices = torch.tensor(plan.neurons_to_load, dtype=torch.int32)
        
        fc1_w = self.weight_store.load_indexed_weight(
            fc1_weight_key, neuron_indices, fc1_shape, dim=0
        )
        fc1_b = self.weight_store.load_indexed_weight(
            fc1_bias_key, neuron_indices, (fc1_shape[0],), dim=0
        )
        fc2_w = self.weight_store.load_indexed_weight(
            fc2_weight_key, neuron_indices, fc2_shape, dim=0
        )
        
        # Async transfer to GPU
        gpu_fc1_w, gpu_fc1_b, gpu_fc2_w = self.async_transfer.schedule_batch_transfer(
            [fc1_w, fc1_b, fc2_w],
            device=self.config.primary_device
        )
        
        return gpu_fc1_w, gpu_fc1_b, gpu_fc2_w
    
    def warmup_layer(self, layer_idx: int, 
                     fc1_weight_key: str, fc1_bias_key: str, fc2_weight_key: str,
                     fc1_shape: Tuple, fc2_shape: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Warmup layer by loading initial random neurons
        
        Returns full weights for initial prompt processing
        """
        # For first pass, load complete weights
        fc1_w = self.weight_store.load_weight(fc1_weight_key, fc1_shape)
        fc1_b = self.weight_store.load_weight(fc1_bias_key, (fc1_shape[0],))
        fc2_w = self.weight_store.load_weight(fc2_weight_key, fc2_shape)
        
        # Transfer to GPU
        gpu_fc1_w, gpu_fc1_b, gpu_fc2_w = self.async_transfer.schedule_batch_transfer(
            [fc1_w, fc1_b, fc2_w],
            device=self.config.primary_device
        )
        
        return gpu_fc1_w, gpu_fc1_b, gpu_fc2_w
    
    def get_cached_neuron_indices(self, layer_idx: int) -> torch.Tensor:
        """Get currently cached neuron indices for a layer"""
        cache = self.get_layer_cache(layer_idx)
        return cache.get_cached_indices()

