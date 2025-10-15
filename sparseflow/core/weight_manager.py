"""
Weight Management System
Handles loading, caching, and transfer of model weights with LRU eviction
"""

import os
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import numpy as np
import torch
from huggingface_hub import snapshot_download


class WeightCache:
    """LRU cache for neuron weights on GPU"""
    
    def __init__(self, capacity: int, device: str = 'cuda'):
        self.capacity = capacity
        self.device = device
        self.cache: OrderedDict[int, int] = OrderedDict()  # neuron_idx -> position in buffer
        self.available_positions: List[int] = list(range(capacity))
        
    def access(self, neuron_idx: int) -> Optional[int]:
        """Mark neuron as recently used and return its buffer position"""
        if neuron_idx in self.cache:
            # Move to end (most recent)
            self.cache.move_to_end(neuron_idx)
            return self.cache[neuron_idx]
        return None
    
    def insert(self, neuron_idx: int) -> Tuple[int, Optional[int]]:
        """
        Insert neuron into cache, returns (position, evicted_neuron_idx)
        """
        if neuron_idx in self.cache:
            self.cache.move_to_end(neuron_idx)
            return self.cache[neuron_idx], None
        
        # Get position for new neuron
        if self.available_positions:
            position = self.available_positions.pop(0)
            evicted = None
        else:
            # Evict LRU
            evicted_idx, position = self.cache.popitem(last=False)
            evicted = evicted_idx
        
        self.cache[neuron_idx] = position
        return position, evicted
    
    def bulk_insert(self, neuron_indices: List[int]) -> Dict[int, int]:
        """Insert multiple neurons, returns mapping of neuron_idx -> position"""
        mapping = {}
        for idx in neuron_indices:
            pos, _ = self.insert(idx)
            mapping[idx] = pos
        return mapping
    
    def get_cached_indices(self) -> torch.Tensor:
        """Return tensor of currently cached neuron indices"""
        return torch.tensor(list(self.cache.keys()), dtype=torch.int32)
    
    def get_missing_indices(self, requested: torch.Tensor) -> torch.Tensor:
        """Find which requested indices are not in cache"""
        cached = set(self.cache.keys())
        requested_set = set(requested.tolist())
        missing = requested_set - cached
        return torch.tensor(list(missing), dtype=torch.int32)


class DiskWeightStore:
    """Manages memory-mapped weights stored on disk"""
    
    def __init__(self, model_name: str, cache_dir: str = './sparseflow_cache'):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.weight_maps: Dict[str, np.memmap] = {}
        self.weight_shapes: Dict[str, Tuple] = {}
        
    def initialize_from_pretrained(self, hf_config):
        """Download and setup memory-mapped weight files"""
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Try to download pre-processed weights
        local_cache = os.path.join(self.cache_dir, self.model_name.replace('/', '_'))
        
        if not os.path.exists(local_cache):
            print(f"Downloading model weights for {self.model_name}")
            self._download_and_process_weights(hf_config, local_cache)
        
        self._load_memmaps(local_cache)
    
    def _download_and_process_weights(self, hf_config, output_dir):
        """Download original weights and create optimized memory maps"""
        os.makedirs(output_dir, exist_ok=True)
        
        weights_location = snapshot_download(
            repo_id=self.model_name,
            ignore_patterns=['flax*', 'tf*', '*.md', '*.txt']
        )
        
        # Load and process weight files
        import glob
        weight_files = glob.glob(os.path.join(weights_location, "pytorch_model*.bin"))
        
        for weight_file in weight_files:
            print(f"Processing {os.path.basename(weight_file)}")
            state_dict = torch.load(weight_file, map_location='cpu')
            
            for key, tensor in state_dict.items():
                # Remove 'model.' prefix if present
                clean_key = key.replace('model.', '')
                
                # Special handling for MLP weights - transpose fc2 for efficient indexing
                if 'fc2.weight' in clean_key or 'mlp.down_proj.weight' in clean_key:
                    tensor = tensor.t().contiguous()
                
                # Save as memory map
                output_path = os.path.join(output_dir, clean_key.replace('/', '_'))
                memmap = np.memmap(
                    output_path,
                    dtype='float16',
                    mode='w+',
                    shape=tuple(tensor.shape)
                )
                memmap[:] = tensor.cpu().numpy()
                memmap.flush()
    
    def _load_memmaps(self, cache_dir):
        """Load all memory-mapped weight files"""
        for filename in os.listdir(cache_dir):
            filepath = os.path.join(cache_dir, filename)
            if os.path.isfile(filepath):
                # Infer shape from file size
                file_size = os.path.getsize(filepath)
                # This is simplified - in practice, store metadata
                self.weight_maps[filename] = None  # Lazy load
    
    def load_weight(self, key: str, shape: Optional[Tuple] = None) -> torch.Tensor:
        """Load weight from disk to pinned CPU memory"""
        filename = key.replace('/', '_')
        filepath = os.path.join(
            self.cache_dir,
            self.model_name.replace('/', '_'),
            filename
        )
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Weight file not found: {filepath}")
        
        if shape is None:
            # Try to infer or use cached shape
            if key in self.weight_shapes:
                shape = self.weight_shapes[key]
            else:
                raise ValueError(f"Shape required for {key}")
        
        # Load as memory map
        memmap = np.memmap(filepath, dtype='float16', mode='r', shape=shape)
        
        # Convert to pinned tensor
        tensor = torch.from_numpy(np.array(memmap))
        pinned = torch.empty(tensor.shape, dtype=torch.float16, pin_memory=True)
        pinned.copy_(tensor)
        
        return pinned
    
    def load_indexed_weight(self, key: str, indices: torch.Tensor, 
                           shape: Tuple, dim: int = 0) -> torch.Tensor:
        """Load only specific indices from a weight tensor"""
        filename = key.replace('/', '_')
        filepath = os.path.join(
            self.cache_dir,
            self.model_name.replace('/', '_'),
            filename
        )
        
        memmap = np.memmap(filepath, dtype='float16', mode='r', shape=shape)
        
        # Index along specified dimension
        if dim == 0:
            indexed_data = memmap[indices.cpu().numpy()]
        else:
            indexed_data = memmap[:, indices.cpu().numpy()]
        
        # Convert to pinned tensor
        tensor = torch.from_numpy(np.array(indexed_data))
        pinned = torch.empty(tensor.shape, dtype=torch.float16, pin_memory=True)
        pinned.copy_(tensor)
        
        return pinned
    
    def register_weight_shape(self, key: str, shape: Tuple):
        """Register the shape of a weight tensor"""
        self.weight_shapes[key] = shape


class AsyncWeightTransfer:
    """Manages asynchronous weight transfers between devices"""
    
    def __init__(self):
        self.transfer_stream = torch.cuda.Stream()
        self.pending_transfers: List[torch.Tensor] = []
    
    def schedule_transfer(self, source: torch.Tensor, 
                         device: str = 'cuda') -> torch.Tensor:
        """Schedule async transfer of tensor to device"""
        with torch.cuda.stream(self.transfer_stream):
            destination = torch.empty(
                source.shape,
                dtype=source.dtype,
                device=device
            )
            destination.copy_(source, non_blocking=True)
            self.pending_transfers.append(destination)
            return destination
    
    def schedule_batch_transfer(self, sources: List[torch.Tensor],
                               device: str = 'cuda') -> List[torch.Tensor]:
        """Schedule multiple transfers in batch"""
        destinations = []
        with torch.cuda.stream(self.transfer_stream):
            for source in sources:
                dest = torch.empty(
                    source.shape,
                    dtype=source.dtype,
                    device=device
                )
                dest.copy_(source, non_blocking=True)
                destinations.append(dest)
                self.pending_transfers.append(dest)
        return destinations
    
    def synchronize(self):
        """Wait for all pending transfers to complete"""
        self.transfer_stream.synchronize()
        self.pending_transfers.clear()
    
    def get_stream(self):
        """Get the CUDA stream used for transfers"""
        return self.transfer_stream

