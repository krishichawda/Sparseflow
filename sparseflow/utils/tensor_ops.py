"""
Tensor Operations Utilities
Helper functions for tensor manipulations and transfers
"""

from typing import List, Set, Tuple
import torch
import numpy as np


def compute_set_difference(requested: torch.Tensor, cached: torch.Tensor) -> torch.Tensor:
    """
    Compute elements in requested that are not in cached
    
    Args:
        requested: Tensor of requested indices
        cached: Tensor of cached indices
        
    Returns:
        Tensor of indices in requested but not in cached
    """
    requested_set = set(requested.tolist())
    cached_set = set(cached.tolist())
    diff = requested_set - cached_set
    
    return torch.tensor(list(diff), dtype=requested.dtype, device=requested.device)


def select_top_k_with_threshold(scores: torch.Tensor, k: int, 
                                threshold: float = 0.0) -> torch.Tensor:
    """
    Select top-k elements that exceed a threshold
    
    Args:
        scores: Tensor of scores
        k: Number of elements to select
        threshold: Minimum threshold value
        
    Returns:
        Indices of selected elements
    """
    values, indices = torch.topk(scores, min(k, scores.numel()), sorted=False)
    
    # Filter by threshold
    mask = values > threshold
    filtered = indices[mask]
    
    return filtered.int()


def pin_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Pin tensor to CPU memory for faster GPU transfer
    
    Args:
        tensor: Input tensor
        
    Returns:
        Pinned tensor
    """
    if tensor.is_pinned():
        return tensor
    
    pinned = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
    pinned.copy_(tensor)
    return pinned


def batch_index_select(source: torch.Tensor, indices: torch.Tensor, 
                       dim: int = 0) -> torch.Tensor:
    """
    Select indices from source tensor along specified dimension
    
    Args:
        source: Source tensor
        indices: Indices to select
        dim: Dimension along which to index
        
    Returns:
        Indexed tensor
    """
    if dim == 0:
        return source[indices]
    elif dim == 1:
        return source[:, indices]
    else:
        raise ValueError(f"Unsupported dimension: {dim}")


def convert_memmap_to_tensor(memmap: np.memmap, pin_memory: bool = False) -> torch.Tensor:
    """
    Convert numpy memmap to torch tensor
    
    Args:
        memmap: Numpy memory map
        pin_memory: Whether to pin memory
        
    Returns:
        Torch tensor
    """
    # Create tensor from memmap
    tensor = torch.from_numpy(np.array(memmap))
    
    if pin_memory:
        tensor = pin_tensor(tensor)
    
    return tensor


def circular_buffer_update(buffer: torch.Tensor, new_data: torch.Tensor,
                           write_position: int) -> int:
    """
    Update circular buffer with new data
    
    Args:
        buffer: Buffer tensor to update
        new_data: New data to write
        write_position: Current write position
        
    Returns:
        New write position after update
    """
    buffer_size = buffer.size(0)
    data_size = new_data.size(0)
    
    if data_size == 0:
        return write_position
    
    next_position = (write_position + data_size) % buffer_size
    
    if next_position > write_position:
        # Contiguous write
        buffer[write_position:next_position] = new_data
    else:
        # Wrap around
        split = buffer_size - write_position
        buffer[write_position:] = new_data[:split]
        buffer[:next_position] = new_data[split:]
    
    return next_position


class StreamManager:
    """Manages CUDA streams for async operations"""
    
    def __init__(self, num_streams: int = 2):
        self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
        self.current_stream_idx = 0
    
    def get_next_stream(self) -> torch.cuda.Stream:
        """Get next stream in round-robin fashion"""
        stream = self.streams[self.current_stream_idx]
        self.current_stream_idx = (self.current_stream_idx + 1) % len(self.streams)
        return stream
    
    def synchronize_all(self):
        """Synchronize all streams"""
        for stream in self.streams:
            stream.synchronize()


def measure_memory_usage() -> Tuple[float, float]:
    """
    Get current and peak GPU memory usage in GB
    
    Returns:
        Tuple of (current_gb, peak_gb)
    """
    current = torch.cuda.memory_allocated() / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return current, peak


def reset_memory_stats():
    """Reset GPU memory statistics"""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

