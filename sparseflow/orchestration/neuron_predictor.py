"""
Neuron Activation Predictor
Predicts which neurons will be active in MLP layers
"""

import os
from typing import Optional, Callable
import torch
import torch.nn.functional as F


class ActivationPredictor:
    """Predicts active neurons based on layer input"""
    
    def __init__(self, input_dim: int, output_dim: int, 
                 device: str = 'cuda', dtype: torch.dtype = torch.float16):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.dtype = dtype
        
        # Predictor weights (2-layer MLP)
        self.fc1_weight: Optional[torch.Tensor] = None
        self.fc2_weight: Optional[torch.Tensor] = None
        self.is_loaded = False
    
    def load_weights(self, weight_dir: str, layer_idx: int):
        """Load pre-trained predictor weights"""
        fc1_path = os.path.join(weight_dir, f'layer_{layer_idx}_predictor_fc1.pt')
        fc2_path = os.path.join(weight_dir, f'layer_{layer_idx}_predictor_fc2.pt')
        
        if os.path.exists(fc1_path) and os.path.exists(fc2_path):
            self.fc1_weight = torch.load(fc1_path).to(self.device).to(self.dtype)
            self.fc2_weight = torch.load(fc2_path).to(self.device).to(self.dtype)
            self.is_loaded = True
            return True
        return False
    
    def predict(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Predict neuron activation scores
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            
        Returns:
            scores: Activation scores for each neuron [ffn_dim]
        """
        if not self.is_loaded:
            # Fallback to random prediction if no weights loaded
            return torch.rand(
                self.output_dim,
                device=self.device,
                dtype=self.dtype
            )
        
        # Use last token for prediction
        x = hidden_states[0, -1, :]  # [hidden_dim]
        
        # Two-layer prediction network
        x = F.linear(x, self.fc1_weight)
        x = F.relu(x)
        scores = F.linear(x, self.fc2_weight)
        
        return scores.squeeze()
    
    def select_top_k(self, hidden_states: torch.Tensor, k: int,
                     threshold: float = 0.0) -> torch.Tensor:
        """
        Predict and select top-k active neurons
        
        Args:
            hidden_states: Input tensor
            k: Number of neurons to select
            threshold: Minimum activation threshold
            
        Returns:
            indices: Indices of selected neurons [k]
        """
        scores = self.predict(hidden_states)
        
        # Get top-k indices
        values, indices = torch.topk(scores, min(k, scores.size(0)), sorted=False)
        
        # Filter by threshold
        mask = values > threshold
        filtered_indices = indices[mask]
        
        return filtered_indices.int()


class PredictorFactory:
    """Factory for creating and managing predictors for each layer"""
    
    def __init__(self, config, predictor_weights_dir: Optional[str] = None):
        self.config = config
        self.predictor_weights_dir = predictor_weights_dir
        self.predictors: dict[int, ActivationPredictor] = {}
    
    def get_predictor(self, layer_idx: int) -> ActivationPredictor:
        """Get or create predictor for a layer"""
        if layer_idx not in self.predictors:
            predictor = ActivationPredictor(
                input_dim=self.config.profile.hidden_dimension,
                output_dim=self.config.profile.ffn_dimension,
                device=self.config.primary_device,
                dtype=self.config.compute_dtype
            )
            
            # Try to load pre-trained weights
            if self.predictor_weights_dir:
                loaded = predictor.load_weights(
                    self.predictor_weights_dir,
                    layer_idx
                )
                if loaded:
                    print(f"Loaded predictor weights for layer {layer_idx}")
                else:
                    print(f"Using random predictor for layer {layer_idx}")
            
            self.predictors[layer_idx] = predictor
        
        return self.predictors[layer_idx]
    
    def create_fallback_predictor(self, layer_idx: int) -> Callable:
        """Create a simple fallback predictor function"""
        def fallback(hidden_states: torch.Tensor) -> torch.Tensor:
            # Random scores for fallback
            return torch.rand(
                self.config.profile.ffn_dimension,
                device=self.config.primary_device,
                dtype=self.config.compute_dtype
            )
        return fallback

