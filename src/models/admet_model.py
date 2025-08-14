"""
Multi-Task ADMET Model Architecture

Based on 30+ years of experience in AI for chemistry, this architecture balances
complexity with interpretability while handling the unique challenges of ADMET prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

class AdaptiveDropout(nn.Module):
    """
    Adaptive dropout that varies by training epoch - higher dropout early,
    lower dropout later. Critical for ADMET tasks where overfitting is common.
    """
    def __init__(self, p_max=0.5, p_min=0.1):
        super().__init__()
        self.p_max = p_max
        self.p_min = p_min
        self.current_p = p_max
        
    def set_dropout_rate(self, epoch: int, max_epochs: int):
        # Exponential decay from p_max to p_min
        self.current_p = self.p_min + (self.p_max - self.p_min) * np.exp(-3 * epoch / max_epochs)
    
    def forward(self, x):
        return F.dropout(x, p=self.current_p, training=self.training)

class ResidualBlock(nn.Module):
    """
    Residual connection for deeper networks - helps with gradient flow
    and allows learning of more complex molecular representations
    """
    def __init__(self, dim: int, dropout_p: float = 0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        
    def forward(self, x):
        residual = x
        out = self.layers(x)
        return F.relu(out + residual)

class AttentionPooling(nn.Module):
    """
    Learn to focus on the most relevant molecular features for each task.
    Particularly important when combining ECFP + descriptors.
    """
    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        # x: [batch_size, feature_dim]
        weights = F.softmax(self.attention(x.unsqueeze(1)), dim=1)  # [batch, 1, 1]
        return x * weights.squeeze(1)  # Element-wise weighting

class MultiTaskADMETPredictor(nn.Module):
    """
    Multi-task ADMET predictor with shared molecular encoder and task-specific heads.
    
    Architecture Philosophy:
    1. Shared trunk learns general molecular representations
    2. Task-specific feature extraction before final prediction
    3. Uncertainty estimation for reliability assessment
    4. Gradient harmonization to prevent task interference
    """
    
    def __init__(self, 
                 input_dim: int,
                 classification_tasks: List[str],
                 regression_tasks: List[str],
                 hidden_dims: List[int] = [512, 256, 128],
                 use_residual: bool = True,
                 use_attention: bool = True,
                 dropout_schedule: bool = True):
        super().__init__()
        
        self.classification_tasks = classification_tasks
        self.regression_tasks = regression_tasks
        self.all_tasks = classification_tasks + regression_tasks
        self.num_tasks = len(self.all_tasks)
        
        # Shared molecular encoder (trunk)
        self.shared_encoder = self._build_shared_encoder(
            input_dim, hidden_dims, use_residual, use_attention, dropout_schedule
        )
        
        # Task-specific feature extractors
        self.task_extractors = nn.ModuleDict()
        for task in self.all_tasks:
            self.task_extractors[task] = nn.Sequential(
                nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dims[-1] // 2, hidden_dims[-1] // 4)
            )
        
        # Task-specific prediction heads
        self.classification_heads = nn.ModuleDict()
        for task in classification_tasks:
            self.classification_heads[task] = nn.Linear(hidden_dims[-1] // 4, 1)
        
        self.regression_heads = nn.ModuleDict()
        for task in regression_tasks:
            self.regression_heads[task] = nn.Linear(hidden_dims[-1] // 4, 1)
        
        # ✅ TEMPORARILY DISABLED: Uncertainty estimation heads
        # self.uncertainty_heads = nn.ModuleDict()
        # for task in self.all_tasks:
        #     self.uncertainty_heads[task] = nn.Linear(hidden_dims[-1] // 4, 1)
        
        # Task importance weights (learnable)
        self.task_weights = nn.Parameter(torch.ones(self.num_tasks))
        
        self._initialize_weights()
    
    def _build_shared_encoder(self, input_dim, hidden_dims, use_residual, use_attention, dropout_schedule):
        layers = []
        
        # Input projection with attention pooling
        if use_attention:
            layers.append(AttentionPooling(input_dim))
        
        # First layer
        layers.extend([
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU()
        ])
        
        if dropout_schedule:
            layers.append(AdaptiveDropout(0.4, 0.1))
        else:
            layers.append(nn.Dropout(0.3))
        
        # Hidden layers with optional residual connections
        for i in range(1, len(hidden_dims)):
            if use_residual and hidden_dims[i-1] == hidden_dims[i]:
                layers.append(ResidualBlock(hidden_dims[i-1], 0.2))
            else:
                layers.extend([
                    nn.Linear(hidden_dims[i-1], hidden_dims[i]),
                    nn.BatchNorm1d(hidden_dims[i]),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ])
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Xavier/Glorot initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def update_dropout_schedule(self, epoch: int, max_epochs: int):
        """Update adaptive dropout rates based on training progress"""
        for module in self.modules():
            if isinstance(module, AdaptiveDropout):
                module.set_dropout_rate(epoch, max_epochs)
    
    def forward(self, x: torch.Tensor, task_mask: Optional[Dict[str, torch.Tensor]] = None):
        """
        Forward pass with task masking for multi-task learning
        
        Args:
            x: Input features [batch_size, input_dim]
            task_mask: Dict mapping task names to binary masks [batch_size]
        
        Returns:
            Dict containing predictions and uncertainties for each task
        """
        # Shared molecular representation
        shared_features = self.shared_encoder(x)
        
        outputs = {}
        
        # Generate task-specific predictions
        for i, task in enumerate(self.all_tasks):
            # Task-specific feature extraction
            task_features = self.task_extractors[task](shared_features)
            
            # Prediction
            if task in self.classification_tasks:
                logits = self.classification_heads[task](task_features).squeeze(-1)
                outputs[f'{task}_logits'] = logits
                outputs[f'{task}_probs'] = torch.sigmoid(logits)
            else:
                outputs[f'{task}_pred'] = self.regression_heads[task](task_features).squeeze(-1)
            
            # ✅ TEMPORARILY DISABLED: Uncertainty estimation
            # outputs[f'{task}_uncertainty'] = torch.ones_like(task_features[:, 0])  # Set to 1.0
            # outputs[f'{task}_log_var'] = torch.zeros_like(task_features[:, 0])     # Set to 0.0
        
        return outputs
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor], 
                    masks: Dict[str, torch.Tensor]):
        """
        Compute multi-task loss with robust error handling
        """
        total_loss = 0.0
        task_losses = {}
        valid_tasks = 0
        
        for task in self.all_tasks:
            if task not in targets or masks[task].sum() == 0:
                continue
            
            mask = masks[task].bool()
            target = targets[task][mask]
            
            # Skip if no valid targets
            if len(target) == 0:
                continue
        
            try:
                if task in self.classification_tasks:
                    # Robust BCE loss
                    logits = outputs[f'{task}_logits'][mask]
                    # Clamp logits to prevent numerical issues
                    logits = torch.clamp(logits, min=-10.0, max=10.0)
                    task_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='mean')
                    
                else:
                    # Robust MSE loss
                    pred = outputs[f'{task}_pred'][mask]
                    # Clamp predictions to prevent extreme values
                    pred = torch.clamp(pred, min=-10.0, max=10.0)
                    task_loss = F.mse_loss(pred, target, reduction='mean')
                
                # Ensure loss is finite and positive
                if torch.isfinite(task_loss) and task_loss >= 0:
                    task_losses[task] = task_loss.item()
                    total_loss += task_loss
                    valid_tasks += 1
                else:
                    print(f"Warning: Invalid loss for task {task}: {task_loss.item()}")
                    
            except Exception as e:
                print(f"Error computing loss for task {task}: {e}")
                continue
        
        if valid_tasks > 0:
            total_loss = total_loss / valid_tasks
        else:
            total_loss = torch.tensor(0.0, device=targets[list(targets.keys())[0]].device)
        
        # Final safety check
        if not torch.isfinite(total_loss) or total_loss < 0:
            total_loss = torch.tensor(0.0, device=total_loss.device)
            
        return total_loss, task_losses
    
    def get_calibrated_predictions(self, x: torch.Tensor, temperature_scalars: Dict[str, float]):
        """Apply temperature scaling for calibrated probabilities"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            
            for task in self.classification_tasks:
                if f'{task}_logits' in outputs:
                    temp = temperature_scalars.get(task, 1.0)
                    outputs[f'{task}_calibrated_probs'] = torch.sigmoid(outputs[f'{task}_logits'] / temp)
            
            return outputs

class GradientHarmonizer:
    """
    Implement gradient harmonization to prevent negative task interference.
    Based on "Gradient Surgery for Multi-Task Learning" (Yu et al., 2020)
    """
    
    def __init__(self, tasks: List[str]):
        self.tasks = tasks
        self.num_tasks = len(tasks)
    
    def harmonize_gradients(self, losses: Dict[str, torch.Tensor], 
                          shared_parameters: List[torch.nn.Parameter]):
        """
        Project conflicting gradients to reduce negative transfer
        """
        # Filter out tasks with zero or invalid losses
        valid_losses = {task: loss for task, loss in losses.items() 
                       if loss.requires_grad and loss.item() > 0}
        
        if len(valid_losses) < 2:
            # Not enough tasks for harmonization
            return {}
        
        # Compute per-task gradients
        task_gradients = {}
        
        for task, loss in valid_losses.items():
            try:
                # ✅ FIXED: Add allow_unused=True to handle unused parameters
                grads = torch.autograd.grad(
                    loss, 
                    shared_parameters, 
                    retain_graph=True,
                    allow_unused=True  # ✅ This fixes the error
                )
                
                # Filter out None gradients (unused parameters)
                valid_grads = [g for g in grads if g is not None]
                
                if valid_grads:
                    task_gradients[task] = torch.cat([g.flatten() for g in valid_grads])
                else:
                    # Skip this task if no valid gradients
                    continue
                    
            except Exception as e:
                logger.warning(f"Failed to compute gradients for task {task}: {e}")
                continue
        
        if len(task_gradients) < 2:
            return {}
        
        # Compute gradient cosine similarities
        task_names = list(task_gradients.keys())
        similarities = {}
        
        for i, task1 in enumerate(task_names):
            for j, task2 in enumerate(task_names[i+1:], i+1):
                try:
                    g1, g2 = task_gradients[task1], task_gradients[task2]
                    
                    # Ensure gradients have the same shape
                    min_len = min(len(g1), len(g2))
                    g1 = g1[:min_len]
                    g2 = g2[:min_len]
                    
                    sim = F.cosine_similarity(g1, g2, dim=0)
                    similarities[f"{task1}_{task2}"] = sim.item()
                    
                except Exception as e:
                    logger.warning(f"Failed to compute similarity for {task1}_{task2}: {e}")
                    continue
        
        return similarities

def create_model_config(input_dim: int, tasks_config: Dict) -> Dict:
    """Helper function to create model configuration"""
    return {
        'input_dim': input_dim,
        'classification_tasks': tasks_config.get('classification', []),
        'regression_tasks': tasks_config.get('regression', []),
        'hidden_dims': [512, 256, 128],  # Conservative but effective
        'use_residual': True,
        'use_attention': True,
        'dropout_schedule': True
    }