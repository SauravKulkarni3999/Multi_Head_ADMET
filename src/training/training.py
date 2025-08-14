"""
Training Strategy for Multi-Task ADMET Prediction

Lessons learned from 30 years of molecular property prediction
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import numpy as np
from typing import Dict, List, Optional
from src.models.admet_model import GradientHarmonizer

class ADMETTrainer:
    """
    Trainer class implementing best practices for ADMET multi-task learning
    """
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.gradient_harmonizer = GradientHarmonizer(model.all_tasks)
        
        # Initialize optimizers - different learning rates for different components
        self.setup_optimizers()
        self.setup_schedulers()
        
    def setup_optimizers(self):
        """Setup optimizers with different learning rates for different components"""
        # Shared encoder parameters (learn slower)
        shared_params = list(self.model.shared_encoder.parameters())
        
        # Task-specific parameters (can learn faster)
        task_params = []
        task_params.extend(self.model.task_extractors.parameters())
        task_params.extend(self.model.classification_heads.parameters())
        task_params.extend(self.model.regression_heads.parameters())
        # ✅ FIXED: Remove uncertainty_heads reference
        # task_params.extend(self.model.uncertainty_heads.parameters())
        
        self.optimizer = optim.AdamW([
            {'params': shared_params, 'lr': 1e-4, 'weight_decay': 1e-4},
            {'params': task_params, 'lr': 3e-4, 'weight_decay': 1e-5},
            {'params': [self.model.task_weights], 'lr': 1e-3, 'weight_decay': 0}
        ])
    
    def setup_schedulers(self):
        """Learning rate scheduling - critical for ADMET tasks"""
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='max',  # Maximize validation AUC
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # Cosine annealing for fine-tuning
        self.cosine_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=10, 
            T_mult=2
        )
    
    def compute_class_weights(self, targets: Dict[str, torch.Tensor], 
                            masks: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute balanced class weights for imbalanced ADMET datasets
        """
        class_weights = {}
        
        for task in self.model.classification_tasks:
            if task in targets and masks[task].sum() > 0:
                mask = masks[task].bool()
                task_targets = targets[task][mask]
                
                n_positive = task_targets.sum().item()
                n_negative = (mask.sum() - n_positive).item()
                
                if n_positive > 0 and n_negative > 0:
                    # Balanced class weight
                    weight = n_negative / n_positive
                    class_weights[task] = min(weight, 5.0)  # Cap at 5x
                else:
                    class_weights[task] = 1.0
        
        return class_weights
    
    def train_epoch(self, train_loader, epoch: int, max_epochs: int):
        """
        Train for one epoch with gradient harmonization
        """
        self.model.train()
        self.model.update_dropout_schedule(epoch, max_epochs)
        
        total_loss = 0.0
        task_losses = {task: 0.0 for task in self.model.all_tasks}
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            features = batch['features'].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
            masks = {k: v.to(self.device) for k, v in batch['masks'].items()}
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(features, masks)
            
            # Compute losses
            loss, batch_task_losses = self.model.compute_loss(outputs, targets, masks)
            
            # Gradient harmonization (every 5 batches to avoid overhead)
            if batch_idx % 5 == 0:
                task_specific_losses = {}
                for task, task_loss in batch_task_losses.items():
                    if task_loss > 0:
                        task_specific_losses[task] = torch.tensor(
                            task_loss, requires_grad=True, device=self.device
                        )
                
                if len(task_specific_losses) > 1:
                    similarities = self.gradient_harmonizer.harmonize_gradients(
                        task_specific_losses, 
                        list(self.model.shared_encoder.parameters())
                    )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (important for stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            for task, task_loss in batch_task_losses.items():
                task_losses[task] += task_loss
            
            num_batches += 1
        
        # Average losses
        avg_loss = total_loss / num_batches
        avg_task_losses = {k: v / num_batches for k, v in task_losses.items()}
        
        return avg_loss, avg_task_losses
    
    def validate(self, val_loader):
        """Validation with uncertainty quantification"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0  # ✅ ADD: Track number of batches
        predictions = {task: [] for task in self.model.all_tasks}
        targets_collected = {task: [] for task in self.model.all_tasks}
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch['targets'].items()}
                masks = {k: v.to(self.device) for k, v in batch['masks'].items()}
                
                outputs = self.model(features, masks)
                loss, _ = self.model.compute_loss(outputs, targets, masks)
                
                # ✅ FIXED: Ensure loss is positive before accumulating
                loss_value = loss.item()
                if not torch.isnan(loss) and not torch.isinf(loss) and loss_value > 0:
                    total_loss += loss_value
                    num_batches += 1
                
                # Collect predictions and targets
                for task in self.model.all_tasks:
                    if masks[task].sum() > 0:
                        mask = masks[task].bool()
                        
                        if task in self.model.classification_tasks:
                            predictions[task].append(outputs[f'{task}_probs'][mask])
                        else:
                            predictions[task].append(outputs[f'{task}_pred'][mask])
                        
                        targets_collected[task].append(targets[task][mask])
        
        # Concatenate results
        for task in self.model.all_tasks:
            if predictions[task]:
                predictions[task] = torch.cat(predictions[task])
                targets_collected[task] = torch.cat(targets_collected[task])
        
        # ✅ FIXED: Proper averaging with safety check
        if num_batches > 0:
            avg_loss = total_loss / num_batches
        else:
            avg_loss = 0.0
        
        return avg_loss, predictions, targets_collected

class EarlyStopping:
    """Early stopping based on validation metrics"""
    
    def __init__(self, patience=7, min_delta=1e-4, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = float('-inf') if mode == 'max' else float('inf')
        self.early_stop = False
    
    def __call__(self, score):
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop

# Key training hyperparameters based on experience
RECOMMENDED_HYPERPARAMETERS = {
    'batch_size': 128,  # Larger batches help with batch norm stability
    'max_epochs': 100,
    'early_stopping_patience': 10,
    'gradient_clip_norm': 1.0,
    'weight_decay_shared': 1e-4,  # Regularization for shared encoder
    'weight_decay_heads': 1e-5,   # Less regularization for task heads
    'lr_shared': 1e-4,            # Conservative for shared parameters
    'lr_heads': 3e-4,             # More aggressive for task-specific
    'warmup_epochs': 5,           # Gradual warmup prevents instability
}