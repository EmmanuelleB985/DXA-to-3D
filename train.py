"""
Training script for 3D Spine Shape Estimation from DXA images.

This module provides the training pipeline for spine shape estimation models,
including data loading, model training, validation, and checkpointing.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf
import hydra

from model import create_model, SpineShapeLoss
from dataset import DXADataset, get_data_transforms
from utils import (
    EarlyStopping,
    ModelCheckpoint,
    MetricTracker,
    visualize_predictions,
    set_random_seed
)
from config import TrainingConfig


class Trainer:
    """Main trainer class for spine shape estimation models.
    
    Handles the complete training pipeline including data loading,
    model optimization, validation, and checkpointing.
    
    Args:
        config (TrainingConfig): Training configuration object.
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        device (torch.device): Device to train on.
        
    Attributes:
        config (TrainingConfig): Training configuration.
        model (nn.Module): Model being trained.
        device (torch.device): Training device.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        criterion (nn.Module): Loss function.
        scaler (GradScaler): Mixed precision training scaler.
        metric_tracker (MetricTracker): Metrics tracking utility.
        checkpoint_manager (ModelCheckpoint): Checkpoint manager.
        early_stopping (EarlyStopping): Early stopping handler.
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device
    ) -> None:
        """Initialize the Trainer."""
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize loss function
        self.criterion = SpineShapeLoss(
            point_weight=config.loss.point_weight,
            curvature_weight=config.loss.curvature_weight,
            smoothness_weight=config.loss.smoothness_weight,
            symmetry_weight=config.loss.symmetry_weight
        )
        
        # Mixed precision training
        self.scaler = GradScaler() if config.use_amp else None
        
        # Initialize tracking utilities
        self.metric_tracker = MetricTracker()
        self.checkpoint_manager = ModelCheckpoint(
            checkpoint_dir=config.checkpoint_dir,
            best_only=config.save_best_only,
            max_keep=config.max_checkpoints
        )
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_delta
        )
        
        # Initialize wandb if enabled
        if config.use_wandb:
            self._init_wandb()
            
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration.
        
        Returns:
            Configured optimizer.
        """
        optimizer_map = {
            'adam': optim.Adam,
            'adamw': optim.AdamW,
            'sgd': optim.SGD,
            'rmsprop': optim.RMSprop
        }
        
        optimizer_class = optimizer_map.get(
            self.config.optimizer.name.lower(),
            optim.AdamW
        )
        
        return optimizer_class(
            self.model.parameters(),
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
            **self.config.optimizer.kwargs
        )
        
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on configuration.
        
        Returns:
            Configured scheduler or None.
        """
        if not self.config.scheduler.enabled:
            return None
            
        scheduler_map = {
            'cosine': optim.lr_scheduler.CosineAnnealingLR,
            'step': optim.lr_scheduler.StepLR,
            'reduce_on_plateau': optim.lr_scheduler.ReduceLROnPlateau,
            'exponential': optim.lr_scheduler.ExponentialLR
        }
        
        scheduler_class = scheduler_map.get(
            self.config.scheduler.name.lower(),
            optim.lr_scheduler.CosineAnnealingLR
        )
        
        return scheduler_class(
            self.optimizer,
            **self.config.scheduler.kwargs
        )
        
    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        wandb.init(
            project=self.config.wandb_project,
            name=self.config.experiment_name,
            config=OmegaConf.to_container(self.config),
            tags=self.config.tags
        )
        wandb.watch(self.model, log_freq=100)
        
    def train(self) -> Dict[str, float]:
        """Execute the complete training loop.
        
        Returns:
            Dictionary containing final metrics.
        """
        print(f"Starting training for {self.config.num_epochs} epochs...")
        
        for epoch in range(1, self.config.num_epochs + 1):
            # Training phase
            train_metrics = self._train_epoch(epoch)
            
            # Validation phase
            val_metrics = self._validate_epoch(epoch)
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['total_loss'])
                else:
                    self.scheduler.step()
                    
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Save checkpoint
            self.checkpoint_manager.save(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                metrics=val_metrics,
                config=self.config
            )
            
            # Early stopping check
            if self.early_stopping(val_metrics['total_loss']):
                print(f"Early stopping triggered at epoch {epoch}")
                break
                
        # Load best model
        best_checkpoint = self.checkpoint_manager.load_best()
        self.model.load_state_dict(best_checkpoint['model_state_dict'])
        
        # Final evaluation
        final_metrics = self._validate_epoch(epoch, final=True)
        
        return final_metrics
        
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number.
            
        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        epoch_metrics = MetricTracker()
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.config.num_epochs} [Train]",
            leave=False
        )
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device)
            targets = {
                key: val.to(self.device)
                for key, val in batch.items()
                if key != 'image'
            }
            
            # Forward pass with mixed precision
            if self.config.use_amp and self.scaler:
                with autocast():
                    predictions = self.model(images)
                    losses = self.criterion(predictions, targets)
                    loss = losses['total']
            else:
                predictions = self.model(images)
                losses = self.criterion(predictions, targets)
                loss = losses['total']
                
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.config.use_amp and self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_val
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_val
                )
                self.optimizer.step()
                
            # Update metrics
            epoch_metrics.update(losses)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
        return epoch_metrics.get_averages()
        
    def _validate_epoch(
        self,
        epoch: int,
        final: bool = False
    ) -> Dict[str, float]:
        """Validate for one epoch.
        
        Args:
            epoch: Current epoch number.
            final: Whether this is the final evaluation.
            
        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        epoch_metrics = MetricTracker()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(
                self.val_loader,
                desc=f"Epoch {epoch}/{self.config.num_epochs} [Val]",
                leave=False
            )
            
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                images = batch['image'].to(self.device)
                targets = {
                    key: val.to(self.device)
                    for key, val in batch.items()
                    if key != 'image'
                }
                
                # Forward pass
                if self.config.use_amp:
                    with autocast():
                        predictions = self.model(images)
                        losses = self.criterion(predictions, targets)
                else:
                    predictions = self.model(images)
                    losses = self.criterion(predictions, targets)
                    
                # Update metrics
                epoch_metrics.update(losses)
                
                # Store for visualization
                if batch_idx < 5 or final:
                    all_predictions.append(predictions)
                    all_targets.append(targets)
                    
                # Update progress bar
                pbar.set_postfix({'loss': f"{losses['total'].item():.4f}"})
                
        # Visualize predictions
        if self.config.visualize_predictions and (epoch % 10 == 0 or final):
            self._visualize_batch(all_predictions[0], all_targets[0], epoch)
            
        return epoch_metrics.get_averages()
        
    def _log_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ) -> None:
        """Log metrics to console and wandb.
        
        Args:
            epoch: Current epoch number.
            train_metrics: Training metrics.
            val_metrics: Validation metrics.
        """
        # Console logging
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
        print(f"  Val Loss: {val_metrics['total_loss']:.4f}")
        print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        # Wandb logging
        if self.config.use_wandb:
            log_dict = {
                'epoch': epoch,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            
            for key, value in train_metrics.items():
                log_dict[f'train/{key}'] = value
                
            for key, value in val_metrics.items():
                log_dict[f'val/{key}'] = value
                
            wandb.log(log_dict)
            
    def _visualize_batch(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        epoch: int
    ) -> None:
        """Visualize predictions for a batch.
        
        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            epoch: Current epoch number.
        """
        vis_dir = Path(self.config.visualization_dir) / f"epoch_{epoch}"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to numpy for visualization
        pred_np = {k: v.cpu().numpy() for k, v in predictions.items()}
        target_np = {k: v.cpu().numpy() for k, v in targets.items()}
        
        # Generate visualizations
        for idx in range(min(4, pred_np['coronal_centerline'].shape[0])):
            fig = visualize_predictions(
                pred_np,
                target_np,
                idx=idx
            )
            
            # Save figure
            fig.savefig(vis_dir / f"sample_{idx}.png", dpi=150, bbox_inches='tight')
            
            # Log to wandb
            if self.config.use_wandb:
                wandb.log({f"predictions/sample_{idx}": wandb.Image(fig)})
                
            plt.close(fig)


def main(config: DictConfig) -> None:
    """Main training function.
    
    Args:
        config: Hydra configuration object.
    """
    # Set random seed for reproducibility
    set_random_seed(config.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_model(
        model_type=config.model.type,
        **config.model.params
    )
    
    # Load dataset
    train_dataset = DXADataset(
        data_dir=config.data.train_dir,
        transform=get_data_transforms(training=True),
        **config.data.dataset_params
    )
    
    val_dataset = DXADataset(
        data_dir=config.data.val_dir,
        transform=get_data_transforms(training=False),
        **config.data.dataset_params
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Create trainer
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    # Train model
    final_metrics = trainer.train()
    
    # Print final results
    print("\nTraining completed!")
    print("Final validation metrics:")
    for key, value in final_metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig
    
    @hydra.main(version_base=None, config_path="configs", config_name="train")
    def hydra_main(cfg: DictConfig) -> None:
        """Hydra wrapper for main function."""
        main(cfg)
        
    hydra_main()
