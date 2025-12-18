"""
Configuration module for DXA to 3D Spine Shape Estimation.

This module provides configuration classes and utilities for managing
experiment settings, hyperparameters, and model configurations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import yaml
import json
from omegaconf import DictConfig, OmegaConf


@dataclass
class ModelConfig:
    """Configuration for model architecture.
    
    Attributes:
        type (str): Model type ('transformer' or 'resnet50').
        params (Dict[str, Any]): Model-specific parameters.
        pretrained (bool): Whether to use pretrained weights.
        checkpoint_path (Optional[str]): Path to model checkpoint.
    """
    type: str = 'transformer'
    params: Dict[str, Any] = field(default_factory=lambda: {
        'input_channels': 1,
        'hidden_dim': 512,
        'num_heads': 8,
        'num_layers': 6,
        'num_points': 100,
        'dropout': 0.1
    })
    pretrained: bool = True
    checkpoint_path: Optional[str] = None


@dataclass
class OptimizerConfig:
    """Configuration for optimizer.
    
    Attributes:
        name (str): Optimizer name ('adam', 'adamw', 'sgd', 'rmsprop').
        lr (float): Learning rate.
        weight_decay (float): Weight decay for regularization.
        kwargs (Dict[str, Any]): Additional optimizer arguments.
    """
    name: str = 'adamw'
    lr: float = 1e-4
    weight_decay: float = 1e-5
    kwargs: Dict[str, Any] = field(default_factory=lambda: {
        'betas': (0.9, 0.999),
        'eps': 1e-8
    })


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduler.
    
    Attributes:
        enabled (bool): Whether to use scheduler.
        name (str): Scheduler name ('cosine', 'step', 'reduce_on_plateau').
        kwargs (Dict[str, Any]): Scheduler-specific arguments.
    """
    enabled: bool = True
    name: str = 'cosine'
    kwargs: Dict[str, Any] = field(default_factory=lambda: {
        'T_max': 100,
        'eta_min': 1e-6
    })


@dataclass
class LossConfig:
    """Configuration for loss function.
    
    Attributes:
        point_weight (float): Weight for point-wise loss.
        curvature_weight (float): Weight for curvature loss.
        smoothness_weight (float): Weight for smoothness loss.
        symmetry_weight (float): Weight for symmetry loss.
    """
    point_weight: float = 1.0
    curvature_weight: float = 0.5
    smoothness_weight: float = 0.3
    symmetry_weight: float = 0.2


@dataclass
class DataConfig:
    """Configuration for dataset and data loading.
    
    Attributes:
        train_dir (str): Training data directory.
        val_dir (str): Validation data directory.
        test_dir (str): Test data directory.
        target_size (tuple): Target image size (height, width).
        num_points (int): Number of points per curve.
        dataset_params (Dict[str, Any]): Additional dataset parameters.
    """
    train_dir: str = '/data/dxa_mri/train'
    val_dir: str = '/data/dxa_mri/val'
    test_dir: str = '/data/dxa_mri/test'
    target_size: tuple = (512, 512)
    num_points: int = 100
    dataset_params: Dict[str, Any] = field(default_factory=lambda: {
        'load_3d': False,
        'cache_data': False
    })


@dataclass
class TrainingConfig:
    """Main training configuration.
    
    Attributes:
        experiment_name (str): Name of the experiment.
        seed (int): Random seed for reproducibility.
        num_epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        num_workers (int): Number of dataloader workers.
        gradient_clip_val (float): Gradient clipping value.
        use_amp (bool): Whether to use mixed precision training.
        checkpoint_dir (str): Directory for saving checkpoints.
        save_best_only (bool): Whether to save only best models.
        max_checkpoints (int): Maximum number of checkpoints to keep.
        early_stopping_patience (int): Early stopping patience.
        early_stopping_delta (float): Minimum improvement for early stopping.
        visualize_predictions (bool): Whether to visualize predictions.
        visualization_dir (str): Directory for saving visualizations.
        use_wandb (bool): Whether to use Weights & Biases logging.
        wandb_project (str): W&B project name.
        tags (List[str]): Experiment tags.
        model (ModelConfig): Model configuration.
        optimizer (OptimizerConfig): Optimizer configuration.
        scheduler (SchedulerConfig): Scheduler configuration.
        loss (LossConfig): Loss configuration.
        data (DataConfig): Data configuration.
    """
    # Basic settings
    experiment_name: str = 'dxa_to_3d_spine'
    seed: int = 42
    
    # Training settings
    num_epochs: int = 100
    batch_size: int = 8
    num_workers: int = 4
    gradient_clip_val: float = 1.0
    use_amp: bool = True
    
    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    save_best_only: bool = True
    max_checkpoints: int = 3
    
    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_delta: float = 1e-4
    
    # Visualization
    visualize_predictions: bool = True
    visualization_dir: str = 'visualizations'
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = 'dxa-3d-spine'
    tags: List[str] = field(default_factory=lambda: ['spine', 'medical', '3d'])
    
    # Component configs
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)


@dataclass
class InferenceConfig:
    """Configuration for model inference.
    
    Attributes:
        model_checkpoint (str): Path to model checkpoint.
        input_dir (str): Directory containing input images.
        output_dir (str): Directory for saving outputs.
        batch_size (int): Inference batch size.
        device (str): Device for inference ('cuda' or 'cpu').
        save_visualizations (bool): Whether to save visualizations.
        save_3d_mesh (bool): Whether to save 3D mesh files.
        compute_metrics (bool): Whether to compute evaluation metrics.
        target_size (tuple): Target image size for inference.
        num_points (int): Number of points per curve.
    """
    model_checkpoint: str = 'checkpoints/best_model.pt'
    input_dir: str = 'test_images'
    output_dir: str = 'outputs'
    batch_size: int = 1
    device: str = 'cuda'
    save_visualizations: bool = True
    save_3d_mesh: bool = True
    compute_metrics: bool = False
    target_size: tuple = (512, 512)
    num_points: int = 100


def load_config(config_path: str) -> Union[TrainingConfig, InferenceConfig]:
    """Load configuration from file.
    
    Args:
        config_path: Path to configuration file (YAML or JSON).
        
    Returns:
        Configuration object.
        
    Raises:
        ValueError: If file format is not supported.
    """
    config_path = Path(config_path)
    
    if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
    # Determine config type
    if 'num_epochs' in config_dict:
        return TrainingConfig(**config_dict)
    else:
        return InferenceConfig(**config_dict)


def save_config(config: Union[TrainingConfig, InferenceConfig], save_path: str) -> None:
    """Save configuration to file.
    
    Args:
        config: Configuration object to save.
        save_path: Path to save configuration.
    """
    save_path = Path(save_path)
    
    # Convert to dictionary
    if hasattr(config, '__dict__'):
        config_dict = config.__dict__
    else:
        config_dict = OmegaConf.to_container(config)
        
    # Save based on extension
    if save_path.suffix == '.yaml' or save_path.suffix == '.yml':
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    elif save_path.suffix == '.json':
        with open(save_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    else:
        # Default to YAML
        save_path = save_path.with_suffix('.yaml')
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
            
    print(f"Configuration saved to {save_path}")


def create_default_configs(output_dir: str = 'configs') -> None:
    """Create default configuration files.
    
    Args:
        output_dir: Directory to save configuration files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create training config
    train_config = TrainingConfig()
    save_config(train_config, output_dir / 'train_config.yaml')
    
    # Create inference config
    inference_config = InferenceConfig()
    save_config(inference_config, output_dir / 'inference_config.yaml')
    
    # Create Hydra config
    hydra_config = {
        'defaults': ['_self_'],
        'hydra': {
            'run': {'dir': 'outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'},
            'sweep': {'dir': 'multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}'},
            'launcher': {'_target_': 'hydra._internal.core_plugins.basic_launcher.BasicLauncher'}
        },
        'experiment_name': 'dxa_to_3d_spine',
        'seed': 42,
        'num_epochs': 100,
        'batch_size': 8,
        'model': {
            'type': 'transformer',
            'params': {
                'input_channels': 1,
                'hidden_dim': 512,
                'num_heads': 8,
                'num_layers': 6
            }
        }
    }
    
    with open(output_dir / 'train.yaml', 'w') as f:
        yaml.dump(hydra_config, f, default_flow_style=False)
        
    print(f"Default configurations created in {output_dir}")


if __name__ == '__main__':
    # Create default configuration files
    create_default_configs()