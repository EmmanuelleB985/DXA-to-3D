"""
Utility functions for 3D Spine Shape Estimation.

This module provides utility functions for training, visualization,
and data processing in the spine shape estimation pipeline.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
from scipy.interpolate import UnivariateSpline
from scipy.spatial.distance import euclidean
import cv2
import trimesh
from sklearn.metrics import mean_squared_error, mean_absolute_error


def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value. Default: 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stopping handler for training.
    
    Monitors a metric and stops training when it stops improving.
    
    Args:
        patience (int): Number of epochs to wait before stopping. Default: 10.
        min_delta (float): Minimum change to qualify as improvement. Default: 1e-4.
        mode (str): One of 'min' or 'max'. Default: 'min'.
        
    Attributes:
        patience (int): Patience value.
        min_delta (float): Minimum delta for improvement.
        mode (str): Optimization mode.
        best_value (float): Best metric value seen.
        counter (int): Counter for patience.
        early_stop (bool): Whether to stop training.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'min'
    ) -> None:
        """Initialize EarlyStopping."""
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = np.inf if mode == 'min' else -np.inf
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, value: float) -> bool:
        """Check if training should stop.
        
        Args:
            value: Current metric value.
            
        Returns:
            True if training should stop, False otherwise.
        """
        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
            
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop
        
    def reset(self) -> None:
        """Reset the early stopping state."""
        self.best_value = np.inf if self.mode == 'min' else -np.inf
        self.counter = 0
        self.early_stop = False


class ModelCheckpoint:
    """Model checkpoint manager.
    
    Handles saving and loading of model checkpoints during training.
    
    Args:
        checkpoint_dir (str): Directory to save checkpoints.
        best_only (bool): Whether to save only best models. Default: True.
        max_keep (int): Maximum number of checkpoints to keep. Default: 3.
        
    Attributes:
        checkpoint_dir (Path): Checkpoint directory path.
        best_only (bool): Save best only flag.
        max_keep (int): Maximum checkpoints to keep.
        best_metric (float): Best metric value.
        checkpoints (List[Path]): List of saved checkpoints.
    """
    
    def __init__(
        self,
        checkpoint_dir: str = 'checkpoints',
        best_only: bool = True,
        max_keep: int = 3
    ) -> None:
        """Initialize ModelCheckpoint."""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_only = best_only
        self.max_keep = max_keep
        self.best_metric = np.inf
        self.checkpoints: List[Path] = []
        
    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        config: Any
    ) -> None:
        """Save a checkpoint.
        
        Args:
            model: Model to save.
            optimizer: Optimizer state to save.
            epoch: Current epoch.
            metrics: Current metrics.
            config: Training configuration.
        """
        # Check if should save
        current_metric = metrics.get('total_loss', np.inf)
        is_best = current_metric < self.best_metric
        
        if self.best_only and not is_best:
            return
            
        # Update best metric
        if is_best:
            self.best_metric = current_metric
            
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': config,
            'best_metric': self.best_metric
        }
        
        # Save checkpoint
        filename = f"epoch_{epoch:04d}_loss_{current_metric:.4f}.pt"
        if is_best:
            filename = f"best_{filename}"
            
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        self.checkpoints.append(filepath)
        
        print(f"Saved checkpoint: {filepath}")
        
        # Remove old checkpoints
        if len(self.checkpoints) > self.max_keep:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists() and 'best' not in old_checkpoint.name:
                old_checkpoint.unlink()
                
    def load_best(self) -> Dict[str, Any]:
        """Load the best checkpoint.
        
        Returns:
            Dictionary containing checkpoint data.
            
        Raises:
            FileNotFoundError: If no best checkpoint is found.
        """
        best_checkpoints = list(self.checkpoint_dir.glob("best_*.pt"))
        if not best_checkpoints:
            raise FileNotFoundError("No best checkpoint found")
            
        # Get most recent best checkpoint
        best_checkpoint = sorted(best_checkpoints)[-1]
        return torch.load(best_checkpoint)
        
    def load_latest(self) -> Dict[str, Any]:
        """Load the latest checkpoint.
        
        Returns:
            Dictionary containing checkpoint data.
            
        Raises:
            FileNotFoundError: If no checkpoints are found.
        """
        all_checkpoints = list(self.checkpoint_dir.glob("*.pt"))
        if not all_checkpoints:
            raise FileNotFoundError("No checkpoints found")
            
        # Get most recent checkpoint
        latest_checkpoint = sorted(all_checkpoints)[-1]
        return torch.load(latest_checkpoint)


class MetricTracker:
    """Metric tracking utility for training.
    
    Tracks and averages metrics during training and validation.
    
    Attributes:
        metrics (Dict[str, List[float]]): Dictionary of metric values.
    """
    
    def __init__(self) -> None:
        """Initialize MetricTracker."""
        self.metrics: Dict[str, List[float]] = {}
        
    def update(self, metric_dict: Dict[str, Union[float, torch.Tensor]]) -> None:
        """Update metrics with new values.
        
        Args:
            metric_dict: Dictionary of metric values.
        """
        for key, value in metric_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
                
            if key not in self.metrics:
                self.metrics[key] = []
                
            self.metrics[key].append(value)
            
    def get_averages(self) -> Dict[str, float]:
        """Get average values for all metrics.
        
        Returns:
            Dictionary of average metric values.
        """
        return {
            f"{key}_loss": np.mean(values)
            for key, values in self.metrics.items()
        }
        
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics = {}


def visualize_predictions(
    predictions: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    idx: int = 0
) -> Figure:
    """Visualize predicted and target spine curves.
    
    Args:
        predictions: Dictionary of predicted curves.
        targets: Dictionary of target curves.
        idx: Sample index to visualize.
        
    Returns:
        Matplotlib figure object.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Define curve pairs and titles
    curve_pairs = [
        ('coronal_centerline', 'Coronal Centerline'),
        ('coronal_lateral', 'Coronal Lateral'),
        ('sagittal_centerline', 'Sagittal Centerline'),
        ('sagittal_lateral', 'Sagittal Lateral')
    ]
    
    for ax, (key, title) in zip(axes.flat, curve_pairs):
        if key in predictions and key in targets:
            pred = predictions[key][idx]
            target = targets[key][idx]
            
            # Plot curves
            ax.plot(pred[:, 0], pred[:, 1], 'b-', label='Predicted', linewidth=2)
            ax.plot(target[:, 0], target[:, 1], 'r--', label='Ground Truth', linewidth=2)
            
            # Calculate error
            error = np.mean(np.linalg.norm(pred - target, axis=1))
            
            ax.set_title(f"{title}\nMean Error: {error:.3f}")
            ax.set_xlabel('X coordinate')
            ax.set_ylabel('Y coordinate')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
    plt.suptitle(f'Spine Curve Predictions - Sample {idx}')
    plt.tight_layout()
    
    return fig


def reconstruct_3d_spine(
    coronal_curves: Dict[str, np.ndarray],
    sagittal_curves: Dict[str, np.ndarray],
    num_vertebrae: int = 24
) -> np.ndarray:
    """Reconstruct 3D spine from orthogonal projections.
    
    Args:
        coronal_curves: Dictionary with coronal view curves.
        sagittal_curves: Dictionary with sagittal view curves.
        num_vertebrae: Number of vertebrae to reconstruct.
        
    Returns:
        3D coordinates of spine points (num_points, 3).
    """
    # Extract centerlines
    coronal_center = coronal_curves['centerline']
    sagittal_center = sagittal_curves['centerline']
    
    # Ensure same number of points
    num_points = min(len(coronal_center), len(sagittal_center))
    
    # Combine projections to get 3D coordinates
    spine_3d = np.zeros((num_points, 3))
    spine_3d[:, 0] = coronal_center[:num_points, 0]  # X from coronal
    spine_3d[:, 1] = sagittal_center[:num_points, 0]  # Y from sagittal
    spine_3d[:, 2] = (coronal_center[:num_points, 1] + sagittal_center[:num_points, 1]) / 2  # Z average
    
    return spine_3d


def smooth_curve(
    curve: np.ndarray,
    smoothing_factor: float = 0.5,
    num_points: Optional[int] = None
) -> np.ndarray:
    """Smooth a curve using spline interpolation.
    
    Args:
        curve: Input curve points (N, 2).
        smoothing_factor: Smoothing factor for spline.
        num_points: Number of output points.
        
    Returns:
        Smoothed curve points.
    """
    if num_points is None:
        num_points = len(curve)
        
    # Parameterize curve
    t = np.linspace(0, 1, len(curve))
    t_new = np.linspace(0, 1, num_points)
    
    # Fit splines
    spline_x = UnivariateSpline(t, curve[:, 0], s=smoothing_factor)
    spline_y = UnivariateSpline(t, curve[:, 1], s=smoothing_factor)
    
    # Evaluate at new points
    smoothed = np.column_stack([
        spline_x(t_new),
        spline_y(t_new)
    ])
    
    return smoothed


def calculate_cobb_angle(spine_curve: np.ndarray) -> float:
    """Calculate Cobb angle from spine curve.
    
    The Cobb angle is a measure of spinal curvature used in
    scoliosis assessment.
    
    Args:
        spine_curve: Spine curve points (N, 2).
        
    Returns:
        Cobb angle in degrees.
    """
    # Find points with maximum curvature
    curvatures = []
    for i in range(1, len(spine_curve) - 1):
        p1 = spine_curve[i - 1]
        p2 = spine_curve[i]
        p3 = spine_curve[i + 1]
        
        # Calculate angle
        v1 = p1 - p2
        v2 = p3 - p2
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        curvatures.append(angle)
        
    # Find maximum curvature points
    curvatures = np.array(curvatures)
    max_idx = np.argmax(curvatures)
    
    # Calculate Cobb angle (simplified)
    if max_idx > 5 and max_idx < len(spine_curve) - 5:
        upper_tangent = spine_curve[max_idx - 5] - spine_curve[max_idx - 3]
        lower_tangent = spine_curve[max_idx + 3] - spine_curve[max_idx + 5]
        
        angle = np.arccos(
            np.dot(upper_tangent, lower_tangent) /
            (np.linalg.norm(upper_tangent) * np.linalg.norm(lower_tangent))
        )
        
        return np.degrees(angle)
        
    return 0.0


def save_3d_spine_mesh(
    spine_3d: np.ndarray,
    filename: str,
    radius: float = 5.0
) -> None:
    """Save 3D spine as a mesh file.
    
    Args:
        spine_3d: 3D spine coordinates (N, 3).
        filename: Output filename (.ply, .obj, .stl).
        radius: Radius for vertebrae representation.
    """
    # Create spheres for each vertebra
    meshes = []
    for point in spine_3d:
        sphere = trimesh.primitives.Sphere(radius=radius, center=point)
        meshes.append(sphere)
        
    # Combine meshes
    combined = trimesh.util.concatenate(meshes)
    
    # Save to file
    combined.export(filename)
    print(f"Saved 3D spine mesh to {filename}")


def compute_spine_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """Compute comprehensive metrics for spine estimation.
    
    Args:
        predictions: Predicted spine points.
        targets: Target spine points.
        
    Returns:
        Dictionary of computed metrics.
    """
    metrics = {}
    
    # Point-wise metrics
    metrics['mse'] = mean_squared_error(targets.flatten(), predictions.flatten())
    metrics['mae'] = mean_absolute_error(targets.flatten(), predictions.flatten())
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # Curve-specific metrics
    point_distances = np.linalg.norm(predictions - targets, axis=-1)
    metrics['mean_distance'] = np.mean(point_distances)
    metrics['max_distance'] = np.max(point_distances)
    metrics['std_distance'] = np.std(point_distances)
    
    # Percentile errors
    metrics['error_95th'] = np.percentile(point_distances, 95)
    metrics['error_99th'] = np.percentile(point_distances, 99)
    
    return metrics


def load_dxa_image(
    filepath: str,
    target_size: Tuple[int, int] = (512, 512)
) -> np.ndarray:
    """Load and preprocess DXA image.
    
    Args:
        filepath: Path to DXA image file.
        target_size: Target size for resizing.
        
    Returns:
        Preprocessed image array.
    """
    # Load image
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {filepath}")
        
    # Resize
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_uint8 = (image * 255).astype(np.uint8)
    image_enhanced = clahe.apply(image_uint8)
    image = image_enhanced.astype(np.float32) / 255.0
    
    return image