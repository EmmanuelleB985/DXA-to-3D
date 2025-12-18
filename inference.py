"""
Inference script for 3D Spine Shape Estimation from DXA images.

This module provides inference capabilities for trained spine estimation models,
including batch processing, visualization, and 3D reconstruction.
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd

from model import create_model
from config import InferenceConfig, load_config
from utils import (
    load_dxa_image,
    visualize_predictions,
    reconstruct_3d_spine,
    save_3d_spine_mesh,
    calculate_angle,
    compute_spine_metrics,
    smooth_curve,
    set_random_seed
)


class SpineEstimator:
    """Main class for spine shape estimation inference.
    
    This class handles model loading, inference, and result generation
    for spine shape estimation from DXA images.
    
    Args:
        config (InferenceConfig): Inference configuration.
        device (Optional[torch.device]): Device for inference.
        
    Attributes:
        config (InferenceConfig): Configuration object.
        device (torch.device): Inference device.
        model (nn.Module): Loaded model.
        results (List[Dict]): List of inference results.
    """
    
    def __init__(
        self,
        config: InferenceConfig,
        device: Optional[torch.device] = None
    ) -> None:
        """Initialize SpineEstimator."""
        self.config = config
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Results storage
        self.results: List[Dict[str, Any]] = []
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"SpineEstimator initialized on {self.device}")
        
    def _load_model(self) -> nn.Module:
        """Load trained model from checkpoint.
        
        Returns:
            Loaded model in evaluation mode.
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist.
        """
        checkpoint_path = Path(self.config.model_checkpoint)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract model configuration
        if 'config' in checkpoint:
            model_config = checkpoint['config'].model
        else:
            # Use default configuration
            from config import ModelConfig
            model_config = ModelConfig()
            
        # Create model
        model = create_model(
            model_type=model_config.type,
            **model_config.params
        )
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model = model.to(self.device)
        
        print(f"Model loaded from {checkpoint_path}")
        
        return model
        
    def predict_single(
        self,
        image_path: str,
        visualize: bool = True,
        save_results: bool = True
    ) -> Dict[str, np.ndarray]:
        """Perform inference on a single image.
        
        Args:
            image_path: Path to input DXA image.
            visualize: Whether to generate visualizations.
            save_results: Whether to save results to disk.
            
        Returns:
            Dictionary containing predicted curves and metadata.
        """
        # Load and preprocess image
        image = load_dxa_image(image_path, self.config.target_size)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()
        image_tensor = image_tensor.to(self.device)
        
        # Perform inference
        with torch.no_grad():
            predictions = self.model(image_tensor)
            
        # Convert to numpy
        results = {}
        for key, value in predictions.items():
            results[key] = value.cpu().numpy()[0]  # Remove batch dimension
            
        # Add metadata
        results['image_path'] = image_path
        results['timestamp'] = datetime.now().isoformat()
        
        # Calculate Modified Ferguson Angle
        if 'coronal_centerline' in results:
            results['angle'] = calculate_angle(results['coronal_centerline'])
            
        # Reconstruct 3D spine
        if 'coronal_centerline' in results and 'sagittal_centerline' in results:
            spine_3d = reconstruct_3d_spine(
                {'centerline': results['coronal_centerline']},
                {'centerline': results['sagittal_centerline']}
            )
            results['spine_3d'] = spine_3d
            
        # Save results
        if save_results:
            self._save_results(results, Path(image_path).stem)
            
        # Visualize
        if visualize and self.config.save_visualizations:
            self._visualize_results(results, Path(image_path).stem)
            
        return results
        
    def predict_batch(
        self,
        image_paths: List[str],
        batch_size: Optional[int] = None
    ) -> List[Dict[str, np.ndarray]]:
        """Perform batch inference on multiple images.
        
        Args:
            image_paths: List of image paths.
            batch_size: Batch size for inference.
            
        Returns:
            List of prediction dictionaries.
        """
        if batch_size is None:
            batch_size = self.config.batch_size
            
        all_results = []
        
        # Process in batches
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            # Load batch
            for path in batch_paths:
                image = load_dxa_image(path, self.config.target_size)
                batch_images.append(image)
                
            # Stack and convert to tensor
            batch_tensor = np.stack(batch_images)
            batch_tensor = torch.from_numpy(batch_tensor).unsqueeze(1).float()
            batch_tensor = batch_tensor.to(self.device)
            
            # Perform inference
            with torch.no_grad():
                predictions = self.model(batch_tensor)
                
            # Process each sample in batch
            for j, path in enumerate(batch_paths):
                results = {}
                for key, value in predictions.items():
                    results[key] = value[j].cpu().numpy()
                    
                results['image_path'] = path
                results['timestamp'] = datetime.now().isoformat()
                
                # Calculate metrics
                if 'coronal_centerline' in results:
                    results['angle'] = calculate_angle(results['coronal_centerline'])
                    
                all_results.append(results)
                
        self.results = all_results
        return all_results
        
    def predict_directory(
        self,
        input_dir: Optional[str] = None
    ) -> List[Dict[str, np.ndarray]]:
        """Process all images in a directory.
        
        Args:
            input_dir: Input directory path. Uses config if not provided.
            
        Returns:
            List of prediction results.
        """
        if input_dir is None:
            input_dir = self.config.input_dir
            
        input_path = Path(input_dir)
        
        # Find all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(input_path.glob(f'*{ext}'))
            image_paths.extend(input_path.glob(f'*{ext.upper()}'))
            
        if not image_paths:
            print(f"No images found in {input_dir}")
            return []
            
        print(f"Found {len(image_paths)} images to process")
        
        # Process images
        results = self.predict_batch([str(p) for p in image_paths])
        
        # Save summary
        self._save_summary(results)
        
        return results
        
    def _save_results(
        self,
        results: Dict[str, Any],
        sample_id: str
    ) -> None:
        """Save prediction results to disk.
        
        Args:
            results: Prediction results dictionary.
            sample_id: Sample identifier.
        """
        # Create sample directory
        sample_dir = self.output_dir / sample_id
        sample_dir.mkdir(exist_ok=True)
        
        # Save curves as JSON
        json_data = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            elif not key.startswith('_'):
                json_data[key] = value
                
        with open(sample_dir / 'predictions.json', 'w') as f:
            json.dump(json_data, f, indent=2)
            
        # Save 3D mesh if available
        if 'spine_3d' in results and self.config.save_3d_mesh:
            save_3d_spine_mesh(
                results['spine_3d'],
                str(sample_dir / 'spine_3d.ply')
            )
            
    def _visualize_results(
        self,
        results: Dict[str, Any],
        sample_id: str
    ) -> None:
        """Generate and save visualizations.
        
        Args:
            results: Prediction results.
            sample_id: Sample identifier.
        """
        vis_dir = self.output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        # Create multi-panel visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot curves
        curve_types = [
            ('coronal_centerline', 'Coronal Centerline'),
            ('coronal_lateral', 'Coronal Lateral'),
            ('sagittal_centerline', 'Sagittal Centerline'),
            ('sagittal_lateral', 'Sagittal Lateral')
        ]
        
        for idx, (key, title) in enumerate(curve_types):
            if idx < 4 and key in results:
                ax = axes.flat[idx]
                curve = results[key]
                
                ax.plot(curve[:, 0], curve[:, 1], 'b-', linewidth=2)
                ax.set_title(title)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')
                
        # Plot 3D spine if available
        if 'spine_3d' in results:
            ax = axes.flat[4]
            spine_3d = results['spine_3d']
            
            # Project to 2D for visualization
            ax.scatter(spine_3d[:, 0], spine_3d[:, 2], c='red', s=20)
            ax.plot(spine_3d[:, 0], spine_3d[:, 2], 'r-', alpha=0.5)
            ax.set_title('3D Spine (X-Z projection)')
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.grid(True, alpha=0.3)
            
        # Add metrics
        ax = axes.flat[5]
        ax.axis('off')
        metrics_text = f"Sample: {sample_id}\n"
        if 'angle' in results:
            metrics_text += f"Cobb Angle: {results['angle']:.1f}°\n"
        if 'timestamp' in results:
            metrics_text += f"Timestamp: {results['timestamp']}"
            
        ax.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center')
        
        plt.suptitle(f'Spine Shape Estimation - {sample_id}')
        plt.tight_layout()
        
        # Save figure
        fig.savefig(vis_dir / f'{sample_id}_visualization.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
    def _save_summary(self, results: List[Dict[str, Any]]) -> None:
        """Save summary of all predictions.
        
        Args:
            results: List of prediction results.
        """
        # Create summary dataframe
        summary_data = []
        
        for result in results:
            row = {
                'image_path': result['image_path'],
                'timestamp': result.get('timestamp', ''),
                'angle': result.get('angle', np.nan)
            }
            summary_data.append(row)
            
        df = pd.DataFrame(summary_data)
        
        # Save as CSV
        df.to_csv(self.output_dir / 'summary.csv', index=False)
        
        # Print statistics
        print("\n" + "="*50)
        print("INFERENCE SUMMARY")
        print("="*50)
        print(f"Total images processed: {len(results)}")
        
        if 'angle' in df.columns:
            print(f"Mean Modified Ferguson Angle: {df['angle'].mean():.2f}°")
            print(f"Std Modified Ferguson Angle: {df['angle'].std():.2f}°")
            print(f"Min Modified Ferguson Angle: {df['angle'].min():.2f}°")
            print(f"Max Modified Ferguson Angle: {df['angle'].max():.2f}°")
            
        print(f"Results saved to: {self.output_dir}")
        print("="*50)


def main() -> None:
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description='3D Spine Shape Estimation Inference'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/inference_config.yaml',
        help='Path to inference configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to model checkpoint (overrides config)'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Input image or directory (overrides config)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory (overrides config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        help='Device for inference (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for inference (overrides config)'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Disable visualization'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config) if Path(args.config).exists() else InferenceConfig()
    
    # Override config with command line arguments
    if args.checkpoint:
        config.model_checkpoint = args.checkpoint
    if args.input:
        config.input_dir = args.input
    if args.output:
        config.output_dir = args.output
    if args.device:
        config.device = args.device
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.no_viz:
        config.save_visualizations = False
        
    # Set device
    device = torch.device(config.device if torch.cuda.is_available() or config.device == 'cpu' else 'cpu')
    
    # Create estimator
    estimator = SpineEstimator(config, device)
    
    # Determine input type
    input_path = Path(config.input_dir)
    
    if input_path.is_file():
        # Single image inference
        print(f"Processing single image: {input_path}")
        results = estimator.predict_single(str(input_path))
        
    elif input_path.is_dir():
        # Directory inference
        print(f"Processing directory: {input_path}")
        results = estimator.predict_directory(str(input_path))
        
    else:
        print(f"Error: Input path does not exist: {input_path}")
        return



if __name__ == "__main__":
    main()