"""
Dataset module for DXA to 3D Spine Shape Estimation.

This module provides dataset classes and data augmentation pipelines
for loading and preprocessing DXA images and spine annotations.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import h5py
import nibabel as nib
from scipy.interpolate import interp1d


class DXADataset(Dataset):
    """Dataset for loading DXA images and spine annotations.
    
    This dataset loads DXA images along with their corresponding
    spine curve annotations for training the estimation model.
    
    Args:
        data_dir (str): Root directory containing the dataset.
        split (str): Dataset split ('train', 'val', or 'test').
        transform (Optional[Callable]): Image transformations to apply.
        target_size (Tuple[int, int]): Target image size. Default: (512, 512).
        num_points (int): Number of points per curve. Default: 100.
        load_3d (bool): Whether to load 3D annotations. Default: False.
        cache_data (bool): Whether to cache data in memory. Default: False.
        
    Attributes:
        data_dir (Path): Path to data directory.
        split (str): Dataset split name.
        transform (Optional[Callable]): Transformation pipeline.
        target_size (Tuple[int, int]): Target image dimensions.
        num_points (int): Number of curve points.
        load_3d (bool): Whether to load 3D data.
        samples (List[Dict]): List of dataset samples.
        cache (Dict): Data cache if caching is enabled.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (512, 512),
        num_points: int = 100,
        load_3d: bool = False,
        cache_data: bool = False
    ) -> None:
        """Initialize DXADataset."""
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.num_points = num_points
        self.load_3d = load_3d
        self.cache_data = cache_data
        self.cache: Dict[int, Dict] = {}
        
        # Load dataset metadata
        self.samples = self._load_metadata()
        
        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {data_dir} for split {split}")
            
        print(f"Loaded {len(self.samples)} samples for {split} split")
        
    def _load_metadata(self) -> List[Dict[str, Any]]:
        """Load dataset metadata from annotation files.
        
        Returns:
            List of sample dictionaries.
        """
        samples = []
        
        # Check for different annotation formats
        metadata_file = self.data_dir / f"{self.split}_metadata.json"
        csv_file = self.data_dir / f"{self.split}_annotations.csv"
        
        if metadata_file.exists():
            # Load from JSON metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                samples = metadata['samples']
                
        elif csv_file.exists():
            # Load from CSV annotations
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                sample = {
                    'image_path': row['image_path'],
                    'annotation_path': row['annotation_path'],
                    'patient_id': row.get('patient_id', None),
                    'metadata': row.to_dict()
                }
                samples.append(sample)
                
        else:
            # Scan directory for paired files
            image_dir = self.data_dir / self.split / 'images'
            annotation_dir = self.data_dir / self.split / 'annotations'
            
            if image_dir.exists() and annotation_dir.exists():
                image_files = sorted(image_dir.glob('*.png'))
                
                for img_path in image_files:
                    # Find corresponding annotation
                    ann_name = img_path.stem + '.json'
                    ann_path = annotation_dir / ann_name
                    
                    if ann_path.exists():
                        samples.append({
                            'image_path': str(img_path),
                            'annotation_path': str(ann_path),
                            'patient_id': img_path.stem
                        })
                        
        return samples
        
    def __len__(self) -> int:
        """Get dataset length.
        
        Returns:
            Number of samples in the dataset.
        """
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a dataset sample.
        
        Args:
            idx: Sample index.
            
        Returns:
            Dictionary containing:
                - 'image': DXA image tensor
                - 'coronal_centerline': Coronal centerline points
                - 'coronal_lateral': Coronal lateral curve points
                - 'sagittal_centerline': Sagittal centerline points
                - 'sagittal_lateral': Sagittal lateral curve points
                - 'metadata': Additional sample metadata
        """
        # Check cache
        if self.cache_data and idx in self.cache:
            return self.cache[idx]
            
        # Load sample data
        sample_info = self.samples[idx]
        
        # Load image
        image = self._load_image(sample_info['image_path'])
        
        # Load annotations
        annotations = self._load_annotations(sample_info['annotation_path'])
        
        # Prepare sample
        sample = {
            'image': image,
            **annotations,
            'metadata': sample_info.get('metadata', {})
        }
        
        # Apply transforms
        if self.transform:
            sample = self._apply_transforms(sample)
            
        # Convert to tensors
        sample = self._to_tensor(sample)
        
        # Cache if enabled
        if self.cache_data:
            self.cache[idx] = sample
            
        return sample
        
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess DXA image.
        
        Args:
            image_path: Path to image file.
            
        Returns:
            Preprocessed image array.
        """
        # Load image
        if image_path.endswith('.npy'):
            image = np.load(image_path)
        elif image_path.endswith('.h5'):
            with h5py.File(image_path, 'r') as f:
                image = f['image'][:]
        else:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        # Resize to target size
        if image.shape[:2] != self.target_size:
            image = cv2.resize(
                image,
                self.target_size,
                interpolation=cv2.INTER_LINEAR
            )
            
        # Normalize to [0, 1]
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Add channel dimension if needed
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
            
        return image
        
    def _load_annotations(self, annotation_path: str) -> Dict[str, np.ndarray]:
        """Load spine curve annotations.
        
        Args:
            annotation_path: Path to annotation file.
            
        Returns:
            Dictionary of curve annotations.
        """
        annotations = {}
        
        if annotation_path.endswith('.json'):
            # Load from JSON
            with open(annotation_path, 'r') as f:
                ann_data = json.load(f)
                
            # Process each curve type
            for curve_type in ['coronal_centerline', 'coronal_lateral',
                             'sagittal_centerline', 'sagittal_lateral']:
                if curve_type in ann_data:
                    points = np.array(ann_data[curve_type])
                    # Resample to fixed number of points
                    points = self._resample_curve(points, self.num_points)
                    # Normalize coordinates
                    points = self._normalize_points(points)
                    annotations[curve_type] = points
                    
        elif annotation_path.endswith('.npy'):
            # Load from numpy array
            ann_data = np.load(annotation_path, allow_pickle=True).item()
            for key, value in ann_data.items():
                if isinstance(value, np.ndarray):
                    points = self._resample_curve(value, self.num_points)
                    points = self._normalize_points(points)
                    annotations[key] = points
                    
        # Generate default annotations if missing
        for curve_type in ['coronal_centerline', 'coronal_lateral',
                          'sagittal_centerline', 'sagittal_lateral']:
            if curve_type not in annotations:
                # Create placeholder straight line
                annotations[curve_type] = self._generate_default_curve()
                
        return annotations
        
    def _resample_curve(
        self,
        points: np.ndarray,
        target_points: int
    ) -> np.ndarray:
        """Resample curve to fixed number of points.
        
        Args:
            points: Original curve points (N, 2).
            target_points: Target number of points.
            
        Returns:
            Resampled curve points.
        """
        if len(points) == target_points:
            return points
            
        # Parameterize curve
        t = np.linspace(0, 1, len(points))
        t_new = np.linspace(0, 1, target_points)
        
        # Interpolate
        interp_x = interp1d(t, points[:, 0], kind='cubic', fill_value='extrapolate')
        interp_y = interp1d(t, points[:, 1], kind='cubic', fill_value='extrapolate')
        
        resampled = np.column_stack([
            interp_x(t_new),
            interp_y(t_new)
        ])
        
        return resampled
        
    def _normalize_points(
        self,
        points: np.ndarray,
        scale: float = 1.0
    ) -> np.ndarray:
        """Normalize point coordinates.
        
        Args:
            points: Input points (N, 2).
            scale: Scale factor.
            
        Returns:
            Normalized points.
        """
        # Normalize to [-1, 1] range
        points_norm = points.copy()
        
        # Center points
        center = points_norm.mean(axis=0)
        points_norm = points_norm - center
        
        # Scale to unit variance
        std = points_norm.std()
        if std > 0:
            points_norm = points_norm / std * scale
            
        return points_norm
        
    def _generate_default_curve(self) -> np.ndarray:
        """Generate default straight curve.
        
        Returns:
            Default curve points.
        """
        # Generate straight vertical line
        y = np.linspace(-1, 1, self.num_points)
        x = np.zeros(self.num_points)
        return np.column_stack([x, y])
        
    def _apply_transforms(
        self,
        sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply data augmentation transforms.
        
        Args:
            sample: Input sample dictionary.
            
        Returns:
            Transformed sample.
        """
        if self.transform is None:
            return sample
            
        # Apply image transforms
        image = sample['image']
        
        if isinstance(self.transform, A.Compose):
            # Apply albumentations transform
            augmented = self.transform(image=image)
            sample['image'] = augmented['image']
        else:
            # Apply custom transform
            sample['image'] = self.transform(image)
            
        return sample
        
    def _to_tensor(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Convert sample to PyTorch tensors.
        
        Args:
            sample: Input sample dictionary.
            
        Returns:
            Sample with tensor values.
        """
        tensor_sample = {}
        
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                if key == 'image':
                    # Convert image to tensor (C, H, W)
                    if len(value.shape) == 3:
                        value = value.transpose(2, 0, 1)
                    elif len(value.shape) == 2:
                        value = np.expand_dims(value, axis=0)
                        
                tensor_sample[key] = torch.from_numpy(value).float()
            else:
                tensor_sample[key] = value
                
        return tensor_sample


def get_data_transforms(
    training: bool = True,
    target_size: Tuple[int, int] = (512, 512)
) -> A.Compose:
    """Get data augmentation transforms.
    
    Args:
        training: Whether to apply training augmentations.
        target_size: Target image size.
        
    Returns:
        Albumentations composition of transforms.
    """
    if training:
        transforms = A.Compose([
            # Spatial transforms
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.5
            ),
            
            # Intensity transforms
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.RandomGamma(
                gamma_limit=(80, 120),
                p=0.3
            ),
            
            # Noise
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            
            # Elastic deformation (important for medical images)
            A.ElasticTransform(
                alpha=120,
                sigma=6,
                alpha_affine=3.6,
                p=0.3
            ),
            
            # Ensure correct size
            A.Resize(height=target_size[0], width=target_size[1]),
            
            # Normalize
            A.Normalize(mean=0.5, std=0.5),
            
            # Convert to tensor
            ToTensorV2()
        ])
    else:
        transforms = A.Compose([
            # Just resize and normalize for validation/test
            A.Resize(height=target_size[0], width=target_size[1]),
            A.Normalize(mean=0.5, std=0.5),
            ToTensorV2()
        ])
        
    return transforms


class DXADataModule:
    """Data module for managing DXA datasets and dataloaders.
    
    Args:
        data_dir (str): Root data directory.
        batch_size (int): Batch size for dataloaders.
        num_workers (int): Number of dataloader workers.
        target_size (Tuple[int, int]): Target image size.
        num_points (int): Number of curve points.
        cache_data (bool): Whether to cache data.
        
    Attributes:
        data_dir (Path): Data directory path.
        batch_size (int): Batch size.
        num_workers (int): Number of workers.
        train_dataset (Optional[DXADataset]): Training dataset.
        val_dataset (Optional[DXADataset]): Validation dataset.
        test_dataset (Optional[DXADataset]): Test dataset.
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        target_size: Tuple[int, int] = (512, 512),
        num_points: int = 100,
        cache_data: bool = False
    ) -> None:
        """Initialize DXADataModule."""
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = target_size
        self.num_points = num_points
        self.cache_data = cache_data
        
        self.train_dataset: Optional[DXADataset] = None
        self.val_dataset: Optional[DXADataset] = None
        self.test_dataset: Optional[DXADataset] = None
        
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for the specified stage.
        
        Args:
            stage: Stage name ('fit', 'validate', 'test', or None for all).
        """
        if stage in ('fit', None):
            self.train_dataset = DXADataset(
                data_dir=self.data_dir,
                split='train',
                transform=get_data_transforms(training=True, target_size=self.target_size),
                target_size=self.target_size,
                num_points=self.num_points,
                cache_data=self.cache_data
            )
            
            self.val_dataset = DXADataset(
                data_dir=self.data_dir,
                split='val',
                transform=get_data_transforms(training=False, target_size=self.target_size),
                target_size=self.target_size,
                num_points=self.num_points,
                cache_data=self.cache_data
            )
            
        if stage in ('test', None):
            self.test_dataset = DXADataset(
                data_dir=self.data_dir,
                split='test',
                transform=get_data_transforms(training=False, target_size=self.target_size),
                target_size=self.target_size,
                num_points=self.num_points,
                cache_data=False
            )
            
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader.
        
        Returns:
            Training DataLoader.
        """
        if self.train_dataset is None:
            self.setup('fit')
            
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader.
        
        Returns:
            Validation DataLoader.
        """
        if self.val_dataset is None:
            self.setup('fit')
            
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    def test_dataloader(self) -> DataLoader:
        """Get test dataloader.
        
        Returns:
            Test DataLoader.
        """
        if self.test_dataset is None:
            self.setup('test')
            
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )