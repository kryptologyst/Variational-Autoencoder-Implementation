"""Data loading and preprocessing utilities for VAE training."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from torchvision.transforms import functional as F


class MNISTDataset(Dataset):
    """Custom MNIST dataset with additional preprocessing options."""
    
    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        download: bool = True,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        normalize: bool = True,
        add_noise: bool = False,
        noise_std: float = 0.1,
    ) -> None:
        """
        Initialize MNIST dataset.
        
        Args:
            root: Root directory for dataset
            train: Whether to use training set
            download: Whether to download dataset
            transform: Transform to apply to images
            target_transform: Transform to apply to targets
            normalize: Whether to normalize to [0, 1]
            add_noise: Whether to add noise for denoising experiments
            noise_std: Standard deviation of noise
        """
        self.dataset = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transform,
            target_transform=target_transform,
        )
        self.normalize = normalize
        self.add_noise = add_noise
        self.noise_std = noise_std
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item from dataset."""
        image, target = self.dataset[idx]
        
        if self.normalize:
            image = image.float() / 255.0
        
        if self.add_noise:
            noise = torch.randn_like(image) * self.noise_std
            image = torch.clamp(image + noise, 0, 1)
        
        return image, target


class FashionMNISTDataset(Dataset):
    """Custom Fashion-MNIST dataset."""
    
    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        download: bool = True,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        normalize: bool = True,
    ) -> None:
        """Initialize Fashion-MNIST dataset."""
        self.dataset = datasets.FashionMNIST(
            root=root,
            train=train,
            download=download,
            transform=transform,
            target_transform=target_transform,
        )
        self.normalize = normalize
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item from dataset."""
        image, target = self.dataset[idx]
        
        if self.normalize:
            image = image.float() / 255.0
        
        return image, target


class CelebADataset(Dataset):
    """Custom CelebA dataset for face generation."""
    
    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        download: bool = True,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        image_size: int = 64,
    ) -> None:
        """
        Initialize CelebA dataset.
        
        Args:
            root: Root directory for dataset
            split: Dataset split ("train", "valid", "test")
            download: Whether to download dataset
            transform: Transform to apply to images
            target_transform: Transform to apply to targets
            image_size: Target image size
        """
        self.dataset = datasets.CelebA(
            root=root,
            split=split,
            download=download,
            transform=transform,
            target_transform=target_transform,
        )
        self.image_size = image_size
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item from dataset."""
        image, target = self.dataset[idx]
        return image, target


def get_transforms(
    dataset_name: str,
    image_size: int = 28,
    augment: bool = False,
) -> transforms.Compose:
    """
    Get appropriate transforms for different datasets.
    
    Args:
        dataset_name: Name of the dataset
        image_size: Target image size
        augment: Whether to include data augmentation
        
    Returns:
        Compose transform
    """
    transform_list = []
    
    if dataset_name.lower() in ["mnist", "fashionmnist"]:
        transform_list.extend([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
    elif dataset_name.lower() == "celeba":
        transform_list.extend([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
    else:
        transform_list.extend([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
    
    if augment:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ])
    
    return transforms.Compose(transform_list)


def create_data_loaders(
    dataset_name: str,
    data_dir: Union[str, Path],
    batch_size: int = 128,
    num_workers: int = 4,
    image_size: int = 28,
    augment: bool = False,
    val_split: float = 0.1,
    test_split: float = 0.1,
    **kwargs: Any,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        dataset_name: Name of the dataset
        data_dir: Directory to store/load data
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        image_size: Target image size
        augment: Whether to use data augmentation
        val_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        **kwargs: Additional arguments for dataset
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    transform = get_transforms(dataset_name, image_size, augment)
    
    # Load dataset
    if dataset_name.lower() == "mnist":
        train_dataset = MNISTDataset(
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
            **kwargs,
        )
        test_dataset = MNISTDataset(
            root=data_dir,
            train=False,
            download=True,
            transform=transform,
            **kwargs,
        )
    elif dataset_name.lower() == "fashionmnist":
        train_dataset = FashionMNISTDataset(
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
            **kwargs,
        )
        test_dataset = FashionMNISTDataset(
            root=data_dir,
            train=False,
            download=True,
            transform=transform,
            **kwargs,
        )
    elif dataset_name.lower() == "celeba":
        train_dataset = CelebADataset(
            root=data_dir,
            split="train",
            download=True,
            transform=transform,
            **kwargs,
        )
        test_dataset = CelebADataset(
            root=data_dir,
            split="test",
            download=True,
            transform=transform,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Split training data for validation
    if val_split > 0:
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    else:
        val_dataset = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


def get_device() -> torch.device:
    """
    Get the best available device.
    
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Make deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
