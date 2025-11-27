"""Sampling and visualization utilities for VAE models."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.utils import make_grid, save_image


class VAESampler:
    """
    Sampling utilities for VAE models.
    
    This class provides various sampling methods:
    - Random sampling from prior
    - Conditional sampling
    - Interpolation sampling
    - Latent space traversal
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize VAE sampler.
        
        Args:
            model: Trained VAE model
            device: Device to run sampling on
            seed: Random seed for reproducibility
        """
        self.model = model.to(device)
        self.device = device
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
    
    def sample_random(
        self,
        num_samples: int,
        batch_size: int = 100,
    ) -> Tensor:
        """
        Sample random images from the prior distribution.
        
        Args:
            num_samples: Number of samples to generate
            batch_size: Batch size for generation
            
        Returns:
            Generated samples
        """
        self.model.eval()
        
        samples = []
        with torch.no_grad():
            for _ in range(0, num_samples, batch_size):
                current_batch_size = min(batch_size, num_samples - len(samples))
                batch_samples = self.model.sample(current_batch_size, self.device)
                samples.append(batch_samples)
        
        return torch.cat(samples, dim=0)
    
    def sample_interpolation(
        self,
        x1: Tensor,
        x2: Tensor,
        num_steps: int = 10,
    ) -> Tensor:
        """
        Sample interpolated images between two inputs.
        
        Args:
            x1: First input
            x2: Second input
            num_steps: Number of interpolation steps
            
        Returns:
            Interpolated samples
        """
        self.model.eval()
        
        with torch.no_grad():
            return self.model.interpolate(x1, x2, num_steps)
    
    def sample_latent_traversal(
        self,
        base_sample: Tensor,
        latent_dim: int,
        num_steps: int = 10,
        step_size: float = 2.0,
    ) -> Tensor:
        """
        Sample images by traversing latent space.
        
        Args:
            base_sample: Base sample to traverse from
            latent_dim: Dimension to traverse
            num_steps: Number of steps in traversal
            step_size: Size of each step
            
        Returns:
            Traversed samples
        """
        self.model.eval()
        
        with torch.no_grad():
            # Encode base sample
            mu, _ = self.model.encode(base_sample)
            
            # Create traversal
            traversal_samples = []
            for i in range(num_steps):
                # Create modified latent vector
                z_modified = mu.clone()
                z_modified[0, latent_dim] += (i - num_steps // 2) * step_size
                
                # Decode
                recon = self.model.decode(z_modified)
                traversal_samples.append(recon)
        
        return torch.stack(traversal_samples)
    
    def sample_conditional(
        self,
        condition: Tensor,
        num_samples: int = 1,
    ) -> Tensor:
        """
        Sample images conditioned on a latent vector.
        
        Args:
            condition: Conditioning latent vector
            num_samples: Number of samples to generate
            
        Returns:
            Conditioned samples
        """
        self.model.eval()
        
        with torch.no_grad():
            # Repeat condition for multiple samples
            if condition.dim() == 1:
                condition = condition.unsqueeze(0).repeat(num_samples, 1)
            elif condition.size(0) == 1:
                condition = condition.repeat(num_samples, 1)
            
            # Decode
            samples = self.model.decode(condition)
        
        return samples


class VAEVisualizer:
    """
    Visualization utilities for VAE models.
    
    This class provides various visualization methods:
    - Sample grids
    - Reconstruction comparisons
    - Latent space visualizations
    - Interpolation plots
    """
    
    def __init__(
        self,
        sampler: VAESampler,
        save_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Initialize VAE visualizer.
        
        Args:
            sampler: VAE sampler instance
            save_dir: Directory to save visualizations
        """
        self.sampler = sampler
        self.save_dir = Path(save_dir) if save_dir else None
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_samples(
        self,
        num_samples: int = 64,
        grid_size: Tuple[int, int] = (8, 8),
        title: str = "Generated Samples",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize generated samples in a grid.
        
        Args:
            num_samples: Number of samples to generate
            grid_size: Grid dimensions (rows, cols)
            title: Plot title
            save_path: Path to save the plot
        """
        samples = self.sampler.sample_random(num_samples)
        
        # Create grid
        grid = make_grid(samples, nrow=grid_size[1], normalize=True, pad_value=1)
        
        # Plot
        plt.figure(figsize=(12, 12))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.title(title)
        plt.axis("off")
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        elif self.save_dir:
            plt.savefig(self.save_dir / f"{title.lower().replace(' ', '_')}.png", bbox_inches="tight", dpi=150)
        
        plt.show()
    
    def visualize_reconstructions(
        self,
        dataloader,
        num_samples: int = 16,
        grid_size: Tuple[int, int] = (4, 4),
        title: str = "Reconstructions",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize original images and their reconstructions.
        
        Args:
            dataloader: Data loader for original images
            num_samples: Number of samples to visualize
            grid_size: Grid dimensions (rows, cols)
            title: Plot title
            save_path: Path to save the plot
        """
        self.sampler.model.eval()
        
        # Get samples
        original_images = []
        reconstructed_images = []
        
        with torch.no_grad():
            for data, _ in dataloader:
                if len(original_images) >= num_samples:
                    break
                
                data = data.to(self.sampler.device)
                recon_data, _, _ = self.sampler.model(data)
                
                original_images.append(data)
                reconstructed_images.append(recon_data)
        
        original_images = torch.cat(original_images, dim=0)[:num_samples]
        reconstructed_images = torch.cat(reconstructed_images, dim=0)[:num_samples]
        
        # Create comparison grid
        comparison_images = []
        for i in range(num_samples):
            comparison_images.extend([original_images[i], reconstructed_images[i]])
        
        grid = make_grid(comparison_images, nrow=2, normalize=True, pad_value=1)
        
        # Plot
        plt.figure(figsize=(12, 12))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.title(title)
        plt.axis("off")
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        elif self.save_dir:
            plt.savefig(self.save_dir / f"{title.lower().replace(' ', '_')}.png", bbox_inches="tight", dpi=150)
        
        plt.show()
    
    def visualize_interpolation(
        self,
        x1: Tensor,
        x2: Tensor,
        num_steps: int = 10,
        title: str = "Latent Space Interpolation",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize interpolation between two images.
        
        Args:
            x1: First image
            x2: Second image
            num_steps: Number of interpolation steps
            title: Plot title
            save_path: Path to save the plot
        """
        interpolated = self.sampler.sample_interpolation(x1, x2, num_steps)
        
        # Create grid
        grid = make_grid(interpolated, nrow=num_steps, normalize=True, pad_value=1)
        
        # Plot
        plt.figure(figsize=(15, 3))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.title(title)
        plt.axis("off")
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        elif self.save_dir:
            plt.savefig(self.save_dir / f"{title.lower().replace(' ', '_')}.png", bbox_inches="tight", dpi=150)
        
        plt.show()
    
    def visualize_latent_traversal(
        self,
        base_sample: Tensor,
        latent_dim: int,
        num_steps: int = 10,
        step_size: float = 2.0,
        title: str = "Latent Space Traversal",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize latent space traversal.
        
        Args:
            base_sample: Base sample to traverse from
            latent_dim: Dimension to traverse
            num_steps: Number of steps in traversal
            step_size: Size of each step
            title: Plot title
            save_path: Path to save the plot
        """
        traversed = self.sampler.sample_latent_traversal(
            base_sample, latent_dim, num_steps, step_size
        )
        
        # Create grid
        grid = make_grid(traversed, nrow=num_steps, normalize=True, pad_value=1)
        
        # Plot
        plt.figure(figsize=(15, 3))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.title(f"{title} - Dimension {latent_dim}")
        plt.axis("off")
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        elif self.save_dir:
            plt.savefig(self.save_dir / f"{title.lower().replace(' ', '_')}_dim_{latent_dim}.png", bbox_inches="tight", dpi=150)
        
        plt.show()
    
    def visualize_latent_space(
        self,
        dataloader,
        num_samples: int = 1000,
        method: str = "tsne",
        title: str = "Latent Space Visualization",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize latent space using dimensionality reduction.
        
        Args:
            dataloader: Data loader for samples
            num_samples: Number of samples to visualize
            method: Dimensionality reduction method ("tsne", "pca", "umap")
            title: Plot title
            save_path: Path to save the plot
        """
        self.sampler.model.eval()
        
        # Collect latent representations
        latent_vectors = []
        labels = []
        
        with torch.no_grad():
            for data, target in dataloader:
                if len(latent_vectors) >= num_samples:
                    break
                
                data = data.to(self.sampler.device)
                mu, _ = self.sampler.model.encode(data)
                
                latent_vectors.append(mu.cpu())
                labels.extend(target.cpu().numpy())
        
        latent_vectors = torch.cat(latent_vectors, dim=0)[:num_samples]
        labels = np.array(labels)[:num_samples]
        
        # Apply dimensionality reduction
        if method == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
        elif method == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
        elif method == "umap":
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
            except ImportError:
                print("UMAP not available, using PCA instead")
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Reduce dimensions
        latent_2d = reducer.fit_transform(latent_vectors.numpy())
        
        # Plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap="tab10", alpha=0.6)
        plt.colorbar(scatter)
        plt.title(f"{title} - {method.upper()}")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        elif self.save_dir:
            plt.savefig(self.save_dir / f"{title.lower().replace(' ', '_')}_{method}.png", bbox_inches="tight", dpi=150)
        
        plt.show()
    
    def save_samples(
        self,
        samples: Tensor,
        filename: str,
        nrow: int = 8,
    ) -> None:
        """
        Save samples to file.
        
        Args:
            samples: Sample tensor
            filename: Filename to save
            nrow: Number of images per row
        """
        if self.save_dir:
            save_path = self.save_dir / filename
        else:
            save_path = filename
        
        save_image(samples, save_path, nrow=nrow, normalize=True, pad_value=1)
    
    def create_comprehensive_visualization(
        self,
        dataloader,
        num_samples: int = 64,
        num_reconstructions: int = 16,
        num_interpolations: int = 5,
        latent_dims_to_traverse: List[int] = [0, 1, 2, 3],
    ) -> None:
        """
        Create comprehensive visualization of VAE capabilities.
        
        Args:
            dataloader: Data loader for evaluation
            num_samples: Number of random samples
            num_reconstructions: Number of reconstructions
            num_interpolations: Number of interpolation pairs
            latent_dims_to_traverse: Latent dimensions to traverse
        """
        print("Creating comprehensive VAE visualization...")
        
        # Random samples
        print("Generating random samples...")
        self.visualize_samples(num_samples, title="Random Samples")
        
        # Reconstructions
        print("Visualizing reconstructions...")
        self.visualize_reconstructions(dataloader, num_reconstructions, title="Reconstructions")
        
        # Interpolations
        print("Creating interpolations...")
        self.sampler.model.eval()
        with torch.no_grad():
            for i in range(num_interpolations):
                data, _ = next(iter(dataloader))
                if data.size(0) < 2:
                    continue
                
                x1, x2 = data[0:1], data[1:2]
                self.visualize_interpolation(
                    x1, x2, title=f"Interpolation {i+1}"
                )
        
        # Latent traversals
        print("Creating latent traversals...")
        with torch.no_grad():
            data, _ = next(iter(dataloader))
            base_sample = data[0:1]
            
            for dim in latent_dims_to_traverse:
                self.visualize_latent_traversal(
                    base_sample, dim, title=f"Latent Traversal"
                )
        
        # Latent space visualization
        print("Visualizing latent space...")
        self.visualize_latent_space(dataloader, title="Latent Space")
        
        print("Comprehensive visualization complete!")
