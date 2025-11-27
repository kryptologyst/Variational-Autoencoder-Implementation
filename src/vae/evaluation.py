"""Evaluation metrics and utilities for VAE models."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

try:
    from clean_fid import fid
    CLEAN_FID_AVAILABLE = True
except ImportError:
    CLEAN_FID_AVAILABLE = False


class ReconstructionMetrics(Metric):
    """Compute reconstruction quality metrics."""
    
    def __init__(self, **kwargs: Any) -> None:
        """Initialize reconstruction metrics."""
        super().__init__(**kwargs)
        
        self.add_state("mse_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("mae_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metrics with new predictions and targets."""
        # Flatten tensors
        preds = preds.view(preds.size(0), -1)
        target = target.view(target.size(0), -1)
        
        # Compute metrics
        mse = F.mse_loss(preds, target, reduction="sum")
        mae = F.l1_loss(preds, target, reduction="sum")
        
        self.mse_sum += mse
        self.mae_sum += mae
        self.total += preds.size(0)
    
    def compute(self) -> Dict[str, Tensor]:
        """Compute final metrics."""
        mse = self.mse_sum / self.total
        mae = self.mae_sum / self.total
        
        # Compute PSNR
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        
        return {
            "mse": mse,
            "mae": mae,
            "psnr": psnr,
        }


class LatentSpaceMetrics(Metric):
    """Compute latent space quality metrics."""
    
    def __init__(self, **kwargs: Any) -> None:
        """Initialize latent space metrics."""
        super().__init__(**kwargs)
        
        self.add_state("kl_divergence_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, mu: Tensor, logvar: Tensor) -> None:
        """Update metrics with latent distribution parameters."""
        # Compute KL divergence
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_div_sum = torch.sum(kl_div)
        
        self.kl_divergence_sum += kl_div_sum
        self.total += mu.size(0)
    
    def compute(self) -> Dict[str, Tensor]:
        """Compute final metrics."""
        kl_divergence = self.kl_divergence_sum / self.total
        
        return {
            "kl_divergence": kl_divergence,
        }


class VAEEvaluator:
    """
    Comprehensive evaluator for VAE models.
    
    This evaluator provides:
    - Reconstruction quality metrics
    - Latent space analysis
    - Sample quality assessment
    - Interpolation quality
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        use_fid: bool = True,
    ) -> None:
        """
        Initialize VAE evaluator.
        
        Args:
            model: Trained VAE model
            device: Device to run evaluation on
            use_fid: Whether to compute FID (requires clean-fid)
        """
        self.model = model.to(device)
        self.device = device
        self.use_fid = use_fid and CLEAN_FID_AVAILABLE
        
        # Initialize metrics
        self.metrics = MetricCollection({
            "reconstruction": ReconstructionMetrics(),
            "latent_space": LatentSpaceMetrics(),
            "ssim": StructuralSimilarityIndexMeasure(),
            "psnr": PeakSignalNoiseRatio(),
        })
        
        if self.use_fid:
            self.lpip = LearnedPerceptualImagePatchSimilarity(net_type="alex")
        else:
            self.lpip = None
    
    def evaluate_reconstruction(
        self,
        dataloader,
        num_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate reconstruction quality.
        
        Args:
            dataloader: Data loader for evaluation
            num_batches: Number of batches to evaluate (None for all)
            
        Returns:
            Dictionary of reconstruction metrics
        """
        self.model.eval()
        self.metrics.reset()
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(dataloader):
                if num_batches is not None and batch_idx >= num_batches:
                    break
                
                data = data.to(self.device)
                
                # Forward pass
                recon_data, mu, logvar = self.model(data)
                
                # Update metrics
                self.metrics["reconstruction"](recon_data, data)
                self.metrics["latent_space"](mu, logvar)
                self.metrics["ssim"](recon_data, data)
                self.metrics["psnr"](recon_data, data)
        
        # Compute final metrics
        results = self.metrics.compute()
        
        # Convert to float for JSON serialization
        return {k: v.item() if isinstance(v, Tensor) else v for k, v in results.items()}
    
    def evaluate_sample_quality(
        self,
        num_samples: int = 1000,
        batch_size: int = 100,
    ) -> Dict[str, float]:
        """
        Evaluate quality of generated samples.
        
        Args:
            num_samples: Number of samples to generate
            batch_size: Batch size for generation
            
        Returns:
            Dictionary of sample quality metrics
        """
        self.model.eval()
        
        generated_samples = []
        
        with torch.no_grad():
            for _ in range(0, num_samples, batch_size):
                current_batch_size = min(batch_size, num_samples - len(generated_samples))
                samples = self.model.sample(current_batch_size, self.device)
                generated_samples.append(samples)
        
        generated_samples = torch.cat(generated_samples, dim=0)
        
        # Compute statistics
        mean_sample = torch.mean(generated_samples, dim=0)
        std_sample = torch.std(generated_samples, dim=0)
        
        # Compute diversity (average pairwise distance)
        diversity = self._compute_diversity(generated_samples)
        
        results = {
            "sample_mean": mean_sample.mean().item(),
            "sample_std": std_sample.mean().item(),
            "diversity": diversity,
        }
        
        return results
    
    def evaluate_interpolation_quality(
        self,
        dataloader,
        num_pairs: int = 10,
        num_steps: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate interpolation quality in latent space.
        
        Args:
            dataloader: Data loader for evaluation
            num_pairs: Number of pairs to interpolate
            num_steps: Number of interpolation steps
            
        Returns:
            Dictionary of interpolation metrics
        """
        self.model.eval()
        
        interpolation_smoothness = []
        
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(dataloader):
                if batch_idx >= num_pairs:
                    break
                
                # Get two random samples from batch
                if data.size(0) < 2:
                    continue
                
                idx1, idx2 = torch.randperm(data.size(0))[:2]
                x1, x2 = data[idx1:idx1+1], data[idx2:idx2+1]
                
                # Interpolate
                interpolated = self.model.interpolate(x1, x2, num_steps)
                
                # Compute smoothness (L2 distance between consecutive steps)
                smoothness = torch.mean(
                    torch.norm(interpolated[1:] - interpolated[:-1], dim=1)
                ).item()
                
                interpolation_smoothness.append(smoothness)
        
        return {
            "interpolation_smoothness": np.mean(interpolation_smoothness),
            "interpolation_std": np.std(interpolation_smoothness),
        }
    
    def compute_fid(
        self,
        real_dataloader,
        num_samples: int = 1000,
        batch_size: int = 100,
    ) -> Optional[float]:
        """
        Compute Fréchet Inception Distance (FID).
        
        Args:
            real_dataloader: Data loader for real images
            num_samples: Number of samples to generate
            batch_size: Batch size for generation
            
        Returns:
            FID score or None if not available
        """
        if not self.use_fid:
            return None
        
        self.model.eval()
        
        # Generate samples
        generated_samples = []
        with torch.no_grad():
            for _ in range(0, num_samples, batch_size):
                current_batch_size = min(batch_size, num_samples - len(generated_samples))
                samples = self.model.sample(current_batch_size, self.device)
                generated_samples.append(samples)
        
        generated_samples = torch.cat(generated_samples, dim=0)
        
        # Convert to numpy and scale to [0, 255]
        generated_samples = (generated_samples * 255).cpu().numpy().astype(np.uint8)
        
        # Get real samples
        real_samples = []
        for data, _ in real_dataloader:
            real_samples.append((data * 255).cpu().numpy().astype(np.uint8))
            if len(real_samples) * data.size(0) >= num_samples:
                break
        
        real_samples = np.concatenate(real_samples, axis=0)[:num_samples]
        
        # Compute FID
        try:
            fid_score = fid.compute_fid(
                real_samples,
                generated_samples,
                mode="clean",
            )
            return fid_score
        except Exception as e:
            print(f"Error computing FID: {e}")
            return None
    
    def _compute_diversity(self, samples: Tensor) -> float:
        """Compute diversity metric (average pairwise distance)."""
        n_samples = samples.size(0)
        if n_samples < 2:
            return 0.0
        
        # Flatten samples
        samples_flat = samples.view(n_samples, -1)
        
        # Compute pairwise distances
        distances = torch.cdist(samples_flat, samples_flat, p=2)
        
        # Get upper triangular part (excluding diagonal)
        mask = torch.triu(torch.ones_like(distances), diagonal=1).bool()
        pairwise_distances = distances[mask]
        
        return torch.mean(pairwise_distances).item()
    
    def latent_traversal(
        self,
        base_sample: Tensor,
        latent_dim: int,
        num_steps: int = 10,
        step_size: float = 2.0,
    ) -> Tensor:
        """
        Perform latent space traversal.
        
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
    
    def comprehensive_evaluation(
        self,
        dataloader,
        num_samples: int = 1000,
        num_batches: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of the VAE model.
        
        Args:
            dataloader: Data loader for evaluation
            num_samples: Number of samples for generation evaluation
            num_batches: Number of batches for reconstruction evaluation
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        results = {}
        
        # Reconstruction evaluation
        print("Evaluating reconstruction quality...")
        results["reconstruction"] = self.evaluate_reconstruction(dataloader, num_batches)
        
        # Sample quality evaluation
        print("Evaluating sample quality...")
        results["sample_quality"] = self.evaluate_sample_quality(num_samples)
        
        # Interpolation evaluation
        print("Evaluating interpolation quality...")
        results["interpolation"] = self.evaluate_interpolation_quality(dataloader)
        
        # FID evaluation
        if self.use_fid:
            print("Computing FID...")
            fid_score = self.compute_fid(dataloader, num_samples)
            if fid_score is not None:
                results["fid"] = fid_score
        
        return results
