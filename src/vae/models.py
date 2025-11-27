"""Core VAE model implementations with modern features."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BetaVAE(nn.Module):
    """
    Beta-VAE implementation with configurable beta parameter and KL annealing.
    
    This implementation includes:
    - Beta parameter for controlling KL divergence weight
    - KL annealing for stable training
    - Better architecture with residual connections
    - Support for different input sizes
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [400, 200],
        latent_dim: int = 20,
        beta: float = 1.0,
        use_batch_norm: bool = True,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize Beta-VAE model.
        
        Args:
            input_dim: Input dimension (e.g., 784 for MNIST)
            hidden_dims: List of hidden layer dimensions
            latent_dim: Latent space dimension
            beta: Beta parameter for KL divergence weight
            use_batch_norm: Whether to use batch normalization
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.use_batch_norm = use_batch_norm
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space projections
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode input to latent space.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (mu, logvar) for latent distribution
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick for sampling from latent space.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latent vector to reconstruction.
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim)
            
        Returns:
            Reconstructed input
        """
        return self.decoder(z)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def sample(self, num_samples: int, device: torch.device) -> Tensor:
        """
        Sample from the prior distribution.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            Generated samples
        """
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)
    
    def interpolate(self, x1: Tensor, x2: Tensor, num_steps: int = 10) -> Tensor:
        """
        Interpolate between two inputs in latent space.
        
        Args:
            x1: First input
            x2: Second input
            num_steps: Number of interpolation steps
            
        Returns:
            Interpolated samples
        """
        mu1, _ = self.encode(x1)
        mu2, _ = self.encode(x2)
        
        # Linear interpolation in latent space
        alphas = torch.linspace(0, 1, num_steps, device=x1.device)
        interpolated_samples = []
        
        for alpha in alphas:
            z_interp = (1 - alpha) * mu1 + alpha * mu2
            recon = self.decode(z_interp)
            interpolated_samples.append(recon)
        
        return torch.stack(interpolated_samples)


class ConvVAE(nn.Module):
    """
    Convolutional VAE for image data.
    
    This implementation uses convolutional layers for better image processing
    and includes modern architectural improvements.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: List[int] = [32, 64, 128],
        latent_dim: int = 20,
        beta: float = 1.0,
        image_size: int = 28,
    ) -> None:
        """
        Initialize ConvVAE model.
        
        Args:
            input_channels: Number of input channels
            hidden_channels: List of hidden channel dimensions
            latent_dim: Latent space dimension
            beta: Beta parameter for KL divergence weight
            image_size: Input image size (assumed square)
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.beta = beta
        self.image_size = image_size
        
        # Calculate the size after convolutions
        self.hidden_channels = hidden_channels
        self.conv_output_size = self._calculate_conv_output_size()
        
        # Encoder
        encoder_layers = []
        in_channels = input_channels
        
        for out_channels in hidden_channels:
            encoder_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ])
            in_channels = out_channels
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space projections
        self.fc_mu = nn.Linear(self.conv_output_size, latent_dim)
        self.fc_logvar = nn.Linear(self.conv_output_size, latent_dim)
        
        # Decoder
        self.fc_decoder = nn.Linear(latent_dim, self.conv_output_size)
        
        decoder_layers = []
        in_channels = hidden_channels[-1]
        
        for out_channels in reversed(hidden_channels[1:] + [input_channels]):
            decoder_layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ])
            in_channels = out_channels
        
        # Remove batch norm and activation from last layer
        decoder_layers = decoder_layers[:-2] + [nn.ConvTranspose2d(in_channels, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1)]
        decoder_layers.append(nn.Sigmoid())
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        self._init_weights()
    
    def _calculate_conv_output_size(self) -> int:
        """Calculate the output size after encoder convolutions."""
        size = self.image_size
        for _ in self.hidden_channels:
            size = (size + 1) // 2  # stride=2
        return self.hidden_channels[-1] * size * size
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode input to latent space."""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: Tensor) -> Tensor:
        """Decode latent vector to reconstruction."""
        h = self.fc_decoder(z)
        h = h.view(h.size(0), self.hidden_channels[-1], self.image_size // (2 ** len(self.hidden_channels)), self.image_size // (2 ** len(self.hidden_channels)))
        return self.decoder(h)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass through ConvVAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def sample(self, num_samples: int, device: torch.device) -> Tensor:
        """Sample from the prior distribution."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)


def vae_loss(
    recon_x: Tensor,
    x: Tensor,
    mu: Tensor,
    logvar: Tensor,
    beta: float = 1.0,
    reduction: str = "mean",
) -> Dict[str, Tensor]:
    """
    Compute VAE loss with configurable beta parameter.
    
    Args:
        recon_x: Reconstructed input
        x: Original input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Beta parameter for KL divergence weight
        reduction: Reduction method for loss computation
        
    Returns:
        Dictionary containing individual loss components
    """
    # Reconstruction loss (BCE)
    if reduction == "mean":
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="mean")
    else:
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    
    if reduction == "mean":
        kl_loss = torch.mean(kl_loss)
    else:
        kl_loss = torch.sum(kl_loss)
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return {
        "total_loss": total_loss,
        "recon_loss": recon_loss,
        "kl_loss": kl_loss,
    }


def kl_annealing_schedule(
    epoch: int,
    total_epochs: int,
    schedule_type: str = "linear",
    start_beta: float = 0.0,
    end_beta: float = 1.0,
) -> float:
    """
    Compute KL annealing schedule.
    
    Args:
        epoch: Current epoch
        total_epochs: Total number of epochs
        schedule_type: Type of schedule ("linear", "cyclical", "sigmoid")
        start_beta: Starting beta value
        end_beta: Ending beta value
        
    Returns:
        Current beta value
    """
    progress = epoch / total_epochs
    
    if schedule_type == "linear":
        return start_beta + (end_beta - start_beta) * progress
    elif schedule_type == "cyclical":
        return end_beta * (1 + math.cos(math.pi * (1 - progress))) / 2
    elif schedule_type == "sigmoid":
        return start_beta + (end_beta - start_beta) / (1 + math.exp(-10 * (progress - 0.5)))
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
