"""Training utilities and PyTorch Lightning module for VAE."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from .data import get_device
from .models import BetaVAE, ConvVAE, vae_loss, kl_annealing_schedule


class VAELightningModule(LightningModule):
    """
    PyTorch Lightning module for VAE training.
    
    This module handles:
    - Training and validation loops
    - Loss computation with KL annealing
    - Logging and metrics
    - Checkpointing
    - Sampling and interpolation
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        data_config: Dict[str, Any],
    ) -> None:
        """
        Initialize VAE Lightning module.
        
        Args:
            model_config: Model configuration
            training_config: Training configuration
            data_config: Data configuration
        """
        super().__init__()
        
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        
        # Initialize model
        model_type = model_config.get("type", "beta_vae")
        if model_type == "beta_vae":
            self.model = BetaVAE(**model_config["params"])
        elif model_type == "conv_vae":
            self.model = ConvVAE(**model_config["params"])
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Training parameters
        self.learning_rate = training_config.get("learning_rate", 1e-3)
        self.beta_start = training_config.get("beta_start", 0.0)
        self.beta_end = training_config.get("beta_end", 1.0)
        self.beta_schedule = training_config.get("beta_schedule", "linear")
        self.warmup_epochs = training_config.get("warmup_epochs", 10)
        
        # Metrics
        self.train_losses = []
        self.val_losses = []
        
        # Save hyperparameters
        self.save_hyperparameters()
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass through the model."""
        return self.model(x)
    
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Training step."""
        x, _ = batch
        
        # Forward pass
        recon_x, mu, logvar = self.forward(x)
        
        # Compute current beta for KL annealing
        current_epoch = self.current_epoch
        total_epochs = self.training_config.get("max_epochs", 100)
        beta = kl_annealing_schedule(
            current_epoch,
            total_epochs,
            self.beta_schedule,
            self.beta_start,
            self.beta_end,
        )
        
        # Compute loss
        losses = vae_loss(recon_x, x, mu, logvar, beta=beta)
        
        # Log metrics
        self.log("train/total_loss", losses["total_loss"], on_step=True, on_epoch=True)
        self.log("train/recon_loss", losses["recon_loss"], on_step=True, on_epoch=True)
        self.log("train/kl_loss", losses["kl_loss"], on_step=True, on_epoch=True)
        self.log("train/beta", beta, on_step=True, on_epoch=True)
        
        return losses["total_loss"]
    
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        """Validation step."""
        x, _ = batch
        
        # Forward pass
        recon_x, mu, logvar = self.forward(x)
        
        # Compute loss with final beta
        losses = vae_loss(recon_x, x, mu, logvar, beta=self.beta_end)
        
        # Compute additional metrics
        mse_loss = F.mse_loss(recon_x, x)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse_loss))
        
        # Log metrics
        self.log("val/total_loss", losses["total_loss"], on_step=False, on_epoch=True)
        self.log("val/recon_loss", losses["recon_loss"], on_step=False, on_epoch=True)
        self.log("val/kl_loss", losses["kl_loss"], on_step=False, on_epoch=True)
        self.log("val/mse_loss", mse_loss, on_step=False, on_epoch=True)
        self.log("val/psnr", psnr, on_step=False, on_epoch=True)
        
        return {
            "val_loss": losses["total_loss"],
            "val_recon_loss": losses["recon_loss"],
            "val_kl_loss": losses["kl_loss"],
            "val_mse_loss": mse_loss,
            "val_psnr": psnr,
        }
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        # Generate samples for visualization
        if self.current_epoch % 5 == 0:  # Every 5 epochs
            self._log_samples()
    
    def _log_samples(self, num_samples: int = 16) -> None:
        """Generate and log sample images."""
        device = next(self.parameters()).device
        
        # Generate samples from prior
        with torch.no_grad():
            samples = self.model.sample(num_samples, device)
        
        # Log samples (this would need to be implemented based on your logging backend)
        # For now, we'll just store them
        self.logger.log_image("generated_samples", samples, self.current_epoch)
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and schedulers."""
        optimizer = Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.training_config.get("weight_decay", 1e-5),
        )
        
        scheduler_config = self.training_config.get("scheduler", {})
        scheduler_type = scheduler_config.get("type", "cosine")
        
        if scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.training_config.get("max_epochs", 100),
                eta_min=scheduler_config.get("eta_min", 1e-6),
            )
        elif scheduler_type == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=scheduler_config.get("factor", 0.5),
                patience=scheduler_config.get("patience", 10),
                min_lr=scheduler_config.get("min_lr", 1e-6),
            )
        else:
            scheduler = None
        
        if scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/total_loss",
                },
            }
        else:
            return {"optimizer": optimizer}
    
    def sample(self, num_samples: int) -> Tensor:
        """Generate samples from the model."""
        device = next(self.parameters()).device
        return self.model.sample(num_samples, device)
    
    def interpolate(self, x1: Tensor, x2: Tensor, num_steps: int = 10) -> Tensor:
        """Interpolate between two inputs."""
        return self.model.interpolate(x1, x2, num_steps)


class VAETrainer:
    """
    Custom trainer for VAE with additional utilities.
    
    This trainer provides:
    - Custom training loops
    - KL annealing
    - Metrics computation
    - Sample generation
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Dict[str, Any],
    ) -> None:
        """
        Initialize VAE trainer.
        
        Args:
            model: VAE model
            device: Device to train on
            config: Training configuration
        """
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Initialize optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.get("learning_rate", 1e-3),
            weight_decay=config.get("weight_decay", 1e-5),
        )
        
        # Initialize scheduler
        scheduler_config = config.get("scheduler", {})
        if scheduler_config.get("type") == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.get("max_epochs", 100),
                eta_min=scheduler_config.get("eta_min", 1e-6),
            )
        else:
            self.scheduler = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(self.device)
            
            # Forward pass
            recon_data, mu, logvar = self.model(data)
            
            # Compute current beta for KL annealing
            beta = kl_annealing_schedule(
                self.current_epoch,
                self.config.get("max_epochs", 100),
                self.config.get("beta_schedule", "linear"),
                self.config.get("beta_start", 0.0),
                self.config.get("beta_end", 1.0),
            )
            
            # Compute loss
            losses = vae_loss(recon_data, data, mu, logvar, beta=beta)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses["total_loss"].backward()
            
            # Gradient clipping
            if self.config.get("grad_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["grad_clip"],
                )
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += losses["total_loss"].item()
            total_recon_loss += losses["recon_loss"].item()
            total_kl_loss += losses["kl_loss"].item()
        
        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Return average losses
        num_batches = len(train_loader)
        return {
            "total_loss": total_loss / num_batches,
            "recon_loss": total_recon_loss / num_batches,
            "kl_loss": total_kl_loss / num_batches,
            "beta": beta,
        }
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(self.device)
                
                # Forward pass
                recon_data, mu, logvar = self.model(data)
                
                # Compute loss with final beta
                losses = vae_loss(
                    recon_data,
                    data,
                    mu,
                    logvar,
                    beta=self.config.get("beta_end", 1.0),
                )
                
                # Accumulate losses
                total_loss += losses["total_loss"].item()
                total_recon_loss += losses["recon_loss"].item()
                total_kl_loss += losses["kl_loss"].item()
        
        # Return average losses
        num_batches = len(val_loader)
        return {
            "total_loss": total_loss / num_batches,
            "recon_loss": total_recon_loss / num_batches,
            "kl_loss": total_kl_loss / num_batches,
        }
    
    def save_checkpoint(self, path: str, epoch: int, val_loss: float) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.config,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["val_loss"]
