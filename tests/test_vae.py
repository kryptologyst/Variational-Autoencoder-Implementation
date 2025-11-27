"""Unit tests for VAE models."""

import pytest
import torch
import numpy as np
from torch import Tensor

# Add src to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from vae.models import BetaVAE, ConvVAE, vae_loss, kl_annealing_schedule
from vae.data import get_device, set_seed
from vae.evaluation import VAEEvaluator
from vae.visualization import VAESampler


class TestBetaVAE:
    """Test cases for BetaVAE model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = BetaVAE(
            input_dim=784,
            hidden_dims=[400, 200],
            latent_dim=20,
            beta=1.0,
        )
        
        assert model.input_dim == 784
        assert model.latent_dim == 20
        assert model.beta == 1.0
        assert len(model.encoder) > 0
        assert len(model.decoder) > 0
    
    def test_forward_pass(self):
        """Test forward pass through the model."""
        model = BetaVAE(input_dim=784, latent_dim=20)
        x = torch.randn(32, 784)
        
        recon_x, mu, logvar = model(x)
        
        assert recon_x.shape == x.shape
        assert mu.shape == (32, 20)
        assert logvar.shape == (32, 20)
    
    def test_encode_decode(self):
        """Test encode and decode separately."""
        model = BetaVAE(input_dim=784, latent_dim=20)
        x = torch.randn(32, 784)
        
        # Encode
        mu, logvar = model.encode(x)
        assert mu.shape == (32, 20)
        assert logvar.shape == (32, 20)
        
        # Reparameterize
        z = model.reparameterize(mu, logvar)
        assert z.shape == (32, 20)
        
        # Decode
        recon_x = model.decode(z)
        assert recon_x.shape == x.shape
    
    def test_sampling(self):
        """Test sampling from prior."""
        model = BetaVAE(input_dim=784, latent_dim=20)
        device = get_device()
        
        samples = model.sample(16, device)
        assert samples.shape == (16, 784)
        assert torch.all(samples >= 0) and torch.all(samples <= 1)
    
    def test_interpolation(self):
        """Test interpolation between two inputs."""
        model = BetaVAE(input_dim=784, latent_dim=20)
        x1 = torch.randn(1, 784)
        x2 = torch.randn(1, 784)
        
        interpolated = model.interpolate(x1, x2, num_steps=5)
        assert interpolated.shape == (5, 784)
        assert torch.all(interpolated >= 0) and torch.all(interpolated <= 1)


class TestConvVAE:
    """Test cases for ConvVAE model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = ConvVAE(
            input_channels=1,
            hidden_channels=[32, 64],
            latent_dim=20,
            image_size=28,
        )
        
        assert model.input_channels == 1
        assert model.latent_dim == 20
        assert model.image_size == 28
    
    def test_forward_pass(self):
        """Test forward pass through the model."""
        model = ConvVAE(input_channels=1, latent_dim=20, image_size=28)
        x = torch.randn(32, 1, 28, 28)
        
        recon_x, mu, logvar = model(x)
        
        assert recon_x.shape == x.shape
        assert mu.shape == (32, 20)
        assert logvar.shape == (32, 20)
    
    def test_sampling(self):
        """Test sampling from prior."""
        model = ConvVAE(input_channels=1, latent_dim=20, image_size=28)
        device = get_device()
        
        samples = model.sample(16, device)
        assert samples.shape == (16, 1, 28, 28)
        assert torch.all(samples >= 0) and torch.all(samples <= 1)


class TestVAELoss:
    """Test cases for VAE loss function."""
    
    def test_loss_computation(self):
        """Test VAE loss computation."""
        recon_x = torch.rand(32, 784)
        x = torch.rand(32, 784)
        mu = torch.randn(32, 20)
        logvar = torch.randn(32, 20)
        
        losses = vae_loss(recon_x, x, mu, logvar, beta=1.0)
        
        assert "total_loss" in losses
        assert "recon_loss" in losses
        assert "kl_loss" in losses
        
        assert losses["total_loss"] > 0
        assert losses["recon_loss"] > 0
        assert losses["kl_loss"] > 0
    
    def test_beta_parameter(self):
        """Test beta parameter effect on loss."""
        recon_x = torch.rand(32, 784)
        x = torch.rand(32, 784)
        mu = torch.randn(32, 20)
        logvar = torch.randn(32, 20)
        
        losses_beta_0 = vae_loss(recon_x, x, mu, logvar, beta=0.0)
        losses_beta_1 = vae_loss(recon_x, x, mu, logvar, beta=1.0)
        
        # With beta=0, total loss should equal reconstruction loss
        assert torch.allclose(losses_beta_0["total_loss"], losses_beta_0["recon_loss"])
        
        # With beta=1, total loss should be higher
        assert losses_beta_1["total_loss"] > losses_beta_0["total_loss"]


class TestKLAnnealing:
    """Test cases for KL annealing schedule."""
    
    def test_linear_schedule(self):
        """Test linear KL annealing schedule."""
        beta_start = kl_annealing_schedule(0, 100, "linear", 0.0, 1.0)
        beta_mid = kl_annealing_schedule(50, 100, "linear", 0.0, 1.0)
        beta_end = kl_annealing_schedule(100, 100, "linear", 0.0, 1.0)
        
        assert beta_start == 0.0
        assert beta_mid == 0.5
        assert beta_end == 1.0
    
    def test_cyclical_schedule(self):
        """Test cyclical KL annealing schedule."""
        beta_start = kl_annealing_schedule(0, 100, "cyclical", 0.0, 1.0)
        beta_end = kl_annealing_schedule(100, 100, "cyclical", 0.0, 1.0)
        
        assert beta_start == 1.0  # cos(0) = 1
        assert beta_end == 1.0     # cos(π) = -1, but we use (1 + cos(π))/2 = 0
    
    def test_sigmoid_schedule(self):
        """Test sigmoid KL annealing schedule."""
        beta_start = kl_annealing_schedule(0, 100, "sigmoid", 0.0, 1.0)
        beta_end = kl_annealing_schedule(100, 100, "sigmoid", 0.0, 1.0)
        
        assert beta_start < 0.1  # Should be close to 0
        assert beta_end > 0.9    # Should be close to 1


class TestVAESampler:
    """Test cases for VAE sampler."""
    
    def test_initialization(self):
        """Test sampler initialization."""
        model = BetaVAE(input_dim=784, latent_dim=20)
        device = get_device()
        
        sampler = VAESampler(model, device, seed=42)
        
        assert sampler.model == model
        assert sampler.device == device
    
    def test_random_sampling(self):
        """Test random sampling."""
        model = BetaVAE(input_dim=784, latent_dim=20)
        device = get_device()
        
        sampler = VAESampler(model, device, seed=42)
        samples = sampler.sample_random(16)
        
        assert samples.shape == (16, 784)
        assert torch.all(samples >= 0) and torch.all(samples <= 1)
    
    def test_interpolation_sampling(self):
        """Test interpolation sampling."""
        model = BetaVAE(input_dim=784, latent_dim=20)
        device = get_device()
        
        sampler = VAESampler(model, device, seed=42)
        x1 = torch.randn(1, 784)
        x2 = torch.randn(1, 784)
        
        interpolated = sampler.sample_interpolation(x1, x2, num_steps=5)
        
        assert interpolated.shape == (5, 784)
        assert torch.all(interpolated >= 0) and torch.all(interpolated <= 1)
    
    def test_latent_traversal(self):
        """Test latent space traversal."""
        model = BetaVAE(input_dim=784, latent_dim=20)
        device = get_device()
        
        sampler = VAESampler(model, device, seed=42)
        base_sample = torch.randn(1, 784)
        
        traversed = sampler.sample_latent_traversal(base_sample, latent_dim=0, num_steps=5)
        
        assert traversed.shape == (5, 784)
        assert torch.all(traversed >= 0) and torch.all(traversed <= 1)


class TestVAEEvaluator:
    """Test cases for VAE evaluator."""
    
    def test_initialization(self):
        """Test evaluator initialization."""
        model = BetaVAE(input_dim=784, latent_dim=20)
        device = get_device()
        
        evaluator = VAEEvaluator(model, device, use_fid=False)
        
        assert evaluator.model == model
        assert evaluator.device == device
    
    def test_reconstruction_metrics(self):
        """Test reconstruction metrics computation."""
        model = BetaVAE(input_dim=784, latent_dim=20)
        device = get_device()
        
        evaluator = VAEEvaluator(model, device, use_fid=False)
        
        # Create dummy data
        x = torch.rand(32, 784)
        recon_x = torch.rand(32, 784)
        mu = torch.randn(32, 20)
        logvar = torch.randn(32, 20)
        
        # Test metrics update
        evaluator.metrics["reconstruction"](recon_x, x)
        evaluator.metrics["latent_space"](mu, logvar)
        
        # Test metrics computation
        results = evaluator.metrics.compute()
        
        assert "reconstruction" in results
        assert "latent_space" in results
        assert "ssim" in results
        assert "psnr" in results


class TestDataUtilities:
    """Test cases for data utilities."""
    
    def test_get_device(self):
        """Test device selection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Generate some random numbers
        torch_rand = torch.randn(10)
        np_rand = np.random.randn(10)
        
        # Set seed again and generate same numbers
        set_seed(42)
        torch_rand2 = torch.randn(10)
        np_rand2 = np.random.randn(10)
        
        assert torch.allclose(torch_rand, torch_rand2)
        assert np.allclose(np_rand, np_rand2)


if __name__ == "__main__":
    pytest.main([__file__])
