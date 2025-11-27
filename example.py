#!/usr/bin/env python3
"""
Simple example script demonstrating VAE functionality.

This script shows how to:
1. Create a VAE model
2. Train it on MNIST
3. Generate samples
4. Visualize results
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from vae.models import BetaVAE, vae_loss
from vae.data import create_data_loaders, get_device, set_seed
from vae.visualization import VAEVisualizer, VAESampler


def main():
    """Main example function."""
    print("VAE Example Script")
    print("=" * 50)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Loading MNIST dataset...")
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_name="mnist",
        data_dir="./data",
        batch_size=128,
        num_workers=2,
        image_size=28,
        augment=False,
        val_split=0.1,
        test_split=0.1,
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nCreating VAE model...")
    model = BetaVAE(
        input_dim=784,
        hidden_dims=[400, 200],
        latent_dim=20,
        beta=1.0,
        use_batch_norm=True,
        dropout=0.1,
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    print("\nStarting training...")
    model.train()
    
    for epoch in range(5):  # Quick training for demo
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            # Forward pass
            recon_data, mu, logvar = model(data)
            
            # Compute loss
            losses = vae_loss(recon_data, data, mu, logvar, beta=1.0)
            
            # Backward pass
            optimizer.zero_grad()
            losses["total_loss"].backward()
            optimizer.step()
            
            # Accumulate losses
            total_loss += losses["total_loss"].item()
            total_recon_loss += losses["recon_loss"].item()
            total_kl_loss += losses["kl_loss"].item()
        
        # Print epoch results
        avg_loss = total_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_kl_loss = total_kl_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}/5 - Loss: {avg_loss:.4f}, Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}")
    
    print("Training completed!")
    
    # Generate samples
    print("\nGenerating samples...")
    model.eval()
    
    # Create sampler and visualizer
    sampler = VAESampler(model, device, seed=42)
    visualizer = VAEVisualizer(sampler, save_dir="./assets")
    
    # Generate random samples
    samples = sampler.sample_random(16)
    
    # Create sample grid
    grid = make_grid(samples, nrow=4, normalize=True, pad_value=1)
    
    # Display samples
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title("Generated Samples")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("./assets/generated_samples.png", dpi=150, bbox_inches="tight")
    plt.show()
    
    # Test reconstruction
    print("\nTesting reconstruction...")
    with torch.no_grad():
        # Get a batch of test data
        test_data, _ = next(iter(test_loader))
        test_data = test_data.to(device)
        
        # Reconstruct
        recon_data, mu, logvar = model(test_data)
        
        # Create comparison grid
        comparison = torch.cat([test_data[:8], recon_data[:8]], dim=0)
        grid = make_grid(comparison, nrow=8, normalize=True, pad_value=1)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.title("Original (top) vs Reconstructed (bottom)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("./assets/reconstructions.png", dpi=150, bbox_inches="tight")
        plt.show()
    
    # Test interpolation
    print("\nTesting interpolation...")
    with torch.no_grad():
        # Get two random samples
        x1 = sampler.sample_random(1)
        x2 = sampler.sample_random(1)
        
        # Create interpolation
        interpolated = sampler.sample_interpolation(x1, x2, num_steps=10)
        
        # Create interpolation grid
        grid = make_grid(interpolated, nrow=10, normalize=True, pad_value=1)
        
        plt.figure(figsize=(15, 3))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.title("Latent Space Interpolation")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("./assets/interpolation.png", dpi=150, bbox_inches="tight")
        plt.show()
    
    # Test latent traversal
    print("\nTesting latent traversal...")
    with torch.no_grad():
        # Get base sample
        base_sample = sampler.sample_random(1)
        
        # Create traversal along first dimension
        traversed = sampler.sample_latent_traversal(base_sample, latent_dim=0, num_steps=10)
        
        # Create traversal grid
        grid = make_grid(traversed, nrow=10, normalize=True, pad_value=1)
        
        plt.figure(figsize=(15, 3))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.title("Latent Space Traversal (Dimension 0)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("./assets/traversal.png", dpi=150, bbox_inches="tight")
        plt.show()
    
    print("\nExample completed successfully!")
    print("Generated files:")
    print("- ./assets/generated_samples.png")
    print("- ./assets/reconstructions.png")
    print("- ./assets/interpolation.png")
    print("- ./assets/traversal.png")


if __name__ == "__main__":
    main()
