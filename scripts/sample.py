#!/usr/bin/env python3
"""
Sampling script for VAE models.

This script provides utilities for:
- Generating samples from trained models
- Interpolation between samples
- Latent space traversal
- Batch sampling for evaluation
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import numpy as np
from omegaconf import OmegaConf

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from vae.data import get_device, set_seed
from vae.training import VAELightningModule
from vae.visualization import VAEVisualizer, VAESampler


def load_model(config: Dict[str, Any], checkpoint_path: str) -> VAELightningModule:
    """
    Load trained VAE model from checkpoint.
    
    Args:
        config: Configuration dictionary
        checkpoint_path: Path to model checkpoint
        
    Returns:
        Loaded model
    """
    model = VAELightningModule.load_from_checkpoint(checkpoint_path)
    return model


def generate_samples(
    model: VAELightningModule,
    config: Dict[str, Any],
    num_samples: int,
    output_dir: str,
    seed: Optional[int] = None,
) -> None:
    """
    Generate and save samples from the model.
    
    Args:
        model: Trained VAE model
        config: Configuration dictionary
        num_samples: Number of samples to generate
        output_dir: Directory to save samples
        seed: Random seed
    """
    if seed is not None:
        set_seed(seed)
    
    device = get_device()
    model = model.to(device)
    
    # Create sampler
    sampler = VAESampler(model.model, device, seed=seed)
    
    # Generate samples
    print(f"Generating {num_samples} samples...")
    samples = sampler.sample_random(num_samples)
    
    # Save samples
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as individual images
    for i, sample in enumerate(samples):
        torch.save(sample, output_path / f"sample_{i:06d}.pt")
    
    # Save as grid
    from torchvision.utils import save_image
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    save_image(
        samples,
        output_path / "samples_grid.png",
        nrow=grid_size,
        normalize=True,
        pad_value=1,
    )
    
    print(f"Samples saved to {output_path}")


def create_interpolations(
    model: VAELightningModule,
    config: Dict[str, Any],
    dataloader,
    num_pairs: int,
    num_steps: int,
    output_dir: str,
) -> None:
    """
    Create interpolations between random pairs of images.
    
    Args:
        model: Trained VAE model
        config: Configuration dictionary
        dataloader: Data loader for source images
        num_pairs: Number of interpolation pairs
        num_steps: Number of interpolation steps
        output_dir: Directory to save interpolations
    """
    device = get_device()
    model = model.to(device)
    
    # Create sampler and visualizer
    sampler = VAESampler(model.model, device)
    visualizer = VAEVisualizer(sampler, save_dir=output_dir)
    
    # Create interpolations
    print(f"Creating {num_pairs} interpolations...")
    
    for i in range(num_pairs):
        # Get random pair from dataloader
        data, _ = next(iter(dataloader))
        if data.size(0) < 2:
            continue
        
        x1, x2 = data[0:1], data[1:2]
        
        # Create interpolation
        interpolated = sampler.sample_interpolation(x1, x2, num_steps)
        
        # Save interpolation
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        from torchvision.utils import save_image
        save_image(
            interpolated,
            output_path / f"interpolation_{i:03d}.png",
            nrow=num_steps,
            normalize=True,
            pad_value=1,
        )
    
    print(f"Interpolations saved to {output_dir}")


def create_latent_traversals(
    model: VAELightningModule,
    config: Dict[str, Any],
    dataloader,
    latent_dims: list,
    num_steps: int,
    step_size: float,
    output_dir: str,
) -> None:
    """
    Create latent space traversals.
    
    Args:
        model: Trained VAE model
        config: Configuration dictionary
        dataloader: Data loader for base images
        latent_dims: List of latent dimensions to traverse
        num_steps: Number of steps in traversal
        step_size: Size of each step
        output_dir: Directory to save traversals
    """
    device = get_device()
    model = model.to(device)
    
    # Create sampler and visualizer
    sampler = VAESampler(model.model, device)
    visualizer = VAEVisualizer(sampler, save_dir=output_dir)
    
    # Get base sample
    data, _ = next(iter(dataloader))
    base_sample = data[0:1]
    
    # Create traversals
    print(f"Creating traversals for dimensions {latent_dims}...")
    
    for dim in latent_dims:
        # Create traversal
        traversed = sampler.sample_latent_traversal(
            base_sample, dim, num_steps, step_size
        )
        
        # Save traversal
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        from torchvision.utils import save_image
        save_image(
            traversed,
            output_path / f"traversal_dim_{dim:02d}.png",
            nrow=num_steps,
            normalize=True,
            pad_value=1,
        )
    
    print(f"Traversals saved to {output_dir}")


def batch_evaluate(
    model: VAELightningModule,
    config: Dict[str, Any],
    dataloader,
    output_dir: str,
) -> None:
    """
    Run batch evaluation and save results.
    
    Args:
        model: Trained VAE model
        config: Configuration dictionary
        dataloader: Data loader for evaluation
        output_dir: Directory to save results
    """
    device = get_device()
    model = model.to(device)
    
    # Create evaluator
    from vae.evaluation import VAEEvaluator
    evaluator = VAEEvaluator(
        model=model.model,
        device=device,
        use_fid=config["evaluation"].get("compute_fid", True),
    )
    
    # Run evaluation
    print("Running batch evaluation...")
    results = evaluator.comprehensive_evaluation(
        dataloader=dataloader,
        num_samples=config["evaluation"]["num_samples"],
        num_batches=config["evaluation"].get("num_batches"),
    )
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Evaluation results saved to {output_path / 'evaluation_results.json'}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Sample from trained VAE models")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["samples", "interpolations", "traversals", "evaluate", "all"],
        default="all",
        help="Mode to run",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./assets/samples",
        help="Output directory for samples",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=10,
        help="Number of interpolation pairs",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Number of interpolation/traversal steps",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=2.0,
        help="Step size for latent traversals",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(config, args.checkpoint)
    
    # Create data loader for interpolation/traversal
    if args.mode in ["interpolations", "traversals", "evaluate", "all"]:
        from vae.data import create_data_loaders
        data_config = config["data"]
        _, _, test_loader = create_data_loaders(
            dataset_name=data_config["dataset"],
            data_dir=data_config["data_dir"],
            batch_size=config["training"]["batch_size"],
            num_workers=config["training"]["num_workers"],
            image_size=data_config["image_size"],
            augment=False,
            val_split=data_config["val_split"],
            test_split=data_config["test_split"],
        )
    
    # Run based on mode
    if args.mode in ["samples", "all"]:
        generate_samples(
            model=model,
            config=config,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    
    if args.mode in ["interpolations", "all"]:
        create_interpolations(
            model=model,
            config=config,
            dataloader=test_loader,
            num_pairs=args.num_pairs,
            num_steps=args.num_steps,
            output_dir=args.output_dir,
        )
    
    if args.mode in ["traversals", "all"]:
        create_latent_traversals(
            model=model,
            config=config,
            dataloader=test_loader,
            latent_dims=config["evaluation"]["latent_dims_to_traverse"],
            num_steps=args.num_steps,
            step_size=args.step_size,
            output_dir=args.output_dir,
        )
    
    if args.mode in ["evaluate", "all"]:
        batch_evaluate(
            model=model,
            config=config,
            dataloader=test_loader,
            output_dir=args.output_dir,
        )
    
    print("Sampling completed!")


if __name__ == "__main__":
    main()
