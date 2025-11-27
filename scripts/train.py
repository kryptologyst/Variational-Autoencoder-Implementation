#!/usr/bin/env python3
"""
Main training script for VAE models.

This script provides a complete training pipeline with:
- Configuration management
- Data loading
- Model training
- Evaluation
- Visualization
- Checkpointing
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from vae.data import create_data_loaders, get_device, set_seed
from vae.models import BetaVAE, ConvVAE
from vae.training import VAELightningModule
from vae.evaluation import VAEEvaluator
from vae.visualization import VAEVisualizer, VAESampler


def setup_logging(config: Dict[str, Any]) -> Optional[pl.loggers.Logger]:
    """
    Setup logging based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Logger instance or None
    """
    logging_config = config.get("logging", {})
    
    if logging_config.get("wandb", {}).get("project"):
        wandb_config = logging_config["wandb"]
        logger = WandbLogger(
            project=wandb_config["project"],
            entity=wandb_config.get("entity"),
            tags=wandb_config.get("tags", []),
            notes=wandb_config.get("notes", ""),
        )
    elif logging_config.get("tensorboard", {}).get("log_dir"):
        tb_config = logging_config["tensorboard"]
        logger = TensorBoardLogger(
            save_dir=tb_config["log_dir"],
            comment=tb_config.get("comment", ""),
        )
    else:
        logger = None
    
    return logger


def setup_callbacks(config: Dict[str, Any]) -> list:
    """
    Setup training callbacks.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Model checkpointing
    training_config = config.get("training", {})
    checkpoint_config = training_config.get("checkpointing", {})
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["paths"]["checkpoints_dir"],
        filename="vae-{epoch:02d}-{val_loss:.2f}",
        monitor=checkpoint_config.get("monitor", "val/total_loss"),
        mode=checkpoint_config.get("mode", "min"),
        save_top_k=checkpoint_config.get("save_top_k", 3),
        every_n_epochs=checkpoint_config.get("save_every_n_epochs", 10),
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    if training_config.get("early_stopping", False):
        early_stop_callback = EarlyStopping(
            monitor=checkpoint_config.get("monitor", "val/total_loss"),
            mode=checkpoint_config.get("mode", "min"),
            patience=training_config.get("patience", 20),
            verbose=True,
        )
        callbacks.append(early_stop_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)
    
    return callbacks


def train_model(config: Dict[str, Any]) -> pl.Trainer:
    """
    Train VAE model using PyTorch Lightning.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Trained trainer instance
    """
    # Set random seed
    seed = config.get("seed", 42)
    set_seed(seed)
    
    # Create data loaders
    data_config = config["data"]
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_name=data_config["dataset"],
        data_dir=data_config["data_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        image_size=data_config["image_size"],
        augment=data_config["augment"],
        val_split=data_config["val_split"],
        test_split=data_config["test_split"],
    )
    
    # Create model
    model = VAELightningModule(
        model_config=config["model"],
        training_config=config["training"],
        data_config=data_config,
    )
    
    # Setup logging
    logger = setup_logging(config)
    
    # Setup callbacks
    callbacks = setup_callbacks(config)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        logger=logger,
        callbacks=callbacks,
        accelerator="auto",
        devices="auto",
        precision="16-mixed" if torch.cuda.is_available() else "32",
        gradient_clip_val=config["training"].get("grad_clip", 0.0),
        log_every_n_steps=config["logging"].get("log_every_n_steps", 50),
        deterministic=config.get("deterministic", True),
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Test model
    if test_loader is not None:
        trainer.test(model, test_loader)
    
    return trainer


def evaluate_model(config: Dict[str, Any], checkpoint_path: Optional[str] = None) -> None:
    """
    Evaluate trained VAE model.
    
    Args:
        config: Configuration dictionary
        checkpoint_path: Path to model checkpoint
    """
    # Set random seed
    seed = config.get("seed", 42)
    set_seed(seed)
    
    # Create data loaders
    data_config = config["data"]
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_name=data_config["dataset"],
        data_dir=data_config["data_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        image_size=data_config["image_size"],
        augment=False,  # No augmentation for evaluation
        val_split=data_config["val_split"],
        test_split=data_config["test_split"],
    )
    
    # Load model
    if checkpoint_path:
        model = VAELightningModule.load_from_checkpoint(checkpoint_path)
    else:
        model = VAELightningModule(
            model_config=config["model"],
            training_config=config["training"],
            data_config=data_config,
        )
    
    # Get device
    device = get_device()
    model = model.to(device)
    
    # Create evaluator
    evaluator = VAEEvaluator(
        model=model.model,
        device=device,
        use_fid=config["evaluation"].get("compute_fid", True),
    )
    
    # Run evaluation
    print("Running comprehensive evaluation...")
    results = evaluator.comprehensive_evaluation(
        dataloader=test_loader,
        num_samples=config["evaluation"]["num_samples"],
        num_batches=config["evaluation"].get("num_batches"),
    )
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for category, metrics in results.items():
        print(f"\n{category.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Save results
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    import json
    with open(results_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {results_dir / 'evaluation_results.json'}")


def visualize_model(config: Dict[str, Any], checkpoint_path: Optional[str] = None) -> None:
    """
    Create visualizations for trained VAE model.
    
    Args:
        config: Configuration dictionary
        checkpoint_path: Path to model checkpoint
    """
    # Set random seed
    seed = config.get("seed", 42)
    set_seed(seed)
    
    # Create data loaders
    data_config = config["data"]
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_name=data_config["dataset"],
        data_dir=data_config["data_dir"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        image_size=data_config["image_size"],
        augment=False,  # No augmentation for visualization
        val_split=data_config["val_split"],
        test_split=data_config["test_split"],
    )
    
    # Load model
    if checkpoint_path:
        model = VAELightningModule.load_from_checkpoint(checkpoint_path)
    else:
        model = VAELightningModule(
            model_config=config["model"],
            training_config=config["training"],
            data_config=data_config,
        )
    
    # Get device
    device = get_device()
    model = model.to(device)
    
    # Create sampler and visualizer
    sampler = VAESampler(model.model, device, seed=seed)
    visualizer = VAEVisualizer(
        sampler=sampler,
        save_dir=config["visualization"]["save_dir"],
    )
    
    # Create comprehensive visualization
    visualizer.create_comprehensive_visualization(
        dataloader=test_loader,
        num_samples=config["sampling"]["num_samples"],
        num_reconstructions=16,
        num_interpolations=5,
        latent_dims_to_traverse=config["evaluation"]["latent_dims_to_traverse"],
    )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train and evaluate VAE models")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "visualize", "all"],
        default="all",
        help="Mode to run",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint for evaluation/visualization",
    )
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        help="Override configuration values (e.g., training.max_epochs=50)",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Apply overrides
    if args.override:
        for override in args.override:
            key, value = override.split("=", 1)
            OmegaConf.set(config, key, value)
    
    # Create necessary directories
    for path_key in ["data_dir", "checkpoints_dir", "logs_dir", "assets_dir", "results_dir"]:
        Path(config["paths"][path_key]).mkdir(parents=True, exist_ok=True)
    
    # Run based on mode
    if args.mode in ["train", "all"]:
        print("Starting training...")
        trainer = train_model(config)
        print("Training completed!")
    
    if args.mode in ["eval", "all"]:
        print("Starting evaluation...")
        evaluate_model(config, args.checkpoint)
        print("Evaluation completed!")
    
    if args.mode in ["visualize", "all"]:
        print("Starting visualization...")
        visualize_model(config, args.checkpoint)
        print("Visualization completed!")


if __name__ == "__main__":
    main()
