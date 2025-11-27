#!/usr/bin/env python3
"""
Setup script for VAE implementation.

This script helps set up the project environment and run initial tests.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False


def setup_project():
    """Set up the project environment."""
    print("Setting up VAE implementation project...")
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("✗ Python 3.10+ is required")
        return False
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        return False
    
    # Create necessary directories
    directories = ["data", "checkpoints", "logs", "assets", "results"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Run tests
    if not run_command("python -m pytest tests/ -v", "Running tests"):
        print("⚠ Tests failed, but continuing with setup")
    
    # Test training script
    if not run_command(
        "python scripts/train.py --config configs/config.yaml --override training.max_epochs=1 training.batch_size=32",
        "Testing training script"
    ):
        print("⚠ Training test failed, but continuing with setup")
    
    print("\n✓ Project setup completed successfully!")
    print("\nNext steps:")
    print("1. Train a model: python scripts/train.py --config configs/config.yaml")
    print("2. Launch demo: streamlit run demo/streamlit_app.py")
    print("3. Generate samples: python scripts/sample.py --checkpoint checkpoints/last.ckpt")
    
    return True


if __name__ == "__main__":
    success = setup_project()
    sys.exit(0 if success else 1)
