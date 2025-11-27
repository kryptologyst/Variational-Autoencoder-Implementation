#!/usr/bin/env python3
"""
Streamlit demo for VAE models.

This demo provides an interactive interface for:
- Generating samples from trained models
- Interpolation between images
- Latent space traversal
- Model comparison
"""

import sys
from pathlib import Path

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from vae.data import get_device, set_seed
from vae.training import VAELightningModule
from vae.visualization import VAEVisualizer, VAESampler


@st.cache_resource
def load_model(checkpoint_path: str):
    """Load VAE model from checkpoint."""
    try:
        model = VAELightningModule.load_from_checkpoint(checkpoint_path)
        device = get_device()
        model = model.to(device)
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image."""
    # Ensure tensor is in [0, 1] range
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and transpose for PIL
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first image if batch
    
    numpy_array = tensor.permute(1, 2, 0).cpu().numpy()
    
    # Convert to uint8
    numpy_array = (numpy_array * 255).astype(np.uint8)
    
    return Image.fromarray(numpy_array)


def create_sample_grid(samples: torch.Tensor, nrow: int = 8) -> Image.Image:
    """Create a grid of samples."""
    from torchvision.utils import make_grid
    
    grid = make_grid(samples, nrow=nrow, normalize=True, pad_value=1)
    return tensor_to_image(grid)


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="VAE Demo",
        page_icon="🎨",
        layout="wide",
    )
    
    st.title("🎨 Variational Autoencoder Demo")
    st.markdown("Interactive demo for generating and exploring VAE models")
    
    # Sidebar for model selection
    st.sidebar.header("Model Configuration")
    
    # Model selection
    checkpoint_dir = Path("checkpoints")
    if checkpoint_dir.exists():
        checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
        if checkpoint_files:
            checkpoint_path = st.sidebar.selectbox(
                "Select Model Checkpoint",
                checkpoint_files,
                format_func=lambda x: x.name,
            )
        else:
            st.sidebar.error("No checkpoint files found in checkpoints/")
            return
    else:
        st.sidebar.error("Checkpoints directory not found")
        return
    
    # Load model
    model, device = load_model(str(checkpoint_path))
    if model is None:
        return
    
    # Sampling parameters
    st.sidebar.header("Sampling Parameters")
    num_samples = st.sidebar.slider("Number of Samples", 1, 64, 16)
    seed = st.sidebar.number_input("Random Seed", value=42, min_value=0, max_value=2**32-1)
    
    # Create sampler
    sampler = VAESampler(model.model, device, seed=seed)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎲 Random Samples", "🔄 Interpolation", "🧭 Latent Traversal", "📊 Model Info"])
    
    with tab1:
        st.header("Random Sample Generation")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("Generate Samples", type="primary"):
                with st.spinner("Generating samples..."):
                    # Generate samples
                    samples = sampler.sample_random(num_samples)
                    
                    # Create grid
                    grid_image = create_sample_grid(samples, nrow=int(np.ceil(np.sqrt(num_samples))))
                    
                    # Display
                    st.image(grid_image, caption=f"Generated {num_samples} samples")
                    
                    # Download button
                    buf = io.BytesIO()
                    grid_image.save(buf, format="PNG")
                    st.download_button(
                        label="Download Samples",
                        data=buf.getvalue(),
                        file_name=f"vae_samples_{num_samples}.png",
                        mime="image/png",
                    )
        
        with col2:
            st.subheader("Parameters")
            st.write(f"**Model:** {checkpoint_path.name}")
            st.write(f"**Samples:** {num_samples}")
            st.write(f"**Seed:** {seed}")
            st.write(f"**Device:** {device}")
    
    with tab2:
        st.header("Latent Space Interpolation")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Source Images")
            st.write("Upload two images or use random samples from the dataset")
            
            use_random = st.checkbox("Use Random Samples", value=True)
            
            if use_random:
                if st.button("Generate Random Pair"):
                    # Get random samples from model
                    sample1 = sampler.sample_random(1)
                    sample2 = sampler.sample_random(1)
                    
                    st.session_state.interp_image1 = sample1[0]
                    st.session_state.interp_image2 = sample2[0]
            
            if "interp_image1" in st.session_state:
                st.image(tensor_to_image(st.session_state.interp_image1), caption="Image 1")
                st.image(tensor_to_image(st.session_state.interp_image2), caption="Image 2")
        
        with col2:
            st.subheader("Interpolation")
            
            num_steps = st.slider("Number of Steps", 3, 20, 10)
            
            if st.button("Create Interpolation", type="primary"):
                if "interp_image1" in st.session_state:
                    with st.spinner("Creating interpolation..."):
                        # Create interpolation
                        interpolated = sampler.sample_interpolation(
                            st.session_state.interp_image1.unsqueeze(0),
                            st.session_state.interp_image2.unsqueeze(0),
                            num_steps
                        )
                        
                        # Create grid
                        grid_image = create_sample_grid(interpolated, nrow=num_steps)
                        
                        # Display
                        st.image(grid_image, caption=f"Interpolation with {num_steps} steps")
                        
                        # Download button
                        buf = io.BytesIO()
                        grid_image.save(buf, format="PNG")
                        st.download_button(
                            label="Download Interpolation",
                            data=buf.getvalue(),
                            file_name=f"vae_interpolation_{num_steps}.png",
                            mime="image/png",
                        )
                else:
                    st.warning("Please generate or upload source images first")
    
    with tab3:
        st.header("Latent Space Traversal")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Traversal Parameters")
            
            # Get base sample
            if st.button("Generate Base Sample"):
                base_sample = sampler.sample_random(1)
                st.session_state.base_sample = base_sample[0]
            
            if "base_sample" in st.session_state:
                st.image(tensor_to_image(st.session_state.base_sample), caption="Base Sample")
            
            # Traversal parameters
            latent_dim = st.slider("Latent Dimension", 0, model.model.latent_dim - 1, 0)
            num_steps = st.slider("Number of Steps", 5, 20, 10)
            step_size = st.slider("Step Size", 0.5, 5.0, 2.0)
            
            if st.button("Create Traversal", type="primary"):
                if "base_sample" in st.session_state:
                    with st.spinner("Creating traversal..."):
                        # Create traversal
                        traversed = sampler.sample_latent_traversal(
                            st.session_state.base_sample.unsqueeze(0),
                            latent_dim,
                            num_steps,
                            step_size
                        )
                        
                        # Create grid
                        grid_image = create_sample_grid(traversed, nrow=num_steps)
                        
                        # Display
                        st.image(grid_image, caption=f"Traversal along dimension {latent_dim}")
                        
                        # Download button
                        buf = io.BytesIO()
                        grid_image.save(buf, format="PNG")
                        st.download_button(
                            label="Download Traversal",
                            data=buf.getvalue(),
                            file_name=f"vae_traversal_dim_{latent_dim}.png",
                            mime="image/png",
                        )
                else:
                    st.warning("Please generate a base sample first")
        
        with col2:
            st.subheader("Latent Space Information")
            st.write(f"**Latent Dimension:** {model.model.latent_dim}")
            st.write(f"**Model Type:** {type(model.model).__name__}")
            
            # Show latent space statistics
            if st.button("Analyze Latent Space"):
                with st.spinner("Analyzing latent space..."):
                    # Generate samples and analyze latent space
                    samples = sampler.sample_random(1000)
                    
                    # Encode samples
                    model.model.eval()
                    with torch.no_grad():
                        mu, logvar = model.model.encode(samples)
                    
                    # Compute statistics
                    mu_mean = mu.mean(dim=0)
                    mu_std = mu.std(dim=0)
                    logvar_mean = logvar.mean(dim=0)
                    
                    # Display statistics
                    st.write("**Latent Space Statistics:**")
                    st.write(f"Mean of means: {mu_mean.mean().item():.4f}")
                    st.write(f"Std of means: {mu_std.mean().item():.4f}")
                    st.write(f"Mean of log-vars: {logvar_mean.mean().item():.4f}")
    
    with tab4:
        st.header("Model Information")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Model Architecture")
            st.write(f"**Type:** {type(model.model).__name__}")
            st.write(f"**Latent Dimension:** {model.model.latent_dim}")
            
            if hasattr(model.model, 'input_dim'):
                st.write(f"**Input Dimension:** {model.model.input_dim}")
            
            if hasattr(model.model, 'hidden_dims'):
                st.write(f"**Hidden Dimensions:** {model.model.hidden_dims}")
            
            # Model parameters
            total_params = sum(p.numel() for p in model.model.parameters())
            trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
            
            st.write(f"**Total Parameters:** {total_params:,}")
            st.write(f"**Trainable Parameters:** {trainable_params:,}")
        
        with col2:
            st.subheader("Training Configuration")
            st.write(f"**Max Epochs:** {model.training_config.get('max_epochs', 'N/A')}")
            st.write(f"**Learning Rate:** {model.training_config.get('learning_rate', 'N/A')}")
            st.write(f"**Batch Size:** {model.training_config.get('batch_size', 'N/A')}")
            st.write(f"**Beta Start:** {model.training_config.get('beta_start', 'N/A')}")
            st.write(f"**Beta End:** {model.training_config.get('beta_end', 'N/A')}")
            st.write(f"**Beta Schedule:** {model.training_config.get('beta_schedule', 'N/A')}")
        
        # Model performance (if available)
        st.subheader("Model Performance")
        st.info("Performance metrics would be displayed here if available from training logs")
        
        # Quick evaluation
        if st.button("Run Quick Evaluation"):
            with st.spinner("Running evaluation..."):
                # Generate samples for evaluation
                samples = sampler.sample_random(100)
                
                # Compute basic statistics
                mean_sample = samples.mean()
                std_sample = samples.std()
                
                st.write(f"**Sample Mean:** {mean_sample.item():.4f}")
                st.write(f"**Sample Std:** {std_sample.item():.4f}")
                
                # Compute diversity
                samples_flat = samples.view(samples.size(0), -1)
                pairwise_distances = torch.cdist(samples_flat, samples_flat, p=2)
                mask = torch.triu(torch.ones_like(pairwise_distances), diagonal=1).bool()
                diversity = torch.mean(pairwise_distances[mask]).item()
                
                st.write(f"**Diversity:** {diversity:.4f}")


if __name__ == "__main__":
    main()
