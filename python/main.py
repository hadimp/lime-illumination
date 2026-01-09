"""
Main script for low-light image enhancement using LIME algorithm.

This script demonstrates the usage of the LIME class for enhancing
low-light images. It loads configuration, processes the image, and
optionally displays and saves results.
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from config import get_config
from lime import LIME


def main():
    """Main function for LIME image enhancement."""
    # Load configuration
    cfg = get_config()
    
    # Get project root (parent of python/ directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Resolve paths relative to project root
    input_path = (project_root / cfg.input_path).resolve() if not Path(cfg.input_path).is_absolute() else Path(cfg.input_path)
    output_dir = (project_root / cfg.output_dir).resolve() if not Path(cfg.output_dir).is_absolute() else Path(cfg.output_dir)
    
    # Create output directory if it doesn't exist
    if cfg.save_results:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load input image
    if not input_path.exists():
        raise FileNotFoundError(f'Input image not found: {input_path}')
    
    input_image = np.array(Image.open(input_path))
    
    # Create LIME enhancer with configured parameters
    enhancer = LIME(alpha=cfg.alpha,
                    gamma=cfg.gamma,
                    rho=cfg.rho,
                    mu=cfg.mu,
                    num_iterations=cfg.num_iterations)
    
    # Enhance image
    print(f'Enhancing image: {input_path}')
    enhanced_image, results = enhancer.enhance(input_image)
    
    # Display results if requested
    if cfg.display_results:
        display_results(input_image, enhanced_image, results)
    
    # Save results if requested
    if cfg.save_results:
        save_results(output_dir, input_path, enhanced_image, results)
    
    print('Enhancement complete!')


def display_results(input_image, enhanced_image, results):
    """Displays input and enhanced images in separate figures."""
    # Normalize input image for display
    if input_image.dtype != np.uint8:
        input_display = (np.clip(input_image, 0, 1) * 255).astype(np.uint8)
    else:
        input_display = input_image
    
    # Normalize enhanced images for display
    initial_enhanced_display = (np.clip(results['initial_enhanced'], 0, 1) * 255).astype(np.uint8)
    refined_enhanced_display = (np.clip(results['refined_enhanced'], 0, 1) * 255).astype(np.uint8)
    enhanced_display = (np.clip(enhanced_image, 0, 1) * 255).astype(np.uint8)
    
    # Display images
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0, 0].imshow(input_display)
    axes[0, 0].set_title('Original Low-Light Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(initial_enhanced_display)
    axes[0, 1].set_title('Enhanced with Initial Illumination Map')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(refined_enhanced_display)
    axes[1, 0].set_title('Enhanced with Refined Illumination Map')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(enhanced_display)
    axes[1, 1].set_title('Final Enhanced Image (with Gamma Correction)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()


def save_results(output_dir, input_path, enhanced_image, results):
    """Saves enhanced images to output directory."""
    base_name = Path(input_path).stem
    
    # Save initial enhanced image
    initial_path = output_dir / f'{base_name}_initial.png'
    initial_img = Image.fromarray((np.clip(results['initial_enhanced'], 0, 1) * 255).astype(np.uint8))
    initial_img.save(initial_path)
    print(f'Saved: {initial_path}')
    
    # Save refined enhanced image
    refined_path = output_dir / f'{base_name}_refined.png'
    refined_img = Image.fromarray((np.clip(results['refined_enhanced'], 0, 1) * 255).astype(np.uint8))
    refined_img.save(refined_path)
    print(f'Saved: {refined_path}')
    
    # Save final enhanced image
    final_path = output_dir / f'{base_name}_enhanced.png'
    final_img = Image.fromarray((np.clip(enhanced_image, 0, 1) * 255).astype(np.uint8))
    final_img.save(final_path)
    print(f'Saved: {final_path}')


if __name__ == '__main__':
    main()

