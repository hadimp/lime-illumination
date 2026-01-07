"""
Batch processing script for LIME image enhancement.

This script processes all images in the images folder using the LIME
enhancement algorithm.
"""
from pathlib import Path
import numpy as np
from PIL import Image

from config import get_config
from lime import LIME


def batch_process():
    """Processes all images in the images folder."""
    # Get script directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Load configuration
    cfg = get_config()
    
    # Create output directory if it doesn't exist
    if cfg.save_results:
        output_path = project_root / cfg.output_dir
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files in images directory
    images_dir = project_root / 'images'
    
    # Support multiple image formats
    image_extensions = ['.bmp', '.png', '.jpg', '.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(images_dir.glob(f'*{ext}')))
        image_files.extend(list(images_dir.glob(f'*{ext.upper()}')))
    
    if not image_files:
        print(f'No images found in {images_dir}')
        return
    
    print(f'Found {len(image_files)} images to process\n')
    
    # Create LIME enhancer with configured parameters
    enhancer = LIME(alpha=cfg.alpha,
                    gamma=cfg.gamma,
                    rho=cfg.rho,
                    mu=cfg.mu,
                    num_iterations=cfg.num_iterations)
    
    # Process each image
    for i, image_path in enumerate(image_files, 1):
        print(f'[{i}/{len(image_files)}] Processing: {image_path.name}')
        
        try:
            # Load input image
            input_image = np.array(Image.open(image_path))
            
            # Enhance image
            enhanced_image, results = enhancer.enhance(input_image)
            
            # Save results if requested
            if cfg.save_results:
                base_name = image_path.stem
                output_path = project_root / cfg.output_dir
                
                # Save initial enhanced image
                initial_path = output_path / f'{base_name}_initial.png'
                initial_img = Image.fromarray((np.clip(results['initial_enhanced'], 0, 1) * 255).astype(np.uint8))
                initial_img.save(initial_path)
                
                # Save refined enhanced image
                refined_path = output_path / f'{base_name}_refined.png'
                refined_img = Image.fromarray((np.clip(results['refined_enhanced'], 0, 1) * 255).astype(np.uint8))
                refined_img.save(refined_path)
                
                # Save final enhanced image
                final_path = output_path / f'{base_name}_enhanced.png'
                final_img = Image.fromarray((np.clip(enhanced_image, 0, 1) * 255).astype(np.uint8))
                final_img.save(final_path)
                
                print(f'  ✓ Saved: {base_name}_*.png')
            
        except Exception as e:
            print(f'  ✗ Error processing {image_path.name}: {str(e)}')
        
        print()
    
    print(f'Batch processing complete! Processed {len(image_files)} images.')


if __name__ == '__main__':
    batch_process()

