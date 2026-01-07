"""
Configuration module for LIME image enhancement.

This module provides configuration parameters for the LIME algorithm.
"""
from dataclasses import dataclass


@dataclass
class Config:
    """Configuration structure for LIME image enhancement."""
    
    # File paths (relative to project root)
    input_path: str = 'images/cars.bmp'  # Relative to project root
    output_dir: str = 'output'            # Relative to project root
    
    # Algorithm parameters
    alpha: float = 3.0          # Structure preservation weight
    gamma: float = 0.8          # Gamma correction parameter
    rho: float = 1.15           # ADMM penalty parameter update rate
    mu: float = 0.05            # Initial ADMM penalty parameter
    num_iterations: int = 50    # Number of ADMM iterations
    
    # Output options
    display_results: bool = True
    save_results: bool = True


def get_config():
    """
    Creates a configuration object for LIME image enhancement.
    
    Returns:
        Config: Configuration object containing all parameters
    
    Configuration fields:
        - input_path: Path to input image file
        - output_dir: Directory for saving output images
        - alpha: Weight parameter for structure preservation (default: 3)
        - gamma: Gamma correction parameter (default: 0.8)
        - rho: ADMM penalty parameter update rate (default: 1.15)
        - mu: Initial ADMM penalty parameter (default: 0.05)
        - num_iterations: Number of ADMM iterations (default: 50)
        - display_results: Whether to display results in figures (default: True)
        - save_results: Whether to save output images (default: True)
    """
    return Config()

