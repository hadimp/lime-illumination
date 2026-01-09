"""
LIME: Low-Light Image Enhancement via Illumination Map Estimation

This module implements the LIME algorithm for enhancing low-light images
using the Retinex model and ADMM optimization.

Algorithm Overview:
    1. Estimate initial illumination map from input image
    2. Refine illumination map using ADMM optimization
    3. Apply gamma correction to refined map
    4. Enhance image by dividing by corrected illumination map

Reference:
    X. Guo, Y. Li, and H. Ling, "LIME: Low-Light Image Enhancement via
    Illumination Map Estimation," IEEE Transactions on Image Processing,
    vol. 26, pp. 982-993, 2017.

Example:
    enhancer = LIME()
    enhanced = enhancer.enhance(input_image)
"""
import numpy as np
from utils.compute_gradient import compute_gradient
from utils.compute_t_numerator import compute_t_numerator
from utils.compute_t_denominator import compute_t_denominator


class LIME:
    """
    Low-Light Image Enhancement via Illumination Map Estimation.
    
    This class implements the LIME algorithm for enhancing low-light images
    using the Retinex model and ADMM optimization.
    """
    
    def __init__(self, alpha=3, gamma=0.8, rho=1.15, mu=0.05, num_iterations=50):
        """
        Constructor for LIME image enhancer.
        
        Args:
            alpha (float, optional): Structure preservation weight (default: 3)
            gamma (float, optional): Gamma correction parameter (default: 0.8)
            rho (float, optional): ADMM penalty update rate (default: 1.15)
            mu (float, optional): Initial ADMM penalty (default: 0.05)
            num_iterations (int, optional): ADMM iterations (default: 50)
        
        Example:
            enhancer = LIME(alpha=3, gamma=0.8)
        """
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        if gamma <= 0 or gamma > 1:
            raise ValueError("gamma must be > 0 and <= 1")
        if rho <= 1:
            raise ValueError("rho must be > 1")
        if mu <= 0:
            raise ValueError("mu must be > 0")
        if num_iterations <= 0:
            raise ValueError("num_iterations must be > 0")
        
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.mu = mu
        self.num_iterations = num_iterations
    
    def enhance(self, input_image):
        """
        Enhances a low-light image using the LIME algorithm.
        
        Args:
            input_image: Input image (uint8, uint16, or float in [0,1])
                         Can be numpy array of shape (H, W, 3) or (H, W)
        
        Returns:
            enhanced_image: Final enhanced image (float in [0,1])
            results: dict containing intermediate results:
                - initial_map: Initial illumination map
                - refined_map: Refined illumination map
                - gamma_corrected_map: Gamma-corrected illumination map
                - initial_enhanced: Image enhanced with initial map
                - refined_enhanced: Image enhanced with refined map
        """
        # Convert to numpy array and normalize to [0, 1]
        if not isinstance(input_image, np.ndarray):
            raise TypeError("input_image must be a numpy array")
        
        # Convert to float in [0, 1]
        if input_image.dtype == np.uint8:
            input_image = input_image.astype(np.float64) / 255.0
        elif input_image.dtype == np.uint16:
            input_image = input_image.astype(np.float64) / 65535.0
        elif input_image.dtype != np.float64 and input_image.dtype != np.float32:
            input_image = input_image.astype(np.float64)
        
        # Ensure image is 3-channel
        if len(input_image.shape) == 2:
            input_image = np.stack([input_image, input_image, input_image], axis=2)
        elif len(input_image.shape) != 3 or input_image.shape[2] != 3:
            raise ValueError("input_image must be 2D grayscale or 3D RGB (H, W, 3)")
        
        # Step 1: Calculate initial illumination map
        initial_map = self._calculate_initial_illumination_map(input_image)
        
        # Step 2: Apply initial map
        initial_enhanced = self._apply_illumination_map(input_image, initial_map)
        
        # Step 3: Refine illumination map using ADMM
        refined_map = self._refine_illumination_map(initial_map)
        refined_map = np.abs(refined_map)  # Ensure non-negative
        
        # Step 4: Apply refined map
        refined_enhanced = self._apply_illumination_map(input_image, refined_map)
        
        # Step 5: Apply gamma correction
        gamma_corrected_map = refined_map ** self.gamma
        
        # Step 6: Final enhanced image
        enhanced_image = self._apply_illumination_map(input_image, gamma_corrected_map)
        
        # Store results
        results = {
            'initial_map': initial_map,
            'refined_map': refined_map,
            'gamma_corrected_map': gamma_corrected_map,
            'initial_enhanced': initial_enhanced,
            'refined_enhanced': refined_enhanced
        }
        
        return enhanced_image, results
    
    def _calculate_initial_illumination_map(self, image):
        """
        Estimates initial illumination map.
        
        The initial map is computed as the maximum intensity across RGB
        channels at each pixel, following the Retinex model assumption
        that illumination is the maximum channel value.
        
        Args:
            image: Input image (M x N x 3)
        
        Returns:
            initial_map: Initial illumination map (M x N)
        """
        height, width, _ = image.shape
        initial_map = np.zeros((height, width))
        
        for i in range(height):
            for j in range(width):
                # Find channel with highest intensity (Retinex assumption)
                initial_map[i, j] = max(image[i, j, 0],
                                        image[i, j, 1],
                                        image[i, j, 2])
                
                # Avoid division by zero
                if initial_map[i, j] == 0:
                    initial_map[i, j] = 1e-7
        
        return initial_map
    
    def _refine_illumination_map(self, initial_map):
        """
        Refines illumination map using ADMM.
        
        Solves the optimization problem:
            min_T ||T - L||_2^2 + alpha*||DT||_1
        
        where T is the refined map, L is initial map, D is gradient operator,
        and alpha controls structure preservation. Uses ADMM to solve.
        
        Args:
            initial_map: Initial illumination map (M x N)
        
        Returns:
            refined_map: Refined illumination map (M x N)
        """
        height, width = initial_map.shape
        
        # Initialize ADMM variables
        T = np.zeros((height, width))
        Z = np.zeros((2 * height, width))  # Dual variable for gradient
        G = np.zeros((2 * height, width))  # Auxiliary variable
        W = np.ones((2 * height, width))   # Weight matrix
        mu = self.mu                       # Penalty parameter
        
        # ADMM iterations
        for k in range(self.num_iterations + 1):
            # Update shrinkage threshold
            B = self.alpha * W / mu
            
            # Update T: Solve T-subproblem in frequency domain
            T_numerator = compute_t_numerator(initial_map, G, Z, mu)
            T_denominator = compute_t_denominator(height, width, mu)
            T_frequency = T_numerator / T_denominator
            T = np.real(np.fft.ifft2(np.fft.ifftshift(T_frequency)))
            
            # Compute gradient of T
            gradientT = compute_gradient(T)
            
            # Update G: Soft thresholding (shrinkage operator)
            # G = sign(x) * max(|x| - threshold, 0)
            x = gradientT + Z / mu
            G = np.sign(x) * np.maximum(np.abs(x) - B, np.zeros_like(B))
            
            # Update Z: Dual variable update
            Z = Z + mu * (gradientT - G)
            
            # Update mu: Increase penalty parameter
            mu = mu * self.rho
        
        refined_map = T
        return refined_map
    
    def _apply_illumination_map(self, image, illumination_map):
        """
        Applies illumination map to enhance image.
        
        Following Retinex model: R = I / T, where R is reflectance,
        I is observed image, and T is illumination.
        
        Args:
            image: Input image (M x N x 3)
            illumination_map: Illumination map (M x N)
        
        Returns:
            enhanced_image: Enhanced image (M x N x 3)
        """
        # Expand illumination_map to match image dimensions
        illumination_map_expanded = np.expand_dims(illumination_map, axis=2)
        enhanced_image = image / illumination_map_expanded
        return enhanced_image

