"""
Computes gradient of matrix T using difference operator.

Computes the gradient operator D applied to T, where D consists of
horizontal and vertical difference operators. The result is a
concatenated vector of horizontal and vertical gradients.

Mathematical formulation:
    gradientT = [Dx*T; Dy*T]
    where Dx and Dy are horizontal and vertical difference matrices

Args:
    T: Input matrix (M x N)

Returns:
    gradientT: Gradient matrix (2*M x N), first M rows are horizontal
               gradient, last M rows are vertical gradient
"""
import numpy as np
from .create_difference_matrix import create_difference_matrix


def compute_gradient(T):
    height, width = T.shape
    
    # Create difference matrices
    Dy = create_difference_matrix(height)
    Dx = create_difference_matrix(width).T
    
    # Prepare T for vertical gradient computation (with periodic boundary)
    altTy = np.zeros((height + 1, width))
    altTy[0:height, 0:width] = T
    altTy[height, 0:width - 1] = T[0, 1:width]
    altTy[height, width - 1] = T[0, 0]  # Note: MATLAB uses width (1-indexed), Python uses width-1 (0-indexed)
    
    # Compute vertical gradient
    delTy = Dy @ altTy
    
    # Prepare T for horizontal gradient computation (with periodic boundary)
    altTx = np.zeros((height, width + 1))
    altTx[0:height, 0:width] = T
    altTx[0:height, width] = T[0:height, 0]
    
    # Compute horizontal gradient
    delTx = altTx @ Dx
    
    # Concatenate gradients: [horizontal; vertical]
    # MATLAB reshape is column-major by default
    dtx = delTx.reshape(height * width, order='F')
    dty = delTy.reshape(height * width, order='F')
    dt = np.concatenate([dtx, dty])
    gradientT = dt.reshape((2 * height, width), order='F')
    return gradientT

