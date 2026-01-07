"""
Applies the adjoint (transpose) of the gradient operator to G.

Computes D^T * G, where D^T is the adjoint (transpose) of the gradient operator.
This is used in the ADMM update step for the T-subproblem.

Mathematical formulation:
    delG = Dx^T * Gx + Dy^T * Gy
    where G = [Gx; Gy] is split into horizontal and vertical components

Args:
    G: Gradient-like matrix (2*M x N)

Returns:
    delG: Result of D^T * G (M x N)
"""
import numpy as np
from .create_difference_matrix import create_difference_matrix


def apply_gradient_adjoint(G):
    p, n = G.shape
    m = p // 2
    
    # Split G into horizontal and vertical components
    # MATLAB reshape is column-major by default
    g = G.reshape(p * n, order='F')
    Gx = g[0:m * n].reshape((m, n), order='F')
    Gy = g[m * n:p * n].reshape((m, n), order='F')
    
    # Create difference matrices for adjoint operation
    Dyi = create_difference_matrix(m)
    Dy = -Dyi  # Negative sign for adjoint (transpose) of gradient operator
    Dxi = create_difference_matrix(n)
    Dx = Dxi[0:n, 0:n].copy()
    Dx[0:n, 0] = Dx[0:n, 0] + Dxi[0:n, n]
    
    # Prepare Gy for vertical gradient adjoint (with periodic boundary)
    # MATLAB: altGy(2:m+1, 1:n) = Gy; altGy(1, 2:n) = Gy(m, 1:n-1); altGy(1, 1) = Gy(m, n)
    altGy = np.zeros((m + 1, n))
    altGy[1:m + 1, 0:n] = Gy  # MATLAB rows 2:m+1 -> Python rows 1:m+1
    altGy[0, 1:n] = Gy[m - 1, 0:n - 1]  # MATLAB row 1, cols 2:n -> Python row 0, cols 1:n-1
    altGy[0, 0] = Gy[m - 1, n - 1]  # MATLAB (1,1) -> Python (0,0), MATLAB Gy(m,n) -> Python Gy(m-1,n-1)
    
    # Compute adjoint operations
    delGy = Dy @ altGy
    delGx = Gx @ Dx
    delG = delGx + delGy
    return delG

