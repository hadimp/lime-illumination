"""
Computes denominator for T update in frequency domain.

In the ADMM algorithm, the T-subproblem is solved in frequency domain.
This function computes the denominator of the frequency-domain solution:

    T_denominator = 2 + mu*(|Dx|^2 + |Dy|^2)

where Dx and Dy are frequency-domain representations of gradient
operators, and mu is the penalty parameter.

Args:
    height: Image height (M)
    width: Image width (N)
    mu: ADMM penalty parameter (scalar)

Returns:
    T_denominator: Denominator in frequency domain (M x N, complex)
"""
import numpy as np


def compute_t_denominator(height, width, mu):
    # Create unit impulse responses for gradient operators
    dxe = np.zeros((height, width))
    dye = np.zeros((height, width))
    
    # Horizontal difference operator impulse response
    dxe[1, 1] = -1
    dxe[1, 2] = 1
    dx_fft = np.fft.fftshift(np.fft.fft2(dxe))
    dx = np.conj(dx_fft) * dx_fft  # |Dx|^2
    
    # Vertical difference operator impulse response
    dye[1, 1] = -1
    dye[2, 1] = 1
    dy_fft = np.fft.fftshift(np.fft.fft2(dye))
    dy = np.conj(dy_fft) * dy_fft  # |Dy|^2
    
    # Compute denominator
    T_denominator = 2 + mu * (dx + dy)
    return T_denominator

