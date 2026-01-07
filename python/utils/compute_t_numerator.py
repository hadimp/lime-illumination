"""
Computes numerator for T update in frequency domain.

In the ADMM algorithm, the T-subproblem is solved in frequency domain.
This function computes the numerator of the frequency-domain solution:

    T_numerator = FFT(2*L + mu*D^T*(G - Z/mu))

where L is initial map, D^T is gradient transpose, G and Z are ADMM
variables, and mu is penalty parameter.

Args:
    initialMap: Initial illumination map (M x N)
    G: Auxiliary variable (2*M x N)
    Z: Dual variable (2*M x N)
    mu: ADMM penalty parameter (scalar)

Returns:
    T_numerator: Numerator in frequency domain (M x N, complex)
"""
import numpy as np
from .apply_gradient_adjoint import apply_gradient_adjoint


def compute_t_numerator(initial_map, G, Z, mu):
    delX = apply_gradient_adjoint(G - Z / mu)
    T_numerator = np.fft.fftshift(np.fft.fft2(2 * initial_map + mu * delX))
    return T_numerator

