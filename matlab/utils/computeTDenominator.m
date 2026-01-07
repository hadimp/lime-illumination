function T_denominator = computeTDenominator(height, width, mu)
    %COMPUTETDENOMINATOR Computes denominator for T update in frequency domain
    %
    % In the ADMM algorithm, the T-subproblem is solved in frequency domain.
    % This function computes the denominator of the frequency-domain solution:
    %
    %   T_denominator = 2 + mu*(|Dx|^2 + |Dy|^2)
    %
    % where Dx and Dy are frequency-domain representations of gradient
    % operators, and mu is the penalty parameter.
    %
    % Args:
    %   height: Image height (M)
    %   width: Image width (N)
    %   mu: ADMM penalty parameter (scalar)
    %
    % Returns:
    %   T_denominator: Denominator in frequency domain (M x N, complex)
    
    % Create unit impulse responses for gradient operators
    dxe = zeros(height, width);
    dye = zeros(height, width);
    
    % Horizontal difference operator impulse response
    dxe(2, 2) = -1;
    dxe(2, 3) = 1;
    dx_fft = fftshift(fft2(dxe));
    dx = conj(dx_fft) .* dx_fft;  % |Dx|^2
    
    % Vertical difference operator impulse response
    dye(2, 2) = -1;
    dye(3, 2) = 1;
    dy_fft = fftshift(fft2(dye));
    dy = conj(dy_fft) .* dy_fft;  % |Dy|^2
    
    % Compute denominator
    T_denominator = 2 + mu * (dx + dy);
end

