function delG = applyGradientAdjoint(G)
    %APPLYGRADIENTADJOINT Applies the adjoint (transpose) of the gradient operator to G
    %
    % Computes D^T * G, where D^T is the adjoint (transpose) of the gradient operator.
    % This is used in the ADMM update step for the T-subproblem.
    %
    % Mathematical formulation:
    %   delG = Dx^T * Gx + Dy^T * Gy
    %   where G = [Gx; Gy] is split into horizontal and vertical components
    %
    % Args:
    %   G: Gradient-like matrix (2*M x N)
    %
    % Returns:
    %   delG: Result of D^T * G (M x N)
    %
    % Note: This function is based on code from:
    %   https://github.com/estija/LIME/blob/master/multiplydtrans.m
    
    [p, n] = size(G);
    m = floor(p / 2);
    
    % Split G into horizontal and vertical components
    g = reshape(G, [p * n, 1]);
    Gx = reshape(g(1:m * n), [m, n]);
    Gy = reshape(g(m * n + 1:p * n), [m, n]);
    
    % Create difference matrices for adjoint operation
    Dyi = createDifferenceMatrix(m);
    Dy = -Dyi;  % Negative sign for adjoint (transpose) of gradient operator
    Dxi = createDifferenceMatrix(n);
    Dx = Dxi(1:n, 1:n);
    Dx(1:n, 1) = Dx(1:n, 1) + Dxi(1:n, n + 1);
    
    % Prepare Gy for vertical gradient adjoint (with periodic boundary)
    altGy = zeros(m + 1, n);
    altGy(2:m + 1, 1:n) = Gy;
    altGy(1, 2:n) = Gy(m, 1:n - 1);
    altGy(1, 1) = Gy(m, n);
    
    % Compute adjoint operations
    delGy = Dy * altGy;
    delGx = Gx * Dx;
    delG = delGx + delGy;
end

