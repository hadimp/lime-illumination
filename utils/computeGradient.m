function gradientT = computeGradient(T)
    %COMPUTEGRADIENT Computes gradient of matrix T using difference operator
    %
    % Computes the gradient operator D applied to T, where D consists of
    % horizontal and vertical difference operators. The result is a
    % concatenated vector of horizontal and vertical gradients.
    %
    % Mathematical formulation:
    %   gradientT = [Dx*T; Dy*T]
    %   where Dx and Dy are horizontal and vertical difference matrices
    %
    % Args:
    %   T: Input matrix (M x N)
    %
    % Returns:
    %   gradientT: Gradient matrix (2*M x N), first M rows are horizontal
    %              gradient, last M rows are vertical gradient
    %
    % Note: This function is based on code from:
    %   https://github.com/estija/LIME/blob/master/multiplyd.m
    
    [height, width] = size(T);
    
    % Create difference matrices
    Dy = createDifferenceMatrix(height);
    Dx = transpose(createDifferenceMatrix(width));
    
    % Prepare T for vertical gradient computation (with periodic boundary)
    altTy = zeros(height + 1, width);
    altTy(1:height, 1:width) = T;
    altTy(height + 1, 1:width - 1) = T(1, 2:width);
    altTy(height + 1, width) = T(1, 1);
    
    % Compute vertical gradient
    delTy = Dy * altTy;
    
    % Prepare T for horizontal gradient computation (with periodic boundary)
    altTx = zeros(height, width + 1);
    altTx(1:height, 1:width) = T;
    altTx(1:height, width + 1) = T(1:height, 1);
    
    % Compute horizontal gradient
    delTx = altTx * Dx;
    
    % Concatenate gradients: [horizontal; vertical]
    dtx = reshape(delTx, [height * width, 1]);
    dty = reshape(delTy, [height * width, 1]);
    dt = [dtx; dty];
    gradientT = reshape(dt, [2 * height, width]);
end

