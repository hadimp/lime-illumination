function D = createDifferenceMatrix(m)
    %CREATEDIFFERENCEMATRIX Creates a forward difference matrix
    %
    % Creates an M x (M+1) matrix representing the forward difference
    % operator. The matrix has -1 on the diagonal and +1 on the superdiagonal,
    % implementing a first-order forward difference.
    %
    % Mathematical formulation:
    %   D[i, i] = -1
    %   D[i, i+1] = +1
    %   D[i, j] = 0 otherwise
    %
    % Args:
    %   m: Size parameter (creates m x (m+1) matrix)
    %
    % Returns:
    %   D: Difference matrix (m x (m+1))
    %
    % Note: This function is based on code from:
    %   https://github.com/estija/LIME/blob/master/maked_alt.m
    
    D = zeros(m, m + 1);
    for i = 1:m
        D(i, i) = -1;
        D(i, i + 1) = 1;
    end
end

