classdef LIME < handle
    %LIME Low-Light Image Enhancement via Illumination Map Estimation
    %
    % This class implements the LIME algorithm for enhancing low-light images
    % using the Retinex model and ADMM optimization.
    %
    % Algorithm Overview:
    %   1. Estimate initial illumination map from input image
    %   2. Refine illumination map using ADMM optimization
    %   3. Apply gamma correction to refined map
    %   4. Enhance image by dividing by corrected illumination map
    %
    % Reference:
    %   X. Guo, Y. Li, and H. Ling, "LIME: Low-Light Image Enhancement via
    %   Illumination Map Estimation," IEEE Transactions on Image Processing,
    %   vol. 26, pp. 982-993, 2017.
    %
    % Example:
    %   enhancer = LIME();
    %   enhanced = enhancer.enhance(inputImage);
    
    properties (Access = private)
        alpha           % Structure preservation weight
        gamma           % Gamma correction parameter
        rho             % ADMM penalty parameter update rate
        mu              % Initial ADMM penalty parameter
        numIterations   % Number of ADMM iterations
    end
    
    methods
        function obj = LIME(varargin)
            %LIME Constructor for LIME image enhancer
            %
            % Args:
            %   alpha (double, optional): Structure preservation weight (default: 3)
            %   gamma (double, optional): Gamma correction parameter (default: 0.8)
            %   rho (double, optional): ADMM penalty update rate (default: 1.15)
            %   mu (double, optional): Initial ADMM penalty (default: 0.05)
            %   numIterations (int, optional): ADMM iterations (default: 50)
            %
            % Example:
            %   enhancer = LIME('alpha', 3, 'gamma', 0.8);
            
            % Add utils directory to path for utility functions
            classDir = fileparts(mfilename('fullpath'));
            matlabDir = fileparts(classDir);  % Go up from @LIME to matlab directory
            utilsPath = fullfile(matlabDir, 'utils');
            if exist(utilsPath, 'dir')
                addpath(utilsPath);
            end
            
            p = inputParser;
            addParameter(p, 'alpha', 3, @(x) isnumeric(x) && x > 0);
            addParameter(p, 'gamma', 0.8, @(x) isnumeric(x) && x > 0 && x <= 1);
            addParameter(p, 'rho', 1.15, @(x) isnumeric(x) && x > 1);
            addParameter(p, 'mu', 0.05, @(x) isnumeric(x) && x > 0);
            addParameter(p, 'numIterations', 50, @(x) isnumeric(x) && x > 0);
            parse(p, varargin{:});
            
            obj.alpha = p.Results.alpha;
            obj.gamma = p.Results.gamma;
            obj.rho = p.Results.rho;
            obj.mu = p.Results.mu;
            obj.numIterations = p.Results.numIterations;
        end
        
        function [enhancedImage, results] = enhance(obj, inputImage)
            %ENHANCE Enhances a low-light image using the LIME algorithm
            %
            % Args:
            %   inputImage: Input image (uint8, uint16, or double in [0,1])
            %
            % Returns:
            %   enhancedImage: Final enhanced image (double in [0,1])
            %   results: struct containing intermediate results:
            %       - initialMap: Initial illumination map
            %       - refinedMap: Refined illumination map
            %       - gammaCorrectedMap: Gamma-corrected illumination map
            %       - initialEnhanced: Image enhanced with initial map
            %       - refinedEnhanced: Image enhanced with refined map
            
            % Convert to double if needed
            if ~isa(inputImage, 'double')
                inputImage = im2double(inputImage);
            end
            
            % Step 1: Calculate initial illumination map
            initialMap = obj.calculateInitialIlluminationMap(inputImage);
            
            % Step 2: Apply initial map
            initialEnhanced = obj.applyIlluminationMap(inputImage, initialMap);
            
            % Step 3: Refine illumination map using ADMM
            refinedMap = obj.refineIlluminationMap(initialMap);
            refinedMap = abs(refinedMap);  % Ensure non-negative
            
            % Step 4: Apply refined map
            refinedEnhanced = obj.applyIlluminationMap(inputImage, refinedMap);
            
            % Step 5: Apply gamma correction
            gammaCorrectedMap = refinedMap .^ obj.gamma;
            
            % Step 6: Final enhanced image
            enhancedImage = obj.applyIlluminationMap(inputImage, gammaCorrectedMap);
            
            % Store results
            results.initialMap = initialMap;
            results.refinedMap = refinedMap;
            results.gammaCorrectedMap = gammaCorrectedMap;
            results.initialEnhanced = initialEnhanced;
            results.refinedEnhanced = refinedEnhanced;
        end
    end
    
    methods (Access = private)
        function initialMap = calculateInitialIlluminationMap(obj, image)
            %CALCULATEINITIALILLUMINATIONMAP Estimates initial illumination map
            %
            % The initial map is computed as the maximum intensity across RGB
            % channels at each pixel, following the Retinex model assumption
            % that illumination is the maximum channel value.
            %
            % Args:
            %   image: Input image (M x N x 3)
            %
            % Returns:
            %   initialMap: Initial illumination map (M x N)
            
            [height, width, ~] = size(image);
            initialMap = zeros(height, width);
            
            for i = 1:height
                for j = 1:width
                    % Find channel with highest intensity (Retinex assumption)
                    initialMap(i, j) = max([image(i, j, 1), ...
                                            image(i, j, 2), ...
                                            image(i, j, 3)]);
                    
                    % Avoid division by zero
                    if initialMap(i, j) == 0
                        initialMap(i, j) = 1e-7;
                    end
                end
            end
        end
        
        function refinedMap = refineIlluminationMap(obj, initialMap)
            %REFINEILLUMINATIONMAP Refines illumination map using ADMM
            %
            % Solves the optimization problem:
            %   min_T ||T - L||_2^2 + alpha*||DT||_1
            %
            % where T is the refined map, L is initial map, D is gradient operator,
            % and alpha controls structure preservation. Uses ADMM to solve.
            %
            % Args:
            %   initialMap: Initial illumination map (M x N)
            %
            % Returns:
            %   refinedMap: Refined illumination map (M x N)
            
            [height, width] = size(initialMap);
            
            % Initialize ADMM variables
            T = zeros(height, width);
            Z = zeros(2 * height, width);  % Dual variable for gradient
            G = zeros(2 * height, width);  % Auxiliary variable
            W = ones(2 * height, width);   % Weight matrix
            mu = obj.mu;                   % Penalty parameter
            
            % ADMM iterations
            for k = 0:obj.numIterations
                % Update shrinkage threshold
                B = obj.alpha * W / mu;
                
                % Update T: Solve T-subproblem in frequency domain
                T_numerator = computeTNumerator(initialMap, G, Z, mu);
                T_denominator = computeTDenominator(height, width, mu);
                T_frequency = T_numerator ./ T_denominator;
                T = ifft2(ifftshift(T_frequency));
                
                % Compute gradient of T
                gradientT = computeGradient(T);
                
                % Update G: Soft thresholding (shrinkage operator)
                % G = sign(x) * max(|x| - threshold, 0)
                G = sign(gradientT + Z / mu) .* ...
                    max(abs(gradientT + Z / mu) - B, zeros(size(B)));
                
                % Update Z: Dual variable update
                Z = Z + mu * (gradientT - G);
                
                % Update mu: Increase penalty parameter
                mu = mu * obj.rho;
            end
            
            refinedMap = T;
        end
        
        function enhancedImage = applyIlluminationMap(obj, image, illuminationMap)
            %APPLYILLUMINATIONMAP Applies illumination map to enhance image
            %
            % Following Retinex model: R = I / T, where R is reflectance,
            % I is observed image, and T is illumination.
            %
            % Args:
            %   image: Input image (M x N x 3)
            %   illuminationMap: Illumination map (M x N)
            %
            % Returns:
            %   enhancedImage: Enhanced image (M x N x 3)
            
            enhancedImage = image ./ illuminationMap;
        end
    end
end

