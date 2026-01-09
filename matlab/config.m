function config = config()
    %CONFIG Creates a configuration structure for LIME image enhancement.
    %
    % Returns:
    %   config: struct containing all configuration parameters
    %
    % Configuration fields:
    %   - inputPath: Path to input image file
    %   - outputDir: Directory for saving output images
    %   - alpha: Weight parameter for structure preservation (default: 3)
    %   - gamma: Gamma correction parameter (default: 0.8)
    %   - rho: ADMM penalty parameter update rate (default: 1.15)
    %   - mu: Initial ADMM penalty parameter (default: 0.05)
    %   - numIterations: Number of ADMM iterations (default: 50)
    %   - displayResults: Whether to display results in figures (default: true)
    %   - saveResults: Whether to save output images (default: true)
    
    config = struct();
    
    % File paths (relative to project root)
    config.inputPath = '../images/cars.bmp';  % Go up from matlab/ to project root
    config.outputDir = '../output';            % Go up from matlab/ to project root
    
    % Algorithm parameters
    config.alpha = 3;          % Structure preservation weight
    config.gamma = 0.8;        % Gamma correction parameter
    config.rho = 1.15;         % ADMM penalty parameter update rate
    config.mu = 0.05;          % Initial ADMM penalty parameter
    config.numIterations = 50; % Number of ADMM iterations
    
    % Output options
    config.displayResults = true;
    config.saveResults = true;
    
end

