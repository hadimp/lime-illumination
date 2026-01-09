function main()
    %MAIN Main script for low-light image enhancement using LIME algorithm
    %
    % This script demonstrates the usage of the LIME class for enhancing
    % low-light images. It loads configuration, processes the image, and
    % optionally displays and saves results.
    %
    % Example:
    %   main()
    
    % Load configuration
    cfg = config();
    
    % Create output directory if it doesn't exist
    if cfg.saveResults && ~exist(cfg.outputDir, 'dir')
        mkdir(cfg.outputDir);
    end
    
    % Load input image
    if ~exist(cfg.inputPath, 'file')
        error('Input image not found: %s', cfg.inputPath);
    end
    inputImage = imread(cfg.inputPath);
    
    % Create LIME enhancer with configured parameters
    enhancer = LIME('alpha', cfg.alpha, ...
                    'gamma', cfg.gamma, ...
                    'rho', cfg.rho, ...
                    'mu', cfg.mu, ...
                    'numIterations', cfg.numIterations);
    
    % Enhance image
    fprintf('Enhancing image: %s\n', cfg.inputPath);
    [enhancedImage, results] = enhancer.enhance(inputImage);
    
    % Display results if requested
    if cfg.displayResults
        displayResults(inputImage, enhancedImage, results);
    end
    
    % Save results if requested
    if cfg.saveResults
        saveResults(cfg, inputImage, enhancedImage, results);
    end
    
    fprintf('Enhancement complete!\n');
end

function displayResults(inputImage, enhancedImage, results)
    %DISPLAYRESULTS Displays input and enhanced images in separate figures
    
    figure('Name', 'Original Image', 'NumberTitle', 'off');
    imshow(inputImage);
    title('Original Low-Light Image');
    
    figure('Name', 'Initial Enhancement', 'NumberTitle', 'off');
    imshow(results.initialEnhanced);
    title('Enhanced with Initial Illumination Map');
    
    figure('Name', 'Refined Enhancement', 'NumberTitle', 'off');
    imshow(results.refinedEnhanced, []);
    title('Enhanced with Refined Illumination Map');
    
    figure('Name', 'Final Enhanced Image', 'NumberTitle', 'off');
    imshow(enhancedImage, []);
    title('Final Enhanced Image (with Gamma Correction)');
end

function saveResults(cfg, inputImage, enhancedImage, results)
    %SAVERESULTS Saves enhanced images to output directory
    
    [~, baseName, ~] = fileparts(cfg.inputPath);
    
    % Save initial enhanced image
    initialPath = fullfile(cfg.outputDir, [baseName, '_initial.png']);
    imwrite(results.initialEnhanced, initialPath);
    fprintf('Saved: %s\n', initialPath);
    
    % Save refined enhanced image
    refinedPath = fullfile(cfg.outputDir, [baseName, '_refined.png']);
    imwrite(results.refinedEnhanced, refinedPath);
    fprintf('Saved: %s\n', refinedPath);
    
    % Save final enhanced image
    finalPath = fullfile(cfg.outputDir, [baseName, '_enhanced.png']);
    imwrite(enhancedImage, finalPath);
    fprintf('Saved: %s\n', finalPath);
end

