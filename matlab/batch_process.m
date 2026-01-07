function batch_process()
    %BATCH_PROCESS Processes all images in the images folder
    %
    % This script processes all .bmp images in the images directory
    % using the LIME enhancement algorithm.
    %
    % Example:
    %   batch_process()
    
    % Get script directory
    scriptDir = fileparts(mfilename('fullpath'));
    
    % Load configuration
    cfg = config();
    
    % Create output directory if it doesn't exist
    if cfg.saveResults && ~exist(cfg.outputDir, 'dir')
        mkdir(cfg.outputDir);
    end
    
    % Get all .bmp files in images directory
    imagesDir = fullfile(scriptDir, 'images');
    imageFiles = dir(fullfile(imagesDir, '*.bmp'));
    
    if isempty(imageFiles)
        fprintf('No .bmp images found in %s\n', imagesDir);
        return;
    end
    
    fprintf('Found %d images to process\n\n', length(imageFiles));
    
    % Create LIME enhancer with configured parameters
    enhancer = LIME('alpha', cfg.alpha, ...
                    'gamma', cfg.gamma, ...
                    'rho', cfg.rho, ...
                    'mu', cfg.mu, ...
                    'numIterations', cfg.numIterations);
    
    % Process each image
    for i = 1:length(imageFiles)
        imagePath = fullfile(imagesDir, imageFiles(i).name);
        fprintf('[%d/%d] Processing: %s\n', i, length(imageFiles), imageFiles(i).name);
        
        try
            % Load input image
            inputImage = imread(imagePath);
            
            % Enhance image
            [enhancedImage, results] = enhancer.enhance(inputImage);
            
            % Save results if requested
            if cfg.saveResults
                [~, baseName, ~] = fileparts(imageFiles(i).name);
                
                % Save initial enhanced image
                initialPath = fullfile(cfg.outputDir, [baseName, '_initial.png']);
                imwrite(results.initialEnhanced, initialPath);
                
                % Save refined enhanced image
                refinedPath = fullfile(cfg.outputDir, [baseName, '_refined.png']);
                imwrite(results.refinedEnhanced, refinedPath);
                
                % Save final enhanced image
                finalPath = fullfile(cfg.outputDir, [baseName, '_enhanced.png']);
                imwrite(enhancedImage, finalPath);
                
                fprintf('  ✓ Saved: %s_*.png\n', baseName);
            end
            
        catch ME
            fprintf('  ✗ Error processing %s: %s\n', imageFiles(i).name, ME.message);
        end
        
        fprintf('\n');
    end
    
    fprintf('Batch processing complete! Processed %d images.\n', length(imageFiles));
end

