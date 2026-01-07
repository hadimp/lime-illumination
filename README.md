# LIME: Illumination Map Estimation

A modern, refactored and minimal implementation of the "LIME" (Low-Light Image Enhancement via Illumination Map Estimation) algorithm for improving the visibility of low-light images.

## Overview

Images taken in low-light conditions suffer from low visibility. This problem degrades the visual aesthetics of those images and damages the performance of machine vision methods that use those images. This project implements the LIME algorithm to recover hidden information buried in darker regions of captured images by estimating and refining an illumination map using the Retinex model and ADMM optimization.

![Multiple Examples Comparison](comparison_1.png)

## Results

### Processing Pipeline

The LIME algorithm processes images through multiple stages to achieve optimal enhancement:

![Processing Pipeline](comparison_2.png)

## Algorithm

The LIME algorithm works in several steps:

1. **Initial Illumination Map Estimation**: Estimates an initial illumination map as the maximum intensity across RGB channels at each pixel (following the Retinex model).

2. **Illumination Map Refinement**: Refines the initial map using ADMM (Alternating Direction Method of Multipliers) optimization to solve:
   ```
   min_T ||T - L||_2^2 + alpha*||DT||_1
   ```
   where T is the refined map, L is the initial map, D is the gradient operator, and alpha controls structure preservation.

3. **Gamma Correction**: Applies gamma correction to the refined illumination map for better visual results.

4. **Image Enhancement**: Enhances the image by dividing each channel by the corrected illumination map.

## Project Structure

```
lime-illumination/
├── @LIME/                    # LIME class directory
│   └── LIME.m               # Main LIME class implementation
├── utils/                    # Utility functions
│   ├── computeGradient.m
│   ├── applyGradientAdjoint.m
│   ├── computeTDenominator.m
│   ├── computeTNumerator.m
│   └── createDifferenceMatrix.m
├── config.m                  # Configuration function
├── main.m                    # Main script
└── README.md
```

## Usage

### Quick Start

1. **Configure parameters** by editing `config.m`:
   ```matlab
   config.inputPath = 'your_image.bmp';  % Input image path
   config.outputDir = './output';         % Output directory
   config.alpha = 3;                      % Structure preservation weight
   config.gamma = 0.8;                    % Gamma correction parameter
   config.numIterations = 50;             % ADMM iterations
   ```

2. **Run the main script**:
   ```matlab
   main()
   ```

### Using the LIME Class Directly

For more control, you can use the LIME class directly:

```matlab
% Create enhancer with custom parameters
enhancer = LIME('alpha', 3, 'gamma', 0.8, 'numIterations', 50);

% Load and enhance image
inputImage = imread('your_image.bmp');
[enhancedImage, results] = enhancer.enhance(inputImage);

% Display results
imshow(enhancedImage);
```

### Configuration Parameters

- `alpha` (default: 3): Weight parameter for structure preservation. Higher values preserve more structure but may over-smooth.
- `gamma` (default: 0.8): Gamma correction parameter. Values < 1 brighten the image.
- `rho` (default: 1.15): ADMM penalty parameter update rate. Controls convergence speed.
- `mu` (default: 0.05): Initial ADMM penalty parameter.
- `numIterations` (default: 50): Number of ADMM iterations. More iterations improve quality but increase computation time.

## References

[1] Xiaojie Guo, Yu Li, and Haibin Ling, "LIME: Low-Light Image Enhancement via Illumination Map Estimation," IEEE Transactions on Image Processing, vol. 26, pp. 982-993, 2017.

[2] S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein, "Distributed optimization and statistical learning via the alternating direction method of multipliers," Foundations and Trends in Machine Learning, vol. 3, no. 1, pp. 1–122, 2011. [Online]. Available: http://dx.doi.org/10.1561/2200000016

## Features

- **Object-Oriented Design**: Clean OOP structure with the `LIME` class
- **Modular Code**: Well-organized functions with clear responsibilities
- **Comprehensive Documentation**: Google-style docstrings explaining the mathematics
- **Configurable**: Easy-to-use configuration system for parameters and file paths
- **Modern MATLAB**: Uses modern MATLAB conventions and best practices
- **Batch Processing**: Process multiple images automatically
- **Octave Compatible**: Works with both MATLAB and Octave

