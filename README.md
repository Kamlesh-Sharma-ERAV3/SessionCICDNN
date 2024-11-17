# PyTorch MNIST CNN Pipeline

![ML Pipeline](https://github.com/Kamlesh-Sharma-ERAV3/SessionCICDNN/workflows/ML%20Pipeline/badge.svg)

A robust machine learning pipeline implementing a lightweight Convolutional Neural Network (CNN) for MNIST digit classification. The project includes automated training, testing, and CI/CD integration.

## Project Overview

This project demonstrates a complete ML pipeline with:
- Lightweight CNN architecture (<25K parameters)
- Automated training and testing
- CI/CD integration with GitHub Actions
- Model artifact storage
- Comprehensive testing suite

## Project Structure

├── train.py # Training script with model definition
├── test_model.py # Testing and validation scripts
├── requirements.txt # Project dependencies
├── .gitignore # Git ignore rules
├── README.md # Project documentation
└── .github/
└── workflows/
└── ml-pipeline.yml # CI/CD configuration


## Model Architecture

### Layer Structure
| Layer | Type | Input Shape | Output Shape | Parameters |
|-------|------|-------------|--------------|------------|
| conv1 | Conv2d | [1, 28, 28] | [4, 26, 26] | 40 |
| relu | ReLU | [4, 26, 26] | [4, 26, 26] | 0 |
| conv2 | Conv2d | [4, 26, 26] | [8, 24, 24] | 296 |
| relu | ReLU | [8, 24, 24] | [8, 24, 24] | 0 |
| pool1 | MaxPool2d | [8, 24, 24] | [8, 12, 12] | 0 |
| pool2 | MaxPool2d | [8, 12, 12] | [8, 6, 6] | 0 |
| fc1 | Linear | [8 * 6 * 6] | [20] | 5,780 |
| relu | ReLU | [20] | [20] | 0 |
| fc2 | Linear | [20] | [10] | 210 |

Total Parameters: 6,326

### Model Performance
- Training Accuracy: ~91-93%
- Test Accuracy: ~90-92%
- Training Time: ~45-60 seconds (1 epoch)
- Memory Usage: <50MB

## Key Features

1. **Efficient Architecture**
   - Lightweight design (<7K parameters)
   - Double MaxPooling for dimension reduction
   - Optimized layer sizes

2. **Automated Testing**
   - Model architecture validation
   - Parameter count verification
   - Accuracy benchmarking (>80% required)
   - Input/output shape validation

3. **CI/CD Pipeline**
   - Automated training on push
   - Comprehensive testing
   - Model artifact storage
   - Dependency caching

## Requirements

- Python 3.9+
- PyTorch
- torchvision
- tqdm
- pytest

## Local Setup

