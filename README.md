# Exploring Convolutional Layers Through Data and Experiments

Implementation of convolutional neural networks from scratch using NumPy to classify MNIST handwritten digits. This project explores architectural design choices and compares CNN performance against a baseline fully-connected network.

## Problem Description

Classify handwritten digits (0-9) from the MNIST dataset using neural networks built from scratch. The focus is understanding why convolutional layers work better than fully-connected layers for image data by examining inductive biases, parameter efficiency, and architectural trade-offs.

## Dataset Description

MNIST contains 60,000 training and 10,000 test grayscale images of handwritten digits. Each image is 28x28 pixels with values from 0-255, normalized to [0,1]. The dataset is balanced across all 10 digit classes. Preprocessing includes normalization, one-hot encoding, and a 90/10 train-validation split.

## Architecture

### Baseline Model
Fully-connected network: Input (784) → Dense (128) → Dense (64) → Output (10). Total parameters: ~109K. Flattens images, losing spatial structure and translation invariance.

### CNN Model
Convolutional architecture preserving spatial structure:
- Conv1: 16 filters, 3x3 kernel → ReLU → MaxPool (2x2)
- Conv2: 32 filters, 3x3 kernel → ReLU → MaxPool (2x2)
- Flatten → Dense (128) → Output (10)
- Total parameters: ~210K

**Architecture diagram:**

```
Input [28×28×1] 
    ↓
Conv1 [3×3, 16 filters] → [28×28×16] → ReLU → MaxPool → [14×14×16]
    ↓
Conv2 [3×3, 32 filters] → [14×14×32] → ReLU → MaxPool → [7×7×32]
    ↓
Flatten [1568] → Dense [128] → Output [10]
```

**Design choices:** 3x3 kernels balance efficiency and performance. Progressive filter increase (16→32) builds hierarchical features. Max pooling provides translation invariance.

## Experimental Results

**Baseline vs CNN:** CNN achieves 96-98% accuracy vs baseline 92-94%, despite both having similar parameter counts. CNN preserves spatial structure and provides translation invariance, making it superior for image data.

**Kernel size experiments (3x3, 5x5, 7x7):** Parameter count scales quadratically with kernel size. 3x3 kernels (~210K params) provide sharp edge detection with low overfitting. 5x5 kernels (~690K params) capture broader patterns but increase computation. 7x7 kernels (~1.5M params) capture global context but risk overfitting. Result: 3x3 kernels are optimal for MNIST.

**Key insight:** Stacking multiple 3x3 layers achieves larger receptive fields with fewer parameters and more non-linearity than single large kernels.

## Interpretation

**Why CNNs work:** Convolutional layers introduce inductive biases suited for spatial data - locality (nearby pixels are related), translation equivariance (patterns detected regardless of position), and hierarchical composition (simple features combine into complex ones). Baseline networks lack these properties.

**When NOT to use convolution:** Tabular data (customer records, financial data), irregular time series, graph-structured data (social networks), or any domain where spatial locality doesn't apply. Use CNNs for images, video, and regular grids. Use other architectures for arbitrary relationships.

## Getting Started

### Prerequisites

Python 3.8+ with NumPy, Pandas, and Matplotlib:

```bash
pip install numpy pandas matplotlib
```

### Installation

1. Clone the repository
2. Create and activate a virtual environment
3. Install dependencies: `pip install numpy pandas matplotlib`
4. Verify MNIST data exists in `archive/` directory

### Running

Open `ConvolutionalLayers.ipynb` in Jupyter or VS Code and run cells sequentially to train models and view experiments.

## Deployment

AWS SageMaker deployment was prepared but endpoint creation was not completed due to IAM role permission restrictions (lab environment limitations). The inference code in `model.py` demonstrates the intended SageMaker integration pattern.

![Baseline Model in SageMaker](images/baseline%20model%20in%20sagemaker.png)

![Different Kernel Tests](images/Different%20kernel%20test.png)

## Built With

NumPy, Pandas, Matplotlib, AWS SageMaker (attempted deployment)

## Author

SantiagoHM20
