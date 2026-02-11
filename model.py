"""
SageMaker Inference Script
Minimal implementation for MNIST digit classification
"""

import json
import numpy as np


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def conv2d(X, kernel, stride=1, padding=0):
    """Basic 2D convolution"""
    batch_size, h, w, in_channels = X.shape
    kh, kw, _, out_channels = kernel.shape
    
    if padding > 0:
        X = np.pad(X, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')
    
    out_h = (X.shape[1] - kh) // stride + 1
    out_w = (X.shape[2] - kw) // stride + 1
    output = np.zeros((batch_size, out_h, out_w, out_channels))
    
    for b in range(batch_size):
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                w_start = j * stride
                patch = X[b, h_start:h_start+kh, w_start:w_start+kw, :]
                for c in range(out_channels):
                    output[b, i, j, c] = np.sum(patch * kernel[:, :, :, c])
    
    return output


def max_pool2d(X, pool_size=2, stride=2):
    """Max pooling"""
    batch_size, h, w, channels = X.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    output = np.zeros((batch_size, out_h, out_w, channels))
    
    for b in range(batch_size):
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride
                w_start = j * stride
                patch = X[b, h_start:h_start+pool_size, w_start:w_start+pool_size, :]
                output[b, i, j, :] = np.max(patch, axis=(0, 1))
    
    return output


class SimpleCNN:
    """CNN for MNIST classification"""
    
    def __init__(self):
        self.conv1_kernel = np.random.randn(3, 3, 1, 16) * 0.1
        self.conv1_bias = np.zeros((1, 1, 1, 16))
        self.conv2_kernel = np.random.randn(3, 3, 16, 32) * 0.1
        self.conv2_bias = np.zeros((1, 1, 1, 32))
        self.fc1_w = np.random.randn(1568, 128) * 0.1
        self.fc1_b = np.zeros((1, 128))
        self.fc2_w = np.random.randn(128, 10) * 0.1
        self.fc2_b = np.zeros((1, 10))
    
    def forward(self, X):
        batch_size = X.shape[0]
        if len(X.shape) == 3:
            X = X.reshape(batch_size, 28, 28, 1)
        
        conv1 = relu(conv2d(X, self.conv1_kernel, stride=1, padding=1) + self.conv1_bias)
        pool1 = max_pool2d(conv1, pool_size=2, stride=2)
        conv2 = relu(conv2d(pool1, self.conv2_kernel, stride=1, padding=1) + self.conv2_bias)
        pool2 = max_pool2d(conv2, pool_size=2, stride=2)
        flat = pool2.reshape(batch_size, -1)
        fc1 = relu(flat @ self.fc1_w + self.fc1_b)
        output = softmax(fc1 @ self.fc2_w + self.fc2_b)
        
        return output
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)


# SageMaker functions
def model_fn(model_dir):
    """Load model"""
    print(f"Loading model from {model_dir}")
    model = SimpleCNN()
    print("Model initialized")
    return model


def input_fn(request_body, content_type="application/json"):
    """Process input"""
    if content_type == "application/json":
        data = json.loads(request_body)
        inputs = np.array(data.get("inputs", data), dtype=np.float32)
        
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, 28, 28)
        elif len(inputs.shape) == 2 and inputs.shape[1] == 784:
            inputs = inputs.reshape(-1, 28, 28)
        
        if inputs.max() > 1.0:
            inputs = inputs / 255.0
        
        return inputs
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    """Make predictions"""
    return model.predict(input_data)


def output_fn(predictions, accept="application/json"):
    """Format output"""
    if accept == "application/json":
        return json.dumps({"predictions": predictions.tolist()}), accept
    raise ValueError(f"Unsupported accept type: {accept}")


if __name__ == "__main__":
    print("SageMaker inference script ready")
    print("Usage: model.py is loaded by SageMaker endpoint")
    print("Input: JSON with 'inputs' field containing 28x28 image arrays")
    print("Output: JSON with 'predictions' field containing digit classes")
