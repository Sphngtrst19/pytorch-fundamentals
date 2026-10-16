# PyTorch Fundamentals

This project introduces core PyTorch concepts including tensors, tensor operations, automatic differentiation, and neural network construction.

## Project Goals

- Learn PyTorch tensor operations
- Understand autograd and gradients
- Build simple neural networks
- Prepare for deep learning model development

## Project Structure

```text
pytorch-fundamentals/
├── notebooks/
├── reports/
├── results/
├── src/
│   ├── tensor_basics.py
│   ├── autograd_demo.py
│   └── simple_neural_network.py
├── README.md
└── requirements.txt
```

## Fully Connected vs CNN Comparison

The fully connected neural network achieved 97.47% test accuracy on MNIST, while the convolutional neural network achieved 98.92%.

The CNN improved accuracy by approximately 1.45 percentage points, demonstrating the benefit of preserving spatial image structure through convolutional layers.

Although the CNN contained more trainable parameters, its improved generalization performance supports the expectation that convolutional architectures are better suited to image classification tasks than fully connected networks.

| Model | Test Accuracy | Trainable Parameters |
|---|---:|---:|
| Fully Connected NN | 97.47% | 109,386 |
| CNN | **98.92%** | 206,922 |

