import torch

from simple_neural_network import SimpleNeuralNetwork


def main():
    torch.manual_seed(42)

    model = SimpleNeuralNetwork()

    print("=" * 60)
    print("PyTorch Neural Network Fundamentals")
    print("=" * 60)

    print("\nModel Architecture\n")
    print(model)

    print("\n" + "=" * 60)
    print("Forward Pass")
    print("=" * 60)

    x = torch.rand(5, 4)
    output = model(x)

    print("\nInput Shape")
    print(x.shape)

    print("\nOutput Shape")
    print(output.shape)

    print("\nOutput")
    print(output)

    print("\n" + "=" * 60)
    print("Model Parameters")
    print("=" * 60)

    total_parameters = 0

    for name, parameter in model.named_parameters():
        print(f"\n{name}")
        print(parameter.shape)

        total_parameters += parameter.numel()

    print(f"\nTotal Trainable Parameters: {total_parameters}")


if __name__ == "__main__":
    main()