import torch

print("=" * 60)
print("PyTorch Automatic Differentiation")
print("=" * 60)

# Create a tensor that tracks gradients
x = torch.tensor(3.0, requires_grad=True)

print("\nInput Tensor")
print(x)

# Define a simple function
y = x ** 2

print("\nFunction")
print("y = x²")
print(y)

# Compute the gradient
y.backward()

print("\nGradient")
print("dy/dx =", x.grad)

print("\n" + "=" * 60)
print("Multiple Operations")
print("=" * 60)

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(5.0, requires_grad=True)

z = a * b + b ** 2

print("\nExpression")
print("z = a*b + b²")
print(z)

z.backward()

print("\nGradients")

print("dz/da =", a.grad)
print("dz/db =", b.grad)