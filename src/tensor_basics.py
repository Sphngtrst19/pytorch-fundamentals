import torch

print("=" * 60)
print("PyTorch Tensor Fundamentals")
print("=" * 60)

# Scalar
scalar = torch.tensor(7)

print("\nScalar")
print(scalar)
print("Dimensions:", scalar.ndim)
print("Shape:", scalar.shape)

# Vector
vector = torch.tensor([1, 2, 3])

print("\nVector")
print(vector)
print("Dimensions:", vector.ndim)
print("Shape:", vector.shape)

# Matrix
matrix = torch.tensor([
    [1, 2],
    [3, 4]
])

print("\nMatrix")
print(matrix)
print("Dimensions:", matrix.ndim)
print("Shape:", matrix.shape)

# Three-dimensional tensor
tensor3d = torch.tensor([
    [
        [1, 2],
        [3, 4]
    ],
    [
        [5, 6],
        [7, 8]
    ]
])

print("\n3D Tensor")
print(tensor3d)
print("Dimensions:", tensor3d.ndim)
print("Shape:", tensor3d.shape)

print("\nData Types")
print("Scalar:", scalar.dtype)
print("Vector:", vector.dtype)

print("\nDevice")
print(scalar.device)

print("\n" + "=" * 60)
print("Floating Point Tensors")
print("=" * 60)

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

print(x)
print(y)

print("\nTensor Addition")
print(x + y)

print("\nTensor Subtraction")
print(x - y)

print("\nTensor Multiplication")
print(x * y)

print("\nTensor Division")
print(x / y)

print("\n" + "=" * 60)
print("Matrix Multiplication")
print("=" * 60)

A = torch.tensor([
    [1., 2.],
    [3., 4.]
])

B = torch.tensor([
    [5., 6.],
    [7., 8.]
])

print("Matrix A")
print(A)

print("Matrix B")
print(B)

print("\nA @ B")
print(A @ B)

print("\nTranspose")

print(A.T)

print("\nReshape")

v = torch.arange(12)

print(v)

matrix = v.reshape(3, 4)

print(matrix)

print("\nSqueeze")

a = torch.tensor([[5]])

print(a.shape)

a = a.squeeze()

print(a.shape)

print("\nUnsqueeze")

b = a.unsqueeze(0)

print(b.shape)

print("\nRandom Tensor")

random_tensor = torch.rand(3, 4)

print(random_tensor)

torch.manual_seed(42)

print("\nReproducible Random Tensor")

print(torch.rand(2, 3))

