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