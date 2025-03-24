# %% 
import torch
x = torch.randn(2, 3)

# %% Permutation
permuted = torch.einsum('ij->ji', x)
print(permuted)

# %% Summation
summed = torch.einsum('ij->', x)
print(summed)

# %% Column sum
col_sum = torch.einsum('ij->j', x)
print(col_sum)

# %% Row sum
row_sum = torch.einsum('ij->i', x)
print(row_sum)

# %% Matrix-vector Multiplication
v = torch.randn((1,3))
mat_vec = torch.einsum('ij,kj->ik', x, v)
print(mat_vec)

# %% Matrix-Matrix Multiplication
mat_mat = torch.einsum('ij,kj->ik', x, x)
print(mat_mat)

# %% Dot product first row with first row of matrix
dot = torch.einsum('i,i->', x[0], x[0])
print(dot)

# %% Dot product with matrix
dot_m = torch.einsum('ij,ij->', x, x)
print(dot_m)

# %% Hadamard product
hadamard = torch.einsum('ij,ij->ij', x, x)
print(hadamard)

# %% Outer product
outer = torch.einsum('i,j->ij', x[0], x[1])
print(outer)

# %% Batch matrix multiplication
a = torch.randn((3,2,5))
b = torch.randn((3,5,3))
batch_matmul = torch.einsum('ijk,ikl->ijl', a, b)
print(batch_matmul)

# %% Matrix diagonal
mat = torch.randn(3,3)
diag = torch.einsum('ii->i', mat)
print(diag)