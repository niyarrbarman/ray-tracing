# %% Setup
import torch as t
import einops

MAIN = __name__ == "__main__"

# %% Elementwise Logical Operations on Tensors
v = 0.5
(v>=0.0) & (v<=1.0)

# %% Einops
x = t.randn(4,3)
x_repeated = einops.repeat(x, 'b c -> b c d', d=2)
assert x_repeated.shape == (4, 3, 2)
t.testing.assert_close(x_repeated[:, :, 0], x)
t.testing.assert_close(x_repeated[:, :, 1], x)

# %% Logical Reductions
x = t.tensor([True, False, True])
assert x.any() 
assert not x.all()

# %% Broadcasting
B = t.ones(4,3,2)
A = t.ones(3,2)
C = A + B
print(C.shape)
print(C.max(), C.min())

# %% Indexing
D = t.ones(2)
E = t.zeros(3,2)
E[[True, False, False], :].shape
E[[True, False, True], :] = D
print(E)


