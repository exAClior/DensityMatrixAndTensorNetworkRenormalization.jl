# %% [markdown]
# # Linear Algebra
# We explore how to represent ideas in linear algebra using 
# `Tensorkit`. This is largely following this [tutorial](https://quantumghent.github.io/TensorTutorials/2-TensorNetworks/LinearAlgebra.html)

# %% 
using TensorKit


# %% [markdown]
# ## Vector Space
# We may create a vector space using `Tensorkit` as follows:

# %%
V = ℂ^2 # \bbC<TAB>
W = ℂ^3

# creates a matrix (restricted map between two vector spaces) 
A = TensorMap(rand,Float64, V → W) # direction of arrow don't matter as long as semantics is correct

# creates a vector (tensor that lives in the vector space)
v = Tensor(rand, Float64, V) 

@time w = A * v

w[1] ≈ A[1,1] * v[1] + A[1,2] * v[2]

# ## Tensor and Tensor Product

λ =  rand()

(λ * v) ⊗ w ≈ v ⊗ (λ * w) ≈ λ * (v ⊗ w)

t = Tensor(rand, Float64, V ⊗ W)

t[] # API for getting the matrix out of the `Tensor` struct

# t lives in the space V ⊗ W , its basis are the tensor product of the basis of V and W
# the index of t is a tuple of indices (i,j) where i is the index of V and j is the index of W
# You may convert from LinearIndices (i *k + j) to CartesianIndices (i,j)

LinearIndices((1:2,1:3))

collect(CartesianIndices((1:2,1:3)))

V1 = ℂ^2
V2 = ℂ^3
W1 = ℂ^3
W2 = ℂ^2

A = TensorMap(rand, Float64, V1 ⊗ V2 → W1 ⊗ W2)
v = Tensor(rand, Float64, V1 ⊗ V2)

w = A * v

w[]

w[] ≈ reshape(reshape(A[], (6,6)) * reshape(v[],6), (3,2))

# Note, we could also represent the linear map as a vector in another vector space

A = TensorMap(rand, Float64, W ← V)
B = Tensor(rand, Float64, W ⊗ V')
space(A ,2) == space(B, 2)