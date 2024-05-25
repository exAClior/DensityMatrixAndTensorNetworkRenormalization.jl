using TensorKit, Test


A = reshape(1:2^4, 2,2,2,2)

B = reshape(A, 4,2,2)
B = reshape(A, 2,4,2)

A = rand(2,2,2)

B = rand(2,2)

# @tensor is defined in TensorOperations.jl which is the backend for
# doing tensor contractions but it's re-exported in TensorKit.jl
# := means creating new tensor
# = means assigning to a tensor
@tensor C[i,j,k,l,m] := A[i,j,k] * B[l,m]
@tensor D[i] := A[i,j,j]
@tensor E[i,j,l] := A[i,j,k] * B[l,k]
F = similar(A)

@tensor F[i,j,k] = A[i,j,k]
@test F ≈ A

# Instead of the einsum notation, we could also use the NCON notation
# it seems like outputs are denoted with negative indices, summed over
# indices are denoted with positive indices
# they appear in increasing orders

B = rand(2,2,2,2)
C = rand(2,2,2,2,2)
D = rand(2,2,2)
E = rand(2,2)
F = rand(2,2)


@time @tensor begin
    A[-1,-2] := B[-1,1,2,3] * C[3,5,6,7,-2] * D[2,4,5] * E[1,4] * F[6,7]
end

@time @tensor  opt=true begin
    A[-1,-2] := B[-1,1,2,3] * C[3,5,6,7,-2] * D[2,4,5] * E[1,4] * F[6,7]
end

# Tensor Factorization
## Eigenvalue decomposition
S1 = ℂ^2 ⊗ ℂ^2 ⊗ ℂ^2
S2 = ℂ^2 ⊗ ℂ^3

A = TensorMap(randn, ComplexF64, S1,S1)

D,V = eig(A)

@test A * V ≈ V * D

## Singular value decomposition

A = TensorMap(randn, ComplexF64, S1,S2)
partition = ((1,2),(3,4,5))
U,S,V = tsvd(A, partition...) # tensor svd
@test permute(A,partition) ≈ U * S * V
@test U'* U ≈ id(domain(U))
@test V* V' ≈ id(codomain(V))


## Polar Decomposition
# A = U * P where U is rotation/reflection and P is scaling

A = TensorMap(randn, ComplexF64, S1,S2)
partition = ((1,2),(3,4,5))
Q, P = leftorth(A, partition...; alg=Polar())
@test permute(A, partition) ≈ Q * P
@test Q * Q' ≈ id(domain(Q))
@test (Q * Q')^2 ≈ (Q * Q')


## QR Decomposition
# Useful for solving linear systems
# A = Q * R where Q is orthogonal and R is upper triangular
# R^-1 can be computed easily with gaussian elimination
A = TensorMap(randn, ComplexF64, S1,S2)
partition = ((1,2),(3,4,5))
Q, R = leftorth(A, partition...; alg=QR())
@test permute(A, partition) ≈ Q * R
@test Q' * Q ≈ id(domain(Q))


## Nullspaces
A = TensorMap(randn, ComplexF64, S1,S2)
partition = ((1,2,3),(4,5))
N = leftnull(A, partition...)
@test norm(N' * permute(A,partition)) ≈ 0 atol=1e-14
@test N' * N ≈ id(domain(N))
