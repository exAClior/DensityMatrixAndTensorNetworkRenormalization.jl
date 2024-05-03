using LinearAlgebra
using OMEinsum

abstract type AbstractLattice end

struct SquareLattice{T} <: AbstractLattice
    n::Int
    a::T
    pbc::Bool
    neighbors::Matrix{Int}
end

function SquareLattice(n::Int, a::T, pbc::Bool) where {T}
    neighbors = Matrix{Int}(undef, n^2, 4)
    for (li, ci) in enumerate(CartesianIndices((n, n)))
        i, j = Tuple(ci)
        #      1
        #    2 m 4
        #      3
        neighbors[li, 1] = i == 1 ? (pbc ? li + n - 1 : 0) : li - 1  # Up
        neighbors[li, 2] = j == 1 ? (pbc ? li + n^2 - n : 0) : li - n  # Left
        neighbors[li, 3] = i == n ? (pbc ? li - n + 1 : 0) : li + 1  # Down
        neighbors[li, 4] = j == n ? (pbc ? li - n^2 + n : 0) : li + n  # Right
    end
    return SquareLattice(n, a, pbc, neighbors)
end

abstract type AbstractHamiltonian end

struct IsingModel{T,LT<:AbstractLattice} <: AbstractHamiltonian
    lattice::LT
    h::T
    J::T
end

function neighbor_idces(model::IsingModel{T,LT}) where {T,LT}
    n = model.lattice.n
    return Tuple([("ii$ii", "ii$jj") for ii in 1:n^2 for jj in 1:4])
end

function partition_function(β::T, model::IsingModel{T,LT}) where {T,LT}
    n = model.lattice.n
    M = [exp(β * model.J * Si * Sj / 2.0) for Si in (-one(T), one(T)), Sj in (-one(T), one(T))]
    raw_code = EinCode(neighbor_idces(model), ())
    opt_code = optimize_code(raw_code, uniformsize(raw_code, 2), TreeSA())
    return opt_code(repeat([M], n^2 * 4)...)[]
end

