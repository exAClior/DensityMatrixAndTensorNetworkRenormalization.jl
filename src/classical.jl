using LinearAlgebra


abstract type AbstractLattice end

struct SquareLattice{T} <: AbstractLattice
    n::Int
    a::T
    pbc::Bool
    neighbors::Matrix{Int}
end

function SquareLattice(n::Int, a::T, pbc::Bool) where {T}
    neighbors = Matrix{Int}(undef, n^2, 4)
    for idx in CartesianIndices(n, n)
        i, j = Tuple(idx)
        neighbors[idx, 1] = i == 1 ? (pbc ? n : 0) : i - 1
        neighbors[idx, 2] = i == n ? (pbc ? 1 : 0) : i + 1
        neighbors[idx, 3] = j == 1 ? (pbc ? n : 0) : j - 1
        neighbors[idx, 4] = j == n ? (pbc ? 1 : 0) : j + 1
    end
    return SquareLattice(n, a, pbc, neighbors)
end




abstract type AbstractHamiltonian end


struct IsingModel{D,T} <: AbstractHamiltonian
    lattice::SquareLattice{D,T}
    h::T
    J::T
end

