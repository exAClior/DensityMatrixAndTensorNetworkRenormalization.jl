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

function partition_function(β::T, model::IsingModel{T,LT}, method::Symbol) where {T,LT}
    if method == :MT
        partition_function_MT(β, model)
    elseif method == :TN
        partition_function_TN(β, model)
    elseif method == :DL
        partition_function_DL(β, model)
    else
        throw(ArgumentError("Invalid method: $method"))
    end
end


function mtx_idcs(model::IsingModel{T,LT}) where {T,LT}
    num_spins, num_neighbors = size(model.lattice.neighbors)
    return Tuple([("ii$ii", "ii$(model.lattice.neighbors[ii,jj])") for ii in 1:num_spins for jj in 1:num_neighbors])
end


function tensor_idcs(model::IsingModel{T,LT}) where {T,LT}
    lat = model.lattice
    num_spins = size(lat.neighbors, 1)
    return Tuple([Tuple(["i$(sort([ii,jj]))" for jj in lat.neighbors[ii, :]]) for ii in 1:num_spins])
end

function partition_function_MT(β::T, model::IsingModel{T,LT}) where {T,LT}
    n = model.lattice.n
    M = [exp(β * model.J * Si * Sj / 2.0) for Si in (-one(T), one(T)), Sj in (-one(T), one(T))]
    raw_code = EinCode(mtx_idcs(model), ())
    opt_code = optimize_code(raw_code, uniformsize(raw_code, 2), TreeSA())
    return opt_code(repeat([M], n^2 * 4)...)[]
end

function partition_function_TN(β::T, model::IsingModel{T,LT}) where {T,LT}
    # needs to be bipartite
    # currently only work for square lattice
    lat = model.lattice
    n = lat.n
    num_neighbor = size(lat.neighbors, 2)
    M = [exp(β * model.J * Si * Sj) for Si in (-one(T), one(T)), Sj in (-one(T), one(T))]
    U, λ, Vt = svd(M)
    Qa = U * Diagonal(sqrt.(λ))
    Qb = Diagonal(sqrt.(λ)) * transpose(Vt)
    T_rc = EinCode(Tuple([("l", "i$jj") for jj in 1:num_neighbor]), Tuple(["i$jj" for jj in 1:num_neighbor]))
    T_oc = optimize_code(T_rc, uniformsize(T_rc, 2), TreeSA())
    Ta = T_oc(repeat([Qa], num_neighbor)...)
    T_rc = EinCode(Tuple([("i$jj", "l") for jj in 1:num_neighbor]), Tuple(["i$jj" for jj in 1:num_neighbor]))
    T_oc = optimize_code(T_rc, uniformsize(T_rc, 2), TreeSA())
    Tb = T_oc(repeat([Qb], num_neighbor)...)
    raw_code = EinCode(tensor_idcs(model), ())
    opt_code = optimize_code(raw_code, uniformsize(raw_code, 2), TreeSA())
    # bipartite square lattice must have even length
    odd_cols = repeat([Ta, Tb], div(n, 2))
    even_cols = repeat([Tb, Ta], div(n, 2))
    tns = vcat(repeat([vcat(odd_cols, even_cols)], div(n, 2))...)
    return opt_code(tns...)[]
end



M = [exp(0.01 * 1.0 * Si * Sj / 2.0) for Si in (-1.0, 1.0), Sj in (-1.0, 1.0)]
U, λ, Vt = svd(M)

U ≈ Vt