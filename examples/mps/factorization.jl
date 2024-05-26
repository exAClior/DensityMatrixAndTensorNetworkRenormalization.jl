using TensorKit, Test

N = 5 # number of sites
d = 2 # site dimension
V = foldr(⊗, [ℂ^d for _ in 1:N]) 
ψ = Tensor(randn,ComplexF64, V)

dims(V)
ψ.data  
Ds = dim(space(ψ))
Q,R = leftorth(ψ, ((1,),(2,3,4,5)),alg=QR())

function exact_factorize(ψ::Tensor{ComplexF64},Ds::AbstractVector{Int})
    As = Tensor{ComplexF64}[]
    R_t = copy(ψ)
    R_t = reshape(R_t, (1, Ds))
    for i in 1:length(Ds)
        A,R_t = leftorth(R_t, ((1,),(2,3,4,5)),alg=QR()) 
        push!(As, A)
    end
    return As
end