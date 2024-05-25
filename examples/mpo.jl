using TensorKit, MPSKit
β= 1.0

t = TensorMap(ComplexF64[exp(β) exp(-β); exp(-β) exp(β)], ℂ^2, ℂ^2)
q = sqrt(t)

δ = TensorMap(zeros,ComplexF64,ℂ^2 ⊗ ℂ^2,ℂ^2 ⊗ ℂ^2)
δ[1,1,1,1] = 1.0
δ[2,2,2,2] = 1.0


@tensor O[-1 -2; -3 -4] := δ[1 2; 3 4] * q[-1 ; 1] * q[-2; 2] * q[3;-3]  * q[4;-4]
