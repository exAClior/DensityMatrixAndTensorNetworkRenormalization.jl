using TensorKit

T = ComplexF64

X_mat = T[
    0.0 1.0
    1.0 0.0
]

Z_mat = T[
    1.0 0.0
    0.0 -1.0
]   

Y_mat = T[
    0.0 -im
    im 0.0
]

I_mat = T[
    1.0 0.0
    0.0 1.0
]



B = 1.0
J = 1.0
-B * Z_mat
h1 = TensorMap(reshape([-B*Z_mat X_mat Y_mat Z_mat I_mat],2,2,5), ℂ^5 → (ℂ^2 ⊗ ℂ^2))
h_loc = TensorMap(reshape([I_mat J*X_mat J*Y_mat J*Z_mat -B*Z_mat zeros(T,2,2) zeros(T,2,2) zeros(T,2,2) zeros(T,2,2)  X_mat zeros(T,2,2) zeros(T,2,2) zeros(T,2,2)  zeros(T,2,2) Y_mat zeros(T,2,2) zeros(T,2,2) zeros(T,2,2)  zeros(T,2,2) Z_mat zeros(T,2,2) zeros(T,2,2) zeros(T,2,2) zeros(T,2,2) I_mat],2,2,5,5), (ℂ^5 ⊗ ℂ^5) → (ℂ^2 ⊗ ℂ^2))
hN_ob = TensorMap(reshape(hcat(I_mat, repeat(zeros(T,2,2),4)'),2,2,5) ,ℂ^5 → (ℂ^2 ⊗ ℂ^2))  
hN_pb = TensorMap(reshape([I_mat J*X_mat J*Y_mat J*Z_mat zeros(T,2,2)],2,2,5),ℂ^5 → (ℂ^2 ⊗ ℂ^2))  

@tensor opt=true Ham[i,j,k,l] := view(h1[],:,:,:)[i,j,z] * view(h1[],:,:,:)[k,l,z]

A = TensorMap(randn,ComplexF64,ℂ^2 → ℂ^3)
B = TensorMap(randn,ComplexF64,ℂ^2 → ℂ^3)

# what if I don't want to take the complex conjugate of the mapping?
@tensor opt=true myass[i,j] := A[i,k] * adjoint(B)[k,j]
A[] * B'[]
@assert A[] * B'[] ≈ myass[]
@tensor opt=true myass[i,j] := A[i,k] * permute(B,(2,),(1,))[k,j]
@tensor opt=true Ham[i, j, k, l] := h1[i, j, m] * h1[k, l, m]

h1

@tensor Ham[i,j,k,l,m,n,o,p] := h1[i,j,z] * permute(h_loc,(,),(,))[] * permute(h_loc,(),())[m,n,y,x] * permute(AhN_ob,(),())[o,p,x]   

h_rnd = TensorMap(randn,ComplexF64, ℂ^5 → ℂ^2)

@tensor opt=true ham[i,j] := h_rnd[i,z] * h_rnd'[z,j]