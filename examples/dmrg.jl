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

# use einsum instead of ncon for now
# @tensor Ham[-1,-2,-3,-4,-5,-6,-7,-8] := h1[-1;-2;1]*h_loc[1;-3;-4;2] * h_loc[2;-5;-6;3] * hN_ob[3;-7;-8]