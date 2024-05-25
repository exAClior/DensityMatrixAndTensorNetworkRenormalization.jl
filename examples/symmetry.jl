# Symmetry
## Irreducible Representation
# Under arbitrary representation, a linear map may require all of its
# elements to be memorized. However, under correct representation. (With irreducible representation where
# the linear map is block diagonal), the linear map may be represented with less memory.

using TensorKit
# take the direct sum of 1 vector spaces where each vector space is the $s=1/2$ subspace of SU(2)
s = SU2Space(1/2 => 1)
l = SU2Space(1 => 1)

ss = SU2Space(1/2 => 2)

A = TensorMap(l ← s ⊗ s)

@assert dim(domain(A)) == 4
@assert dim(codomain(A)) == 3
blocks(A)

# huh?
