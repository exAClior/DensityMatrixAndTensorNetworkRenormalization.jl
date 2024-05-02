using Test, DensityMatrixAndTensorNetworkRenormalization


@testset "Lattice" begin
    a, n, pbc = 1, 3, true

    lattice = SquareLattice(n, a, pbc)

    @test lattice.n == n
    @test lattice.a == a
    @test lattice.pbc == pbc
    @test lattice.neighbors = [2 3 1 3; 3 1 2 1; 1 2 3 2]


end