using Test, DensityMatrixAndTensorNetworkRenormalization


@testset "Lattice" begin
    a, n, pbc = 1, 3, true

    lattice = SquareLattice(n, a, pbc)

    @test lattice.n == n
    @test lattice.a == a
    @test lattice.pbc == pbc
    @test lattice.neighbors == [3 7 2 4; 1 8 3 5; 2 9 1 6; 6 1 5 7; 4 2 6 8; 5 3 4 9; 9 4 8 1; 7 5 9 2; 8 6 7 3]

    a, n, pbc = 1, 3, false

    lattice = SquareLattice(n, a, pbc)

    @test lattice.n == n
    @test lattice.a == a
    @test lattice.pbc == pbc
    @test lattice.neighbors == [0 0 2 4; 1 0 3 5; 2 0 0 6; 0 1 5 7; 4 2 6 8; 5 3 0 9; 0 4 8 0; 7 5 9 0; 8 6 0 0]
end

@testset "Hamiltonian" begin
    n, a, pbc, h, J, β = 3, 1.0, true, 0.0, 1.0, 0.0000000001

    ising = IsingModel(SquareLattice(n, a, pbc), h, J)

    naive_Z = 0.0
    for σs in Iterators.product(repeat([(-1, 1)], n^2)...)
        ham = 0.0
        for ii in 1:n^2
            for jj in 1:4
                ham += -ising.J * σs[ii] * σs[ising.lattice.neighbors[ii, jj]] / 2.0
            end
        end
        naive_Z += exp(-β * ham)
    end
    @test naive_Z ≈ partition_function(β, ising)
end