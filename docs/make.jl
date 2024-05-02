using DensityMatrixAndTensorNetworkRenormalization
using Documenter

DocMeta.setdocmeta!(DensityMatrixAndTensorNetworkRenormalization, :DocTestSetup, :(using DensityMatrixAndTensorNetworkRenormalization); recursive=true)

makedocs(;
    modules=[DensityMatrixAndTensorNetworkRenormalization],
    authors="Yusheng Zhao <yushengzhao2020@outlook.com> and contributors",
    sitename="DensityMatrixAndTensorNetworkRenormalization.jl",
    format=Documenter.HTML(;
        canonical="https://exAClior.github.io/DensityMatrixAndTensorNetworkRenormalization.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/exAClior/DensityMatrixAndTensorNetworkRenormalization.jl",
    devbranch="main",
)
