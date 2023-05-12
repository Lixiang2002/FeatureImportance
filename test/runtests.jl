using FeatureImportance
using Test
using Flux, Random, LinearAlgebra

@testset "FeatureImportance.jl" begin
    X = randn(4,100)
    Y = X[2,:].^2 - X[3,:].^3
    Y = reshape(Y,1,:)
    model = Chain(Dense(4,2,relu),Dense(2,1))
    lossfn(model,x;target=nothing) = Flux.mse(model(x),target)
    res = importance(model,lossfn,X,target=Y)
    @test length(res) == 4
end
