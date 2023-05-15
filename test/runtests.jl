using FeatureImportance
using Test
using Flux, LinearAlgebra

@testset "FeatureImportance.jl" begin
    X = randn(4,100)
    Y = X[2,:].^2 - X[3,:].^3
    Y = reshape(Y,1,:)
    model = Chain(Dense(4,2,relu),Dense(2,1))
    lossfn(model,x;target=nothing) = Flux.mse(model(x),target)
    res = importance(model,lossfn,X,target=Y)
    @test length(res) == 4
    @test typeof(res) == Vector{Float32}

    m = 3; n = 4
    fm = vcat(reshape([j^k for k = 1:m for j = 1:n], (n,m))',[l^2+l for l = 1:n]')
    model1 = Dense(3 => 1)
    model1.weight .= [1.01,0.99,0.01]' 
    res1 = importance(model1, lossfn, fm[1:3,:], target = fm[4,:]')
    @test res1[2] > res1[1] > res1[3]
end
