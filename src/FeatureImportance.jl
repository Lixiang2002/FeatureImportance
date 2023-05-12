module FeatureImportance

# Write your package code here.
using Random

function importance(model, loss, features; iters=10, seed = 1234, target = nothing)
    features = convert(Matrix{Float32}, features)
    !isnothing(target) && (target = convert(Matrix{Float32},target))
    
    RNG = MersenneTwister(seed)
    obj::Float32 = loss(model,features,target=target) 
    m::Int32,n::Int32 = size(features) 
    δ = zeros(m) |> Vector{Float32}
    temp = copy(features)
    for i = 1:m
        for _ = 1:iters
            temp[i,:] .= features[i,randperm(RNG,n)]
            δ[i] += loss(model,temp,target=target)
        end
        temp[i,:] .= features[i,:]
    end
    δ .= δ./iters .- obj
end

export importance

end
