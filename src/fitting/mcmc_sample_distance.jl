# Implementation of SFH fitting with distance, no metallicity constraints

struct MCMCModelDistance{A <: AbstractMatrix{<:Number},
                         B <: AbstractVector{<:Number},
                         C <: AbstractVector{<:Number},
                         D <: Distribution{Univariate, Continuous},
                         E}
    models::A
    xcolors::B
    ymags::C
    distance_prior::D
    edges::E
end
# edges is a 2-tuple; first element is x-bins, second element is y-bins where
# y-bins are the absolute magnitude bins used to construct the models from isochrones.

function MCMCModelDistance(models::A,
                           xcolors::B,
                           ymags::C,
                           distance_prior::D,
                           edges) where {AA <: Number,
                                         A <: AbstractVector{<:AbstractMatrix{AA}},
                                         B <: AbstractVector{<:Number},
                                         C <: AbstractVector{<:Number},
                                         D <: Distribution{Univariate, Continuous}}
    V = promote_type(AA, eltype(B), eltype(C))
    @assert length(xcolors) == length(ymags)
    # Stack models for more efficient computation
    models = stack_models(models)
    return MCMCModelDistance(models, xcolors, ymags, distance_prior, edges)
end

# This model will return loglikelihood only
LogDensityProblems.capabilities(::Type{<:MCMCModelDistance}) = LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.dimension(problem::MCMCModelDistance) = length(problem.models)

# Make the type callable for the loglikelihood
function (problem::MCMCModelDistance)(θ)
    T = promote_type(eltype(problem.xcolors), eltype(problem.ymags), eltype(first(problem.models)))
    new_distance = first(θ)
    for coeff in θ # If any coefficients are negative, return -Inf
        if coeff < zero(T)
            return typemin(T)
        end
    end
    # Construct new Hess diagram for the data given the new distance (in kpc)
    data = bin_cmd(problem.xcolors, problem.ymags,
                   edges=(problem.edges[1], problem.edges[2] .+ distance_modulus(new_distance * 1000)))

    models = problem.models
    C = Vector{T}(undef, first(size(models))) # Vector to contain composite model
    composite!(C, view(θ,2:lastindex(θ)), models)
    return loglikelihood(C, vec(data.weights)) + convert(T, logpdf(problem.distance_prior, new_distance))
end
LogDensityProblems.logdensity(problem::MCMCModelDistance, θ) = problem(θ)

## Convenience functions
## This is an advanced enough use case that we are not providing a convenience function at this time.


############################################
# # Currently broken; KissMCMC.emcee will iterate and accept transitions but samples returned are all identical
# import Distributions: Distribution, Univariate, Continuous, Poisson, Uniform, logpdf
# import LogDensityProblems
# import StarFormationHistories: bin_cmd, loglikelihood, composite!, distance_modulus, convert_kissmcmc
# import KissMCMC
# import MCMCChains

# # coeffs = rand(10) # SFH coefficients we want to sample
# # models = [rand(100,100) .* 100 for i in 1:length(coeffs)] # Vector of model Hess diagrams
# distance = 20 # kpc
# xdist = Uniform(0,2)
# xcolors = rand(xdist, 10000)
# ydist = Uniform(0,10)
# ymags = rand(ydist + distance_modulus(distance * 1000), 10000)
# model_edges = (range(extrema(xdist)...; length=101), range(extrema(ydist)...; length=101))
# model = rand.(Poisson.(bin_cmd(xcolors, ymags .- distance_modulus(distance * 1000), edges=model_edges).weights))
# # model = bin_cmd(xcolors, ymags .- distance_modulus(distance * 1000), edges=model_edges).weights
# instance = MCMCModelDistance([model], xcolors, ymags, Uniform(0,100), model_edges)

# nwalkers=100
# nsteps=50
# result = MCMCChains.Chains(convert_kissmcmc(KissMCMC.emcee(instance, [[distance+1,0.5] for i in 1:nwalkers]; niter=nwalkers*nsteps, nburnin=0, nthin=1, use_progress_meter=true)[1]))
