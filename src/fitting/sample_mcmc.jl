struct MCMCModel{T <: AbstractVector{<:AbstractMatrix{<:Number}},
                 S <: AbstractVector{<:AbstractMatrix{<:Number}},
                 V <: AbstractMatrix{<:Number}}
    models::T
    composites::S # Vector of matrices, one per thread
    data::V
end

# Function to construct the above type when provided with a single `composite` matrix
function MCMCModel(models::T, composite::S, data::V) where {T <: AbstractVector{<:AbstractMatrix{<:Number}},
                                                            S <: AbstractMatrix{<:Number},
                                                            V <: AbstractMatrix{<:Number}}
    return MCMCModel(models, [similar(composite) for i in 1:Threads.nthreads()], data)
end

# This model will return loglikelihood only
LogDensityProblems.capabilities(::Type{<:MCMCModel}) = LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.dimension(problem::MCMCModel) = length(problem.models)

# Make the type callable for the loglikelihood
function (problem::MCMCModel)(θ)
    T = promote_type(eltype(first(problem.composites)), eltype(problem.data))
    for coeff in θ # If any coefficients are negative, return -Inf
        if coeff < zero(T)
            return typemin(T)
        end
    end
    # idx = Threads.threadid()
    idx = first(axes(problem.composites))[Threads.threadid()] # This should be valid for any vector
    composite!( problem.composites[idx], θ, problem.models )
    return loglikelihood(problem.composites[idx], problem.data)
end
LogDensityProblems.logdensity(problem::MCMCModel, θ) = problem(θ)

"""
    convert_kissmcmc(chains::Vector{Vector{Vector{<:Number}}})
Converts output `Vector{Vector{Vector{<:Number}}}` from multivariate `KissMCMC.emcee` sample to 3-D matrix with size `(nsteps, npar, nwalkers)` to match `MCMCChains.Chains` API.
"""
function convert_kissmcmc(chains::AbstractVector{<:AbstractVector{<:AbstractVector{T}}}) where T <: Number
    nwalkers = length(chains)
    nsteps = length(first(chains))
    npar = length(first(first(chains)))
    newmat = Array{T}(undef,(nsteps,npar,nwalkers))

    for i in eachindex(chains)
        walker = chains[i]
        for j in eachindex(walker)
            step = walker[j]
            newmat[j,:,i] .= step
        end
    end
    return newmat
end

function mcmc_sample(models::AbstractVector{<:AbstractMatrix{T}}, data::AbstractMatrix{S}, x0::AbstractVector{<:AbstractVector{<:Number}}, nwalkers::Integer, nsteps::Integer; rng::AbstractRNG=default_rng(), nburnin::Integer=0, nthin::Integer=1, a_scale::Number=2.0, use_progress_meter::Bool=true) where {T <: Number, S <: Number}
    instance = MCMCModel( models,
                          [Matrix{promote_type(T,S)}(undef, size(data)) for i in 1:Threads.nthreads()],
                          data )
    samples, _ = KissMCMC.emcee(instance, x0; niter=nwalkers*nsteps, nburnin=nburnin*nwalkers, nthin=nthin, a_scale=a_scale, use_progress_meter=use_progress_meter)
    return MCMCChains.Chains(convert_kissmcmc(samples))
end
# KissMCMC Output:
# - samples: Vector{Vector{Vector{eltype(first(x0))}}}` with lengths (nwalkers, nsteps, length(models))
# - accept_ratio: ratio of accepted to total steps, average per walker `Vector` of size(nwalkers)
# - logdensities: the value of the log-density for each sample, same shape as `samples`

# Method for x0 as a Matrix rather than vector of vectors
function mcmc_sample(models::AbstractVector{<:AbstractMatrix{<:Number}}, data::AbstractMatrix{<:Number}, x0::AbstractMatrix{<:Number}, nwalkers::Integer, nsteps::Integer; kws...)
    if size(x0) == (nwalkers, length(models))
        mcmc_sample(models, data, [copy(i) for i in eachrow(x0)], nwalkers, nsteps; kws...)
    elseif size(x0) == (length(models), nwalkers)
        mcmc_sample(models, data, [copy(i) for i in eachcol(x0)], nwalkers, nsteps; kws...)
    else
        throw(ArgumentError("You provided a misshapen `x0` argument of type `AbstractMatrix{<:Number}` to `mcmc_sample`. When providing a matrix for `x0`, it must be of size `(nwalkers, length(models))` or `(length(models), nwalkers)`."))
    end
end

import Random: AbstractRNG, default_rng
import Distributions: Poisson
import LogDensityProblems
import MCMCChains
import KissMCMC
import StarFormationHistories: loglikelihood, composite!

rng=default_rng()
T=Float64
nmodels=10
hist_size=(100,100)

coeffs = rand(rng,nmodels) #.* 100
coeffs[begin] = 0 # Set some coeffs to zero to model real applications
coeffs[end] = 0
models = [rand(rng,T,hist_size) .* 100 for i in 1:nmodels]
data = rand.(Ref(rng), Poisson.( sum(models .* coeffs) ) )

nwalkers=1000
nsteps=400
x0 = rand(nwalkers,nmodels) #.* 100
result = mcmc_sample(models, data, x0, nwalkers, nsteps)
