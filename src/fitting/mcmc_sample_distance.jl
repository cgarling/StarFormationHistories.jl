struct MCMCModelDistance{A <: AbstractVector{<:AbstractMatrix{<:Number}},
                         B <: AbstractVector{<:AbstractMatrix{<:Number}},
                         C <: AbstractVector{<:Number},
                         D <: AbstractVector{<:Number},
                         E <: Distribution{Univariate, Continuous},
                         F}
    models::A
    composites::B # Vector of matrices, one per thread
    xcolors::C
    ymags::D
    distance_prior::E
    edges::F
end

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
    return MCMCModelDistance(models, [Matrix{V}(undef, size(first(models))) for i in 1:Threads.nthreads()], xcolors, ymags, distance_prior, edges)
end

# This model will return loglikelihood only
LogDensityProblems.capabilities(::Type{<:MCMCModelDistance}) = LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.dimension(problem::MCMCModelDistance) = length(problem.models)

# Make the type callable for the loglikelihood
function (problem::MCMCModelDistance)(θ)
    T = promote_type(eltype(first(problem.composites)), eltype(problem.xcolors), eltype(problem.ymags), eltype(first(problem.models)))
    new_distance = first(θ)
    for coeff in θ # If any coefficients are negative, return -Inf
        if coeff < zero(T)
            return typemin(T)
        end
    end
    # data = bin_cmd(problem.xcolors, problem.ymags .+ distance_modulus(new_distance * 1000), edges=problem.edges) # Construct new Hess diagram for the data given the new distance (in kpc)
    data = bin_cmd(problem.xcolors, problem.ymags, edges=(problem.edges[1], problem.edges[2] .+ distance_modulus(new_distance * 1000))) # Construct new Hess diagram for the data given the new distance (in kpc)

    # idx = Threads.threadid()
    # This should be valid for vectors with non 1-based indexing
    idx = first(axes(problem.composites))[Threads.threadid()] 
    composite!( problem.composites[idx], view(θ,2:lastindex(θ)), problem.models )
    return loglikelihood(problem.composites[idx], data.weights) + convert(T, logpdf(problem.distance_prior,new_distance))
end
LogDensityProblems.logdensity(problem::MCMCModelDistance, θ) = problem(θ)

# function (problem::FullSFHDistanceNoTF)(θ)
#     new_distance = first(θ) 
#     coeffs = θ[2:end] 
#     ((new_distance < 0) || (new_distance > 300)) && return -Inf # Set a valid range for the distance parameter
#     for c in coeffs
#         if c < 0
#             return -Inf
#         end
#     end
#     data = bin_cmd(problem.xcolors, problem.ymags .+ distance_modulus(new_distance * 1000), edges=problem.edges) # Construct new Hess diagram for the data given the new distance (in kpc)
#     return loglikelihood( sum(coeffs .* problem.templates), data.weights) + logpdf(Normal(38.0,7.0), new_distance) 
# end

"""
    result::MCMCChains.Chains = mcmc_sample(models::AbstractVector{<:AbstractMatrix{T}}, data::AbstractMatrix{S}, x0::Union{AbstractVector{<:AbstractVector{<:Number}}, AbstractMatrix{<:Number}}, nwalkers::Integer, nsteps::Integer; nburnin::Integer=0, nthin::Integer=1, a_scale::Number=2.0, use_progress_meter::Bool=true)

Samples the posterior of the coefficients `coeffs` such that the full model of the observational `data` is `sum(models .* coeffs)`. Uses the Poisson likelihood ratio as defined by equations 7--10 of Dolphin 2002. Sampling is done using the affine-invariant MCMC sampler implemented in [KissMCMC.jl](https://github.com/mauro3/KissMCMC.jl), which is analogous to Python's [emcee.moves.StretchMove](https://emcee.readthedocs.io/en/stable/). This method will automatically parallelize over threads. If you need distributed execution, you may want to look into [AdvancedMH.jl](https://github.com/TuringLang/AdvancedMH.jl). 

# Arguments
 - `models::AbstractVector{<:AbstractMatrix{<:Number}}` is a vector of equal-sized matrices that represent the template Hess diagrams for the simple stellar populations that compose the observed Hess diagram.
 - `data::AbstractMatrix{<:Number}` is the Hess diagram for the observed data.
 - `x0::Union{AbstractVector{<:AbstractVector{<:Number}}, AbstractMatrix{<:Number}}` are the initial positions for the MCMC walkers. If providing a vector of vectors, it must be a vector of length `nwalkers` with each internal vector having length equal to `length(models)`. You can alternatively provide a matrix of size `(nwalkers, length(models))` or `(length(models), nwalkers)`.
 - `nwalkers::Integer` is the number of unique walkers or chains to use.
 - `nsteps::Integer` is the number of steps evolve the walkers for.

# Keyword Arguments
 - `nburnin::Integer=0` is the number of steps to discard from the start of each chain.
 - `nthin::Integer=1` is the factor by which to thin the chain; walker positions will only be saved every `nthin` steps.
 - `a_scale::Number=2.0` is the scale parameter for the stretch move; probably shouldn't need to be changed.
 - `use_progress_Meter::Bool=true` indicates whether or not to show a progress bar during the MCMC procedure.

# Returns
 - `result` is a `MCMCChains.Chains` instance which enables easy calculation of diagnostic and summary statistics. This type can be indexed and used like a 3-D array of samples with shape `(nsteps, length(models), nwalkers)`.

# Notes
 - When displaying `result` to the terminal it will display summary statistics (`MCMCChains.summarystats`) and quantiles (`MCMCChains.quantile`) by calling the `MCMCChains.describe` method. This can take a second but is nice to have as an option.
 - The highest posterior density interval, which is the narrowest [credible interval](https://en.wikipedia.org/wiki/Credible_interval) that includes the posterior mode, can be calculated with the `MCMCChains.hpd` method. 
 - If you want to extract the array of samples from the `MCMCChains.Chains` object, you can index `result.value` -- this will return an `AxisArray` but can be converted to a normal array with `Array(result.value)`.

# Examples
```julia
import Distributions: Poisson
coeffs = rand(10) # SFH coefficients we want to sample
models = [rand(100,100) .* 100 for i in 1:length(coeffs)] # Vector of model Hess diagrams
data = rand.(Poisson.( sum(models .* coeffs) ) ) # Poisson-sample the model `sum(models .* coeffs)`
nwalkers = 1000
nsteps = 400
x0 = rand(nwalkers, length(coeffs)) # Initial walker positions
result = mcmc_sample(models, data, x0, nwalkers, nsteps) # Sample
Chains MCMC chain (400×10×1000 Array{Float64, 3}) ...
```
"""
function mcmc_sample(models::AbstractVector{<:AbstractMatrix{T}}, data::AbstractMatrix{S}, x0::AbstractVector{<:AbstractVector{<:Number}}, nwalkers::Integer, nsteps::Integer; nburnin::Integer=0, nthin::Integer=1, a_scale::Number=2.0, use_progress_meter::Bool=true) where {T <: Number, S <: Number}
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
