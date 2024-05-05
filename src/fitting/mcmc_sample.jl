struct MCMCModel{T <: AbstractMatrix{<:Number},
                 S <: AbstractVector{<:AbstractVector{<:Number}},
                 V <: AbstractVector{<:Number}}
    models::T
    composites::S # Vector of matrices, one per thread
    data::V
end

# Function to construct the above type when provided with just models and data
function MCMCModel(models::T, data::S) where {A <: Number,
                                              B <: Number,
                                              T <: AbstractMatrix{A},
                                              S <: AbstractVector{B}}
    V = promote_type(A,B)
    return MCMCModel(models, [Vector{V}(undef, length(data)) for i in 1:Threads.nthreads()], data)
end

# This model will return loglikelihood only
LogDensityProblems.capabilities(::Type{<:MCMCModel}) = LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.dimension(problem::MCMCModel) = size(problem.models,2)

# Make the type callable for the loglikelihood
function (problem::MCMCModel)(θ)
    T = promote_type(eltype(first(problem.composites)), eltype(problem.data))
    for coeff in θ # If any coefficients are negative, return -Inf
        if coeff < zero(T)
            return typemin(T)
        end
    end
    # idx = Threads.threadid()
    # This should be valid for vectors with non 1-based indexing
    idx = first(axes(problem.composites))[Threads.threadid()] 
    composite!( problem.composites[idx], θ, problem.models )
    return loglikelihood(problem.composites[idx], problem.data)
end
LogDensityProblems.logdensity(problem::MCMCModel, θ) = problem(θ)

"""
    convert_kissmcmc(chains::Vector{Vector{Vector{<:Number}}})
Converts output `Vector{Vector{Vector{<:Number}}}` from multivariate `KissMCMC.emcee` sample to 3-D matrix with size `(nsteps, npar, nwalkers)` to match `MCMCChains.Chains` API. Specifically, the nested vectors output from KissMCMC have lengths `(nwalkers, nsteps, length(models))` in the first, second, and third dimension, respectively. 
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

"""
    result::MCMCChains.Chains =
    mcmc_sample(models::AbstractVector{<:AbstractMatrix{T}},
                data::AbstractMatrix{S},
                x0::Union{AbstractVector{<:AbstractVector{<:Number}}, AbstractMatrix{<:Number}},
                nwalkers::Integer,
                nsteps::Integer;
                nburnin::Integer=0,
                nthin::Integer=1,
                a_scale::Number=2.0,
                use_progress_meter::Bool=true)
    mcmc_sample(models::AbstractMatrix{<:Number},
                data::AbstractVector{<:Number},
                args...; kws...)

Samples the posterior of the coefficients `coeffs` such that the full model of the observational `data` is `sum(models .* coeffs)`. Uses the Poisson likelihood ratio as defined by equations 7--10 of Dolphin 2002. Sampling is done using the affine-invariant MCMC sampler implemented in [KissMCMC.jl](https://github.com/mauro3/KissMCMC.jl), which is analogous to Python's [emcee.moves.StretchMove](https://emcee.readthedocs.io/en/stable/). This method will automatically parallelize over threads. If you need distributed execution, you may want to look into [AdvancedMH.jl](https://github.com/TuringLang/AdvancedMH.jl).

The second call signature supports the flattened formats for `models` and `data`. See the notes for the flattened call signature of [`StarFormationHistories.composite!`](@ref) for more details.

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
function mcmc_sample(models::AbstractMatrix{T}, data::AbstractVector{S}, x0::AbstractVector{<:AbstractVector{<:Number}}, nwalkers::Integer, nsteps::Integer; nburnin::Integer=0, nthin::Integer=1, a_scale::Number=2.0, use_progress_meter::Bool=true) where {T <: Number, S <: Number}
    instance = MCMCModel( models,
                          [Vector{promote_type(T,S)}(undef, length(data)) for i in 1:Threads.nthreads()],
                          data )
    samples, _ = KissMCMC.emcee(instance, x0; niter=nwalkers*nsteps, nburnin=nburnin*nwalkers, nthin=nthin, a_scale=a_scale, use_progress_meter=use_progress_meter)
    return MCMCChains.Chains(convert_kissmcmc(samples))
end
# KissMCMC Output:
# - samples: Vector{Vector{Vector{eltype(first(x0))}}}` with lengths (nwalkers, nsteps, length(models))
# - accept_ratio: ratio of accepted to total steps, average per walker `Vector` of size(nwalkers)
# - logdensities: the value of the log-density for each sample, same shape as `samples`

# Method for x0 as a Matrix rather than vector of vectors
function mcmc_sample(models::AbstractMatrix{<:Number}, data::AbstractVector{<:Number}, x0::AbstractMatrix{<:Number}, nwalkers::Integer, nsteps::Integer; kws...)
    if size(x0) == (nwalkers, size(models,2))
        mcmc_sample(models, data, [copy(i) for i in eachrow(x0)], nwalkers, nsteps; kws...)
    elseif size(x0) == (size(models,2), nwalkers)
        mcmc_sample(models, data, [copy(i) for i in eachcol(x0)], nwalkers, nsteps; kws...)
    else
        throw(ArgumentError("You provided a misshapen `x0` argument of type `AbstractMatrix{<:Number}` to `mcmc_sample`. When providing a matrix for `x0`, it must be of size `(nwalkers, Nmodels)` or `(Nmodels, nwalkers)`, where `Nmodels` is the number of model templates you are providing (`Nmodels = size(models,2)` if you are passing `models` as a single flattened matrix, or `Nmodels = length(models)` if you are passing `models` as a vector of matrices)."))
    end
end
mcmc_sample(models::AbstractVector{<:AbstractMatrix{<:Number}}, data::AbstractMatrix{<:Number}, args...; kws...) = mcmc_sample(stack_models(models), vec(data), args...; kws...)
