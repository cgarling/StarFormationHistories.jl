struct HMCModel{T,S,V}
    models::T
    composite::S
    data::V
end

# This model will return loglikelihood and gradient
LogDensityProblems.capabilities(::Type{<:HMCModel}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(problem::HMCModel) = size(problem.models, 2)

function (problem::HMCModel)(θ)
    composite = problem.composite
    models = problem.models
    data = problem.data
    # Transform the provided x
    x = exp.(θ) # [ exp(i) for i in θ ]
    # Update the composite model matrix
    composite!(composite, x, models)
    logL = loglikelihood(composite, data) + sum(θ)
    return logL
end
LogDensityProblems.logdensity(problem::HMCModel, θ) = problem(θ)

function LogDensityProblems.logdensity_and_gradient(problem::HMCModel, logx)
    composite = problem.composite
    models = problem.models
    data = problem.data
    # Transform the provided x
    x = exp.(logx) # [ exp(i) for i in logx ]
    # Temp vector for gradient
    G = similar(x)
    # fg! constructs composite and computes -logL and -∇logL performantly
    # returns -logL and writes -∇logL to second arg
    logL = -fg!(true, G, x, models, data, composite) + sum(logx) # The `+ sum(logx)` is the Jacobian correction
    @. G = -G * x + 1 # The `* x[i] + 1` is the Jacobian correction
    return logL, G
end

"""
    hmc_sample(models::AbstractVector{T},
               data::AbstractMatrix{<:Number},
               nsteps::Integer [, nchains::Integer];
               rng::Random.AbstractRNG=Random.default_rng(),
               kws...)
               where {S <: Number, T <: AbstractMatrix{S}}
    hmc_sample(models::AbstractMatrix{S},
               data::AbstractVector{<:Number},
               nsteps::Integer;
               rng::AbstractRNG=default_rng(),
               kws...)
               where S <: Number

Function to sample the posterior of the coefficients `coeffs` such that the full model of the observational `data` is `sum(models .* coeffs)`. Uses the Poisson likelihood ratio as defined by equations 7--10 of Dolphin 2002 along with a logarithmic transformation of the `coeffs` so that the fitting variables are continuous and differentiable over all reals. Sampling is done using the No-U-Turn sampler as implemented in [DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl), which is a form of dynamic Hamiltonian Monte Carlo.

The second call signature supports the flattened formats for `models` and `data`. See the notes for the flattened call signature of [`StarFormationHistories.composite!`](@ref) for more details.

# Arguments
 - `models::AbstractVector{<:AbstractMatrix{<:Number}}` is a vector of equal-sized matrices that represent the template Hess diagrams for the simple stellar populations that compose the observed Hess diagram.
 - `data::AbstractMatrix{<:Number}` is the Hess diagram for the observed data.
 - `nsteps::Integer` is the number of samples to draw per chain.

# Optional Arguments
 - `nchains::Integer`: If this argument is not provided, this method will return a single chain. If this argument is provided, it will sample `nchains` chains using all available Julia threads and will return a vector of the individual chains.

# Keyword Arguments
 - `rng::Random.AbstractRNG` is the random number generator that will be passed to DynamicHMC.jl. If `nchains` is provided this method will attempt to sample in parallel, requiring a thread-safe `rng` such as that provided by `Random.default_rng()`. 
All other keyword arguments `kws...` will be passed to `DynamicHMC.mcmc_with_warmup` or `DynamicHMC.mcmc_keep_warmup` depending on whether `nchains` is provided.

# Returns
 - If `nchains` is not provided, returns a `NamedTuple` as summarized in DynamicHMC.jl's documentation. In short, the matrix of samples can be extracted and transformed as `exp.( result.posterior_matrix )`. Statistics about the chain can be obtained with `DynamicHMC.Diagnostics.summarize_tree_statistics(result.tree_statistics)`; you want to see a fairly high acceptance rate (>0.5) and the majority of samples having termination criteria being "turning." See DynamicHMC.jl's documentation for more information.
 - If `nchains` *is* provided, returns a vector of length `nchains` of the same `NamedTuple`s described above. The samples from each chain in the returned vector can be stacked to a single `(nsamples, nchains, length(models))` matrix with `DynamicHMC.stack_posterior_matrices(result)`. 

# Examples
```julia
import DynamicHMC
import StatFormationHistories: hmc_sample
import Statistics: mean
# Run sampler using progress meter to monitor progress
# assuming you have constructed some templates `models` and your observational Hess diagram `data`
result = hmc_sample( models, data, 1000; reporter=DynamicHMC.ProgressMeterReport())
# The chain values are stored in result.posterior matrix; extract them with `result.posterior_matrix`
# An exponential transformation is needed since the optimization internally uses a logarithmic 
# transformation and samples log(θ) rather than θ directly. 
mc_matrix = exp.( result.posterior_matrix )
# We can look at some statistics from the chain; want to see high acceptance rate (>0.5) and large % of
# "turning" for termination criteria. 
DynamicHMC.Diagnostics.summarize_tree_statistics(result.tree_statistics)
    Hamiltonian Monte Carlo sample of length 1000
      acceptance rate mean: 0.92, 5/25/50/75/95%: 0.65 0.88 0.97 1.0 1.0
      termination: divergence => 0%, max_depth => 0%, turning => 100%
      depth: 0 => 0%, 1 => 64%, 2 => 36%
# mc_matrix has size `(length(models), nsteps)` so each column is an independent
# sample of the SFH as defined by the coefficients and the rows contain the samples
# for each parameter. 
mstar_tot = sum.(eachcol(mc_matrix)) # Total stellar mass of the modelled system per sample
mc_means = mean.(eachrow(mc_matrix)) # Mean of each coefficient evaluated across all samples
# Example with multiple chains sampled in parallel via multi-threading
import Threads
t_result = hmc_sample( models, data, 1000, Threads.nthreads(); reporter=DynamicHMC.ProgressMeterReport())
# Combine the multiple chains into a single matrix and transform
# Can then use the same way as `mc_matrix` above
mc_matrix = exp.( DynamicHMC.pool_posterior_matrices(t_result) )
```
"""
function hmc_sample(models::AbstractMatrix{S}, data::AbstractVector{<:Number}, nsteps::Integer;
                    rng::AbstractRNG=default_rng(), kws...) where S <: Number
    @assert size(models, 1) == length(data)
    composite = Vector{S}(undef, length(data))
    instance = HMCModel(models, composite, data)
    # return instance
    return DynamicHMC.mcmc_with_warmup(rng, instance, nsteps; kws...)
end
hmc_sample(models::AbstractVector{T}, data::AbstractMatrix{<:Number}, nsteps::Integer;
           kws...) where T <: AbstractMatrix{<:Number} = hmc_sample(stack_models(models), vec(data), nsteps; kws...)

function extract_initialization(state)
    # This unpack is legal in Julia 1.7 but not 1.6; might be worth alteration
    (; Q, κ, ϵ) = state.final_warmup_state
    (; q = Q.q, κ, ϵ)
end

# Version with multiple chains and multithreading
function hmc_sample(models::AbstractMatrix{S}, data::AbstractVector{<:Number}, nsteps::Integer, nchains::Integer;
                    rng::AbstractRNG=default_rng(), initialization=(), kws...) where S <: Number
    @assert nchains ≥ 1 "`nchains` argument to `hmc_sample` must be ≥ 1."
    # Construct threadsafe buffer to hold HMCModel instances
    instances = TaskLocalValue{HMCModel}(() -> HMCModel(models, Vector{S}(undef, length(data)), data))
    # Do the warmup
    warmup = DynamicHMC.mcmc_keep_warmup(rng, instances[], 0;
                                         warmup_stages=DynamicHMC.default_warmup_stages(),
                                         initialization=initialization, kws...)
    final_init = extract_initialization(warmup)
    # Do the MCMC
    result_chnl = Channel(nchains) # Use a channel to collect results
    Threads.@threads for i in 1:nchains
        result = DynamicHMC.mcmc_with_warmup(rng, instances[], nsteps; warmup_stages=(),
                                             initialization=final_init, kws...)
        put!(result_chnl, result)
    end
    return [take!(result_chnl) for i in 1:nchains] # Take results from channel and plop into vector
end
hmc_sample(models::AbstractVector{T}, data::AbstractMatrix{<:Number}, nsteps::Integer, nchains::Integer;
           kws...) where {S <: Number, T <: AbstractMatrix{S}} = hmc_sample(stack_models(models), vec(data), nsteps, nchains; kws...)
