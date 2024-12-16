# Generic fitting and sampling methods that work on both AMR and MZR models

# Define struct to hold optimization-time constants
# needed to calculate the logL and gradient with fg!,
# used for sampling with DynamicHMC and it may also be easier
# to reuse for BFGS optimization with Optim.jl rather
# than rewriting the closure

struct HierarchicalOptimizer{A,B,C,D,E,F,G,H}
    Zmodel0::A
    dispmodel0::B
    models::C
    data::D
    composite::E
    logAge::F
    metallicities::G
    G::H
    jacobian_corrections::Bool # Whether or not to apply Jacobian corrections for variable transformations
end

# This model will return loglikelihood and gradient
LogDensityProblems.capabilities(::Type{<:HierarchicalOptimizer}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(problem::HierarchicalOptimizer) = length(problem.G)

"""
    fg!(F, G, Zmodel0::AbstractMetallicityModel,
        dispmodel0::AbstractDispersionModel,
        variables::AbstractVector{<:Number},
        models::Union{AbstractMatrix{<:Number},
                      AbstractVector{<:AbstractMatrix{<:Number}}},
        data::Union{AbstractVector{<:Number}, AbstractMatrix{<:Number}},
        composite::Union{AbstractVector{<:Number}, AbstractMatrix{<:Number}},
        logAge::AbstractVector{<:Number},
        metallicities::AbstractVector{<:Number})

Main function that differs between AMR and MZR models that accounts for the difference in the gradient formulations between models. `F` and `G` mirror the [Optim.jl](https://julianlsolvers.github.io/Optim.jl/stable/user/tipsandtricks/#Avoid-repeating-computations) interface for computing the objective and gradient in a single function to make use of common intermediate computations.

# Arguments
 - `F` controls whether the objective is being requested. If `!isnothing(F)`, the negative log likelihood will be returned from `fg!`.
 - `G` controls whether the gradient of the objective with respect to the `variables` is being requested. If `!isnothing(G)`, `G` will be updated in-place with the gradient of the negative log likelihood with respect to the fitting parameters `variables`.
 - `Zmodel0` is an instance of a concrete subtype of `AbstractMetallicityModel` (e.g., [`PowerLawMZR`](@ref)) that specifies the metallicity evolution model to use. The parameters contained in `Zmodel0` are used as initial parameters to begin the optimization in [`fit_sfh`](@ref), but are not used internally in `fg!` -- new instances are constructed from `variables` instead.
 - `dispmodel0` is an instance of a concrete subtype of `AbstractDispersionModel` (e.g., [`GaussianDispersion`](@ref)) that specifies the PDF of the metallicities of stars forming at fixed time. The parameters contained in `dispmodel0` are used as initial parameters to begin the optimization in [`fit_sfh`](@ref), but are not used internally in `fg!` -- new instances are constructed from `variables` instead.
 - `models` are the template Hess diagrams for the SSPs that compose the observed Hess diagram.
 - `data` is the Hess diagram for the observed data.
 - `composite` is the pre-allocated array in which to store the complex Hess diagram model. Must have same shape as `data`.
 - `logAge::AbstractVector{<:Number}` is the vector containing the effective ages of the stellar populations used to create the templates in `models`, in units of `log10(age [yr])`. For example, if a population has an age of 1 Myr, its entry in `logAge` should be `log10(10^6) = 6.0`.
 - `metallicities::AbstractVector{<:Number}` is the vector containing the effective metallicities of the stellar populations used to create the templates in `models`. This is most commonly a logarithmic abundance like [M/H] or [Fe/H], but you could use a linear abundance like the metal mass fraction Z if you wanted to. There are some notes on the [Wikipedia](https://en.wikipedia.org/wiki/Metallicity) that might be useful.

# Returns
 - Negative log likelihood if `!isnothing(F)`.
"""
function fg! end

function LogDensityProblems.logdensity_and_gradient(problem::HierarchicalOptimizer, xvec)
    # Unpack struct
    Zmodel0 = problem.Zmodel0
    dispmodel0 = problem.dispmodel0
    models = problem.models
    data = problem.data
    composite = problem.composite
    logAge = problem.logAge
    metallicities = problem.metallicities
    G = problem.G
    jacobian_corrections = problem.jacobian_corrections
    
    zpar = nparams(Zmodel0)
    disppar = nparams(dispmodel0)
    tf = SVector(transforms(Zmodel0)..., transforms(dispmodel0)...)
    free = SVector(free_params(Zmodel0)..., free_params(dispmodel0)...)
    Nfixed = count(~, free)
    @assert axes(G) == axes(xvec)
    # Calculate number of age bins from length of xvec and number of Zmodel, disp parameters
    Nbins = lastindex(xvec) - zpar - disppar
    # Subtract off fixed parameters that do not appear in xvec
    Nbins += Nfixed # Gives count of false entries
    # Extract Zmodel and disp parameters from xvec
    par = @view(xvec[Nbins+1:end])
    # Transform the provided x
    # All stellar mass coefficients are transformed as they must be > 0,
    # but Zmodel and dispersion model coefficients may or may not be similarly constrained.
    # Use the transforms() function to determine which parameters should be transformed.
    x = Vector{eltype(xvec)}(undef, Nbins + zpar + disppar)
    # These are the stellar mass coefficients
    for i in eachindex(xvec)[begin:Nbins]; x[i] = exp(xvec[i]); end
    # Apply logarithmic transformations
    x_zdisp = exptransform(par, SVector(tf)[free])
    # Concatenate transformed stellar mass coefficients and Zmodel / disp parameters
    x[(Nbins+1:lastindex(x))[free]] .= x_zdisp
    # Write fixed parameters into x
    init_par = SVector(fittable_params(Zmodel0)..., fittable_params(dispmodel0)...)
    fixed = .~free
    x[(Nbins+1:lastindex(x))[fixed]] .= init_par[fixed]


    G2 = similar(x)
    nlogL = fg!(true, G2, Zmodel0, dispmodel0, x, models, data, composite, logAge, metallicities)
    # Add Jacobian corrections for transformed variables if jacobian_corrections == true
    # fg! returns -logL and fills G with -∇logL, so remember to invert signs in Jacobian corrections
    ptf = findall(==(1), tf)  # Find indices of variables constrained to always be positive
    ptf = ptf[free[ptf]]      # Only keep indices for variables that we are fitting
    ptf_idx = ptf .+ Nbins
    ntf = findall(==(-1), tf) # Find indices of variables constrained to always be negative
    ntf = ntf[free[ntf]]      # Only keep indices for variables that we are fitting
    if jacobian_corrections
        # For positive-constrained parameters, including stellar mass coefficients
        for i in vcat(eachindex(G2)[begin:Nbins], ptf_idx)
            nlogL -= log(x[i])
            G2[i] = G2[i] * x[i] - 1
        end
        for i in ntf
            @warn "Negative transformations have not yet been validated."
            i += Nbins
            nlogL += log(x[i])
            G2[i] = -G2[i] * x[i] + 1
        end
    else
        # Still have to correct gradient for transform, even if not adding Jacobian correction to logL
        for i in vcat(eachindex(G2)[begin:Nbins], ptf_idx)
            G2[i] = G2[i] * x[i]
        end
        for i in ntf
            @warn "Negative transformations have not yet been validated."
            G2[i] = -G2[i] * x[i]
        end        
    end

    # Write gradient from G2 into G for free parameters
    for i in firstindex(G):Nbins; G[i] = G2[i]; end
    free_count = 1
    for i in 1:length(free)
        if free[i]
            G[Nbins+free_count] = G2[Nbins+i]
            free_count += 1
        end
    end
    
    # Optimizers and samplers honoring LogDensityProblems's API will expect positive logL and ∇logL,
    # not -logL and -∇logL as we have (nlogL and G), so return the negatives of these values.
    return -nlogL, -G
    
end

"""
    fit_sfh(Zmodel0::AbstractMetallicityModel,
            dispmodel0::AbstractDispersionModel,
            models::AbstractMatrix{<:Number},
            data::AbstractVector{<:Number},
            logAge::AbstractVector{<:Number},
            metallicities::AbstractVector{<:Number};
            x0::AbstractVector{<:Number} = <...>
            kws...)

Returns a [`CompositeBFGSResult`](@ref StarFormationHistories.CompositeBFGSResult) instance that contains the maximum a posteriori (MAP) and maximum likelihood estimates (MLE) obtained from fitting the provided simple stellar population (SSP) templates `models` (with logarithmic ages `logAge = log10(age [yr])` and metallicities `metallicities`) to the provided `data`. The metallicity evolution is modelled using the provided `Zmodel0`, whose parameters can be free or fixed, with metallicity dispersion at fixed time modelled by `dispmodel0`, whose parameters can be free or fixed.

This method is designed to work best with a grid of stellar models, defined by the outer product of `N` unique entries in `logAge` and `M` unique entries in `metallicities`. See the examples for more information on usage.

We provide several options for age-metallicity relations and mass-metallicity relations that can be used for `Zmodel0` and define APIs for users to create new models that will integrate with this function. Similar flexibility is allowed for the metallicity dispersion model `dispmodel0`.

The primary method signature uses flattened formats for `models` and `data`. See the notes for the flattened call signature of [`StarFormationHistories.composite!`](@ref) for more details, as well as [`stack_models`](@ref StarFormationHistories.stack_models) that facilitates rearranging the `models` into this flattened format.

# Arguments
 - `Zmodel0` is an instance of [`AbstractMetallicityModel`](@ref StarFormationHistories.AbstractMetallicityModel) that defines how the average metallicity stars being formed in the population changes over time. The fittable parameters contained in this instance are used as the initial values to start the optimization. 
 - `dispmodel0` is an instance of [`AbstractDispersionModel`](@ref StarFormationHistories.AbstractDispersionModel) that defines the distribution of metallicities of stars forming in a fixed time bin (i.e., the dispersion in metallicity around the mean at fixed time). The fittable parameters contained in this instance are used as the initial values to start the optimization. 
 - `models` are the template Hess diagrams for the SSPs that compose the observed Hess diagram.
 - `data` is the Hess diagram for the observed data.
 - `logAge::AbstractVector{<:Number}` is the vector containing the effective ages of the stellar populations used to create the templates in `models`, in units of `log10(age [yr])`. For example, if a population has an age of 1 Myr, its entry in `logAge` should be `log10(10^6) = 6.0`.
 - `metallicities::AbstractVector{<:Number}` is the vector containing the effective metallicities of the stellar populations used to create the templates in `models`. This is most commonly a logarithmic abundance like [M/H] or [Fe/H], but you could use a linear abundance like the metal mass fraction Z if you wanted to. There are some notes on the [Wikipedia](https://en.wikipedia.org/wiki/Metallicity) that might be useful.

# Keyword Arguments
 - `x0` is the vector of initial guesses for the stellar mass coefficients per *unique* entry in `logAge`. We try to set reasonable defaults, but in most cases users should be calculating and passing this keyword argument. We provide [`StarFormationHistories.construct_x0_mdf`](@ref) to prepare `x0` assuming a constant star formation rate and total stellar mass, which is typically a good initial guess.

# Returns
 - This function returns a [`CompositeBFGSResult`](@ref StarFormationHistories.CompositeBFGSResult) that contains the output from both MLE and MAP optimizations, accessible via `result.mle` and `result.map`. These are each instances of [`BFGSResult`](@ref StarFormationHistories.BFGSResult). See the docs for these structs for more information.
"""
function fit_sfh(Zmodel0::AbstractMetallicityModel{T}, dispmodel0::AbstractDispersionModel{U},
                 models::AbstractMatrix{S},
                 data::AbstractVector{<:Number},
                 logAge::AbstractVector{<:Number},
                 metallicities::AbstractVector{<:Number};
                 x0::AbstractVector{<:Number} = construct_x0_mdf(logAge, convert(S, 13.7); normalize_value=1e6),
                 kws...) where {T, U, S <: Number}

    unique_logAge = unique(logAge)
    Nbins = length(x0) # Number of unique logAge bins
    @assert length(x0) == length(unique_logAge)
    @assert size(models, 1) == length(data)
    @assert size(models, 2) == length(logAge) == length(metallicities)
    composite = Vector{S}(undef, length(data)) # Scratch matrix for storing complex Hess model
    # Perform logarithmic transformation on the provided x0 (stellar mass coefficients)
    x0 = map(log, x0) # Does not modify x0 in place
    # Perform logarithmic transformation on MZR and dispersion parameters
    par = (values(fittable_params(Zmodel0))..., values(fittable_params(dispmodel0))...)
    tf = (transforms(Zmodel0)..., transforms(dispmodel0)...)
    free = SVector(free_params(Zmodel0)..., free_params(dispmodel0)...)
    # Apply logarithmic transformations
    x0_mzrdisp = logtransform(par, tf)
    # Concatenate transformed stellar mass coefficients and *free* MZR / disp parameters
    x0 = vcat(x0, x0_mzrdisp[free])

    # Set up options for the optimization
    # The InitialStatic(1.0,true) alphaguess helps to regularize the optimization and 
    # makes it less sensitive to initial x0.
    bfgs_struct = Optim.BFGS(; alphaguess=LineSearches.InitialStatic(1.0, true),
                             linesearch=LineSearches.HagerZhang())
    # The extended trace will contain the BFGS estimate of the inverse Hessian, aka the
    # covariance matrix, which we can use to make parameter uncertainty estimates
    bfgs_options = Optim.Options(; allow_f_increases=true, store_trace=true, extended_trace=true, kws...)
    function fg_map!(F, G, X)
        # Creating structs doesn't copy data so this should be free
        tmpstruct = HierarchicalOptimizer(Zmodel0, dispmodel0, models, data, composite, logAge, metallicities, G, true)
        return -LogDensityProblems.logdensity_and_gradient(tmpstruct, X)[1]
    end
    function fg_mle!(F, G, X)
        # Creating structs doesn't copy data so this should be free
        tmpstruct = HierarchicalOptimizer(Zmodel0, dispmodel0, models, data, composite, logAge, metallicities, G, false)
        return -LogDensityProblems.logdensity_and_gradient(tmpstruct, X)[1]
    end
    result_map = Optim.optimize(Optim.only_fg!(fg_map!), x0, bfgs_struct, bfgs_options)
    result_mle = Optim.optimize(Optim.only_fg!(fg_mle!), Optim.minimizer(result_map), bfgs_struct, bfgs_options)
    
    # Random sampling from the inverse Hessian approximation to the Gaussian covariance
    # matrix will use Distributions.MvNormal, which requires a
    # PDMat input, which includes the Cholesky decomposition of the matrix.
    # For sampling efficiency, we will construct this object for the MAP result here.
    # For the MLE, cases can arise where many best-fit stellar mass coefficients are 0,
    # making invH poorly conditioned so that it is not useful for sampling. We will try to
    # construct the PDMat, but if we fail, we will simply save the raw matrix. This will
    # likely result in any calls like rand(result.mle) failing as result.mle.invH will not
    # be positive definite, but you should not really sample from the MLE anyway, and none
    # of our exposed methods use result.mle.invH, so this is acceptable.
    invH_map = PDMat(hermitianpart(Optim.trace(result_map)[end].metadata["~inv(H)"]))
    invH_mle = hermitianpart(Optim.trace(result_mle)[end].metadata["~inv(H)"])
    try
        invH_mle = PDMat(invH_mle)
    catch
        @debug "Inverse Hessian matrix of MLE estimate is not positive definite" invH_mle
    end
    # Allocate vectors to hold best-fit values μ and standard errors σ for all parameters,
    # including fixed parameters
    μ_map = similar(Optim.minimizer(result_map), Nbins + nparams(Zmodel0) + nparams(dispmodel0))
    μ_mle = similar(μ_map)
    σ_map = similar(invH_map, length(μ_map))
    σ_mle = similar(σ_map)
    # Collect values for only free variables
    μ_map_tmp = Optim.minimizer(result_map)
    μ_mle_tmp = Optim.minimizer(result_mle)
    # Diagonal of invH gives vector of parameter variances -- standard error is sqrt
    σ_map_tmp = sqrt.(diag(invH_map))
    σ_mle_tmp = sqrt.(diag(invH_mle))
    # Write stellar mass coefficients, with transformations applied
    for i in 1:Nbins 
        μ_map[i] = exp(μ_map_tmp[i])
        μ_mle[i] = exp(μ_mle_tmp[i])
        σ_map[i] = μ_map[i] * σ_map_tmp[i]
        σ_mle[i] = μ_mle[i] * σ_mle_tmp[i]
    end
    for i in Nbins+1:length(μ_map)
        if free[i-Nbins] # Parameter is free; need to check for transformations
            tfi = tf[i-Nbins]
            if tfi == 1
                μ_map[i] = exp(μ_map_tmp[i])
                μ_mle[i] = exp(μ_mle_tmp[i])
                σ_map[i] = μ_map[i] * σ_map_tmp[i]
                σ_mle[i] = μ_map[i] * σ_mle_tmp[i]
            elseif tfi == 0
                μ_map[i] = μ_map_tmp[i]
                μ_mle[i] = μ_mle_tmp[i]
                σ_map[i] = σ_map_tmp[i]
                σ_mle[i] = σ_mle_tmp[i]
            elseif tfi == -1
                μ_map[i] = -exp(μ_map_tmp[i])
                μ_mle[i] = -exp(μ_mle_tmp[i])
                σ_map[i] = -μ_map[i] * σ_map_tmp[i]
                σ_mle[i] = -μ_map[i] * σ_mle_tmp[i]
            end
        else # Parameter is fixed; no transformation applied
            μ_map[i] = par[i-Nbins]
            μ_mle[i] = par[i-Nbins]
            σ_map[i] = 0
            σ_mle[i] = 0
        end
    end
    
    return CompositeBFGSResult( BFGSResult(μ_map, σ_map, invH_map, result_map,
                                           update_params(Zmodel0, @view(μ_map[Nbins+1:Nbins+nparams(Zmodel0)])),
                                           update_params(dispmodel0, @view(μ_map[Nbins+nparams(Zmodel0)+1:end]))),
                                BFGSResult(μ_mle, σ_mle, invH_mle, result_mle,
                                           update_params(Zmodel0, @view(μ_mle[Nbins+1:Nbins+nparams(Zmodel0)])),
                                           update_params(dispmodel0, @view(μ_mle[Nbins+nparams(Zmodel0)+1:end]))) )
    
end
# For models, data that do not follow the stacked data layout (see stack_models in fitting/utilities.jl)
fit_sfh(Zmodel0::AbstractMetallicityModel, dispmodel0::AbstractDispersionModel, models::AbstractVector{<:AbstractMatrix{<:Number}}, data::AbstractMatrix{<:Number}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}; kws...) = fit_sfh(Zmodel0, dispmodel0, stack_models(models), vec(data), logAge, metallicities; kws...)



# HMC sampling routine; uses stacked data layout
# Use BFGS result to get initial position, Gaussian kinetic energy matrix
function sample_sfh(bfgs_result::CompositeBFGSResult, 
                    models::AbstractMatrix{S},
                    data::AbstractVector{<:Number},
                    logAge::AbstractVector{<:Number},
                    metallicities::AbstractVector{<:Number},
                    Nsteps::Integer;
                    ϵ::Real = 0.05, # HMC step size
                    reporter = DynamicHMC.ProgressMeterReport(),
                    show_convergence::Bool=true,
                    composite::AbstractVector{<:Number}=Vector{S}(undef,length(data)),
                    rng::AbstractRNG=default_rng(),
                    kws...) where {S <: Number}

    # Will use MLE for best-fit values, MAP for invH
    MAP, MLE = bfgs_result.map, bfgs_result.mle
    # Best-fit free parameter values from optimization in transformed fitting variables
    x0 = Optim.minimizer(MLE.result)
    # Best-fit all parameters (fixed included)
    μ = MLE.μ
    Zmodel, dispmodel = MLE.Zmodel, MLE.dispmodel
    
    # Get transformation parameters
    tf = (transforms(Zmodel)..., transforms(dispmodel)...)
    free = SVector(free_params(Zmodel)..., free_params(dispmodel)...)

    # Setup structs to pass to DynamicHMC.mcmc
    instance = HierarchicalOptimizer(Zmodel, dispmodel, models, data, composite, logAge, metallicities,
                            similar(x0), true)
    # The call signature for the kinetic energy is κ = DynamicHMC.GaussianKineticEnergy(M⁻¹),
    # where M is the mass matrix (e.g., equation 5.5 in "Handbook of Markov Chain Monte Carlo").
    # As explained in section 5.4.1 of that text, on pg 134, if you have an estimate for the covariance
    # matrix of the fitting variables Σ (in our case, the inverse Hessian), you can improve the efficiency
    # of the HMC sampling by setting M⁻¹ to Σ, which is what we do here.
    warmup_state = DynamicHMC.initialize_warmup_state(rng, instance;
        q = x0, # Initial position vector
        κ = DynamicHMC.GaussianKineticEnergy(MAP.invH), # Kinetic energy
        ϵ = ϵ) # HMC step size
    sampling_logdensity = DynamicHMC.SamplingLogDensity(rng, instance, DynamicHMC.NUTS(), reporter)
    
    # Sample
    result = DynamicHMC.mcmc(sampling_logdensity, Nsteps, warmup_state)

    # Test convergence
    tree_stats = DynamicHMC.Diagnostics.summarize_tree_statistics(result.tree_statistics)
    show_convergence && display(tree_stats)
    if tree_stats.a_mean < 0.8
        @warn "Acceptance ratio for samples less than 80%, recommend re-running with smaller step size ϵ."
    end
    if tree_stats.termination_counts.divergence > (0.1 * Nsteps)
        @warn "More than 10% of samples diverged, recommend re-running with smaller step size ϵ."
    end

    # Transform samples
    Nbins = length(μ) - nparams(Zmodel) - nparams(dispmodel)
    # Get indices into μ corresponding to free parameters
    row_idxs = vcat(1:Nbins, (Nbins+1:Nbins+length(free))[free])
    exptransform_samples!(result.posterior_matrix, μ[row_idxs], tf[free], free[free])

    # Now we need to expand posterior_samples to include fixed parameters as well
    if false in free
        samples = similar(result.posterior_matrix, (length(μ), Nsteps))
        samples[row_idxs, :] .= result.posterior_matrix
        # Now write in fixed parameters
        par = (values(fittable_params(Zmodel))..., values(fittable_params(dispmodel))...)
        for i in 1:length(free)
            if ~free[i] # if parameter is fixed,
                samples[Nbins+i, :] .= par[i]
            end
        end
        result = (posterior_matrix = samples, tree_statistics = result.tree_statistics)
    end
    return result
end
# For models, data that do not follow the stacked data layout (see stack_models in fitting/utilities.jl)
sample_sfh(bfgs_result::CompositeBFGSResult, models::AbstractVector{<:AbstractMatrix{<:Number}}, data::AbstractMatrix{<:Number}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}, Nsteps::Integer; kws...) = sample_sfh(bfgs_result, stack_models(models), vec(data), logAge, metallicities, Nsteps; kws...)
