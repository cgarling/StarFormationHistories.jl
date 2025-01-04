# Generic fitting and sampling methods that work on both AMR and MZR models

"""
    calculate_coeffs(MH_model::AbstractMetallicityModel,
                     disp_model::AbstractDispersionModel,
                     mstars::AbstractVector{<:Number}, 
                     logAge::AbstractVector{<:Number},
                     metallicities::AbstractVector{<:Number})

Returns per-SSP stellar mass coefficients (``r_{j,k}`` in the [derivation](@ref mzr_derivation)) using the provided metallicity model `MH_model` and metallicity dispersion model `disp_model` for the set of SSPs with logarithmic ages `logAge` and metallicities `metallicities`.

# Examples
```jldoctest; setup = :(import StarFormationHistories: calculate_coeffs, PowerLawMZR, GaussianDispersion)
julia> n_logage, n_mh = 10, 20; # Number of unique logAges, MHs

julia> coeffs = calculate_coeffs(PowerLawMZR(1.0, -1.0),
                                 GaussianDispersion(0.2),
                                 rand(n_logage),
                                 repeat(range(7.0, 10.0; length=n_logage); inner=n_mh),
                                 repeat(range(-2.0, 0.0; length=n_mh); outer=n_logage));

julia> coeffs isa Vector{Float64}
true

julia> length(coeffs) == n_logage * n_mh
true
```
"""
function calculate_coeffs end
# function calculate_coeffs(MH_model::AbstractMetallicityModel,
#                           disp_model::AbstractDispersionModel,
#                           mstars::AbstractVector{<:Number},
#                           logAge::AbstractVector<:Number},
#                           metallicities::AbstractVector{<:Number}) end
# calculate_coeffs(::StarFormationHistories.AbstractMetallicityModel, ::StarFormationHistories.AbstractDispersionModel, ::AbstractVector{<:Number}, ::AbstractVector{<:Number}, ::AbstractVector{<:Number})


# Define struct to hold optimization-time constants
# needed to calculate the logL and gradient with fg!,
# used for sampling with DynamicHMC and it may also be easier
# to reuse for BFGS optimization with Optim.jl rather
# than rewriting the closure

struct HierarchicalOptimizer{A,B,C,D,E,F,G,H,I}
    MH_model0::A
    disp_model0::B
    models::C
    data::D
    composite::E
    logAge::F
    metallicities::G
    F::H
    G::I
    jacobian_corrections::Bool # Whether or not to apply Jacobian corrections for variable transformations
end

# This model will return loglikelihood and gradient
LogDensityProblems.capabilities(::Type{<:HierarchicalOptimizer}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(problem::HierarchicalOptimizer) = length(problem.G)

"""
    fg!(F, G, MH_model0::AbstractMetallicityModel,
        disp_model0::AbstractDispersionModel,
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
 - `MH_model0` is an instance of a concrete subtype of `AbstractMetallicityModel` (e.g., [`PowerLawMZR`](@ref)) that specifies the metallicity evolution model to use. The parameters contained in `MH_model0` are used as initial parameters to begin the optimization in [`fit_sfh`](@ref), but are not used internally in `fg!` -- new instances are constructed from `variables` instead.
 - `disp_model0` is an instance of a concrete subtype of `AbstractDispersionModel` (e.g., [`GaussianDispersion`](@ref)) that specifies the PDF of the metallicities of stars forming at fixed time. The parameters contained in `disp_model0` are used as initial parameters to begin the optimization in [`fit_sfh`](@ref), but are not used internally in `fg!` -- new instances are constructed from `variables` instead.
 - `variables` are the fitting parameters, including *free and fixed* parameters. This vector is split into stellar mass coefficients `R_j`, metallicity model parameters, and dispersion model parameters, and so must contain all relevant fittable parameters, even those that are to be fixed during the solve.
 - `models` are the template Hess diagrams for the SSPs that compose the observed Hess diagram.
 - `data` is the Hess diagram for the observed data.
 - `composite` is the pre-allocated array in which to store the complex Hess diagram model. Must have same shape as `data`.
 - `logAge::AbstractVector{<:Number}` is the vector containing the effective ages of the stellar populations used to create the templates in `models`, in units of `log10(age [yr])`. For example, if a population has an age of 1 Myr, its entry in `logAge` should be `log10(10^6) = 6.0`.
 - `metallicities::AbstractVector{<:Number}` is the vector containing the effective metallicities of the stellar populations used to create the templates in `models`. These should be logarithmic abundances like [M/H] or [Fe/H]. There are some notes on the [Wikipedia](https://en.wikipedia.org/wiki/Metallicity) that might be useful.

# Returns
 - Negative log likelihood if `!isnothing(F)`.
"""
function fg! end

function LogDensityProblems.logdensity_and_gradient(problem::HierarchicalOptimizer, xvec)
    # Unpack struct
    MH_model0 = problem.MH_model0
    disp_model0 = problem.disp_model0
    models = problem.models
    data = problem.data
    composite = problem.composite
    logAge = problem.logAge
    metallicities = problem.metallicities
    F, G = problem.F, problem.G
    # Return F and G if they are not nothing
    ret_F, ret_G = !isnothing(F), !isnothing(G)
    jacobian_corrections = problem.jacobian_corrections
    
    zpar = nparams(MH_model0)
    disppar = nparams(disp_model0)
    tf = SVector(transforms(MH_model0)..., transforms(disp_model0)...)
    free = SVector(free_params(MH_model0)..., free_params(disp_model0)...)
    Nfixed = count(~, free)
    # Calculate number of age bins from length of xvec and number of MH_model, disp parameters
    Nbins = lastindex(xvec) - zpar - disppar
    # Subtract off fixed parameters that do not appear in xvec
    Nbins += Nfixed # Gives count of false entries
    # Extract MH_model and disp parameters from xvec
    par = @view(xvec[Nbins+1:end])
    # Transform the provided x
    # All stellar mass coefficients are transformed as they must be > 0,
    # but MH_model and dispersion model coefficients may or may not be similarly constrained.
    # Use the transforms() function to determine which parameters should be transformed.
    x = similar(xvec, Nbins + zpar + disppar)
    # These are the stellar mass coefficients
    for i in eachindex(xvec)[begin:Nbins]; x[i] = exp(xvec[i]); end
    # Apply logarithmic transformations
    x_zdisp = exptransform(par, SVector(tf)[free])
    # Concatenate transformed stellar mass coefficients and MH_model / disp parameters
    x[(Nbins+1:lastindex(x))[free]] .= x_zdisp
    # Write fixed parameters into x
    init_par = SVector(fittable_params(MH_model0)..., fittable_params(disp_model0)...)
    fixed = .~free
    x[(Nbins+1:lastindex(x))[fixed]] .= init_par[fixed]

    # If we need to calculate and return gradient, allocate extended gradient vector
    ret_G ? G2 = similar(x) : G2 = nothing
    nlogL = fg!(F, G2, MH_model0, disp_model0, x, models, data, composite, logAge, metallicities)
    # Add Jacobian corrections for transformed variables if jacobian_corrections == true
    # fg! returns -logL and fills G with -∇logL, so remember to invert signs in Jacobian corrections
    ptf = findall(==(1), tf)  # Find indices of variables constrained to always be positive
    ptf = ptf[free[ptf]]      # Only keep indices for variables that we are fitting
    ptf_idx = ptf .+ Nbins
    ntf = findall(==(-1), tf) # Find indices of variables constrained to always be negative
    ntf = ntf[free[ntf]]      # Only keep indices for variables that we are fitting
    if jacobian_corrections
        # For positive-constrained parameters, including stellar mass coefficients
        for i in vcat(eachindex(x)[begin:Nbins], ptf_idx)
            if ret_F; nlogL -= log(x[i]); end
            if ret_G; G2[i] = G2[i] * x[i] - 1; end
        end
        for i in ntf
            @warn "Negative transformations have not yet been validated."
            i += Nbins
            if ret_F; nlogL += log(x[i]); end
            if ret_G; G2[i] = -G2[i] * x[i] + 1; end
        end
    else
        # Still have to correct gradient for transform, even if not adding Jacobian correction to logL
        for i in vcat(eachindex(x)[begin:Nbins], ptf_idx)
            if ret_G; G2[i] = G2[i] * x[i]; end
        end
        for i in ntf
            @warn "Negative transformations have not yet been validated."
            if ret_G; G2[i] = -G2[i] * x[i]; end
        end        
    end

    # Return if gradient is not requested
    if ~ret_G
        if ret_F
            return -nlogL
        else
            return
        end
    end
    
    # Write gradient from G2 into G for free parameters
    @assert axes(G) == axes(xvec)
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
    if ret_F
        return -nlogL, -G
    else
        return -G
    end
    
end

"""
    fit_sfh(MH_model0::AbstractMetallicityModel,
            disp_model0::AbstractDispersionModel,
            models::AbstractMatrix{<:Number},
            data::AbstractVector{<:Number},
            logAge::AbstractVector{<:Number},
            metallicities::AbstractVector{<:Number};
            x0::AbstractVector{<:Number} = <...>
            kws...)
    fit_sfh(MH_model0::AbstractMetallicityModel,
            disp_model0::AbstractDispersionModel,
            models::AbstractVector{<:AbstractMatrix{<:Number}},
            data::AbstractMatrix{<:Number},
            logAge::AbstractVector{<:Number},
            metallicities::AbstractVector{<:Number};
            x0::AbstractVector{<:Number} = <...>
            kws...)


Returns a [`CompositeBFGSResult`](@ref StarFormationHistories.CompositeBFGSResult) instance that contains the maximum a posteriori (MAP) and maximum likelihood estimates (MLE) obtained from fitting the provided simple stellar population (SSP) templates `models` (with logarithmic ages `logAge = log10(age [yr])` and metallicities `metallicities`) to the provided `data`. The metallicity evolution is modelled using the provided `MH_model0`, whose parameters can be free or fixed, with metallicity dispersion at fixed time modelled by `disp_model0`, whose parameters can be free or fixed.

This method is designed to work best with a grid of stellar models, defined by the outer product of `N` unique entries in `logAge` and `M` unique entries in `metallicities`. See the examples for more information on usage.

We provide several options for age-metallicity relations and mass-metallicity relations that can be used for `MH_model0` and define APIs for users to create new models that will integrate with this function. Similar flexibility is allowed for the metallicity dispersion model `disp_model0`.

The primary method signature uses flattened formats for `models` and `data`. See the notes for the flattened call signature of [`StarFormationHistories.composite!`](@ref) for more details, as well as [`stack_models`](@ref StarFormationHistories.stack_models) that facilitates rearranging the `models` into this flattened format.

# Arguments
 - `MH_model0` is an instance of [`AbstractMetallicityModel`](@ref StarFormationHistories.AbstractMetallicityModel) that defines how the average metallicity stars being formed in the population changes over time. The fittable parameters contained in this instance are used as the initial values to start the optimization. 
 - `disp_model0` is an instance of [`AbstractDispersionModel`](@ref StarFormationHistories.AbstractDispersionModel) that defines the distribution of metallicities of stars forming in a fixed time bin (i.e., the dispersion in metallicity around the mean at fixed time). The fittable parameters contained in this instance are used as the initial values to start the optimization. 
 - `models` are the template Hess diagrams for the SSPs that compose the observed Hess diagram.
 - `data` is the Hess diagram for the observed data.
 - `logAge::AbstractVector{<:Number}` is the vector containing the effective ages of the stellar populations used to create the templates in `models`, in units of `log10(age [yr])`. For example, if a population has an age of 1 Myr, its entry in `logAge` should be `log10(10^6) = 6.0`.
 - `metallicities::AbstractVector{<:Number}` is the vector containing the effective metallicities of the stellar populations used to create the templates in `models`. These should be logarithmic abundances like [M/H] or [Fe/H]. There are some notes on the [Wikipedia](https://en.wikipedia.org/wiki/Metallicity) that might be useful.
# Keyword Arguments
 - `x0` is the vector of initial guesses for the stellar mass coefficients per *unique* entry in `logAge`. We try to set reasonable defaults, but in most cases users should be calculating and passing this keyword argument. We provide [`StarFormationHistories.construct_x0_mdf`](@ref) to prepare `x0` assuming a constant star formation rate and total stellar mass, which is typically a good initial guess.
 - `kws...` are passed to `Optim.Options` and can be used to control tolerances for convergence.

# Returns
 - This function returns a [`CompositeBFGSResult`](@ref StarFormationHistories.CompositeBFGSResult) that contains the output from both MLE and MAP optimizations, accessible via `result.mle` and `result.map`. These are each instances of [`BFGSResult`](@ref StarFormationHistories.BFGSResult). See the docs for these structs for more information.
"""
function fit_sfh(MH_model0::AbstractMetallicityModel{T}, disp_model0::AbstractDispersionModel{U},
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
    par = (values(fittable_params(MH_model0))..., values(fittable_params(disp_model0))...)
    tf = (transforms(MH_model0)..., transforms(disp_model0)...)
    free = SVector(free_params(MH_model0)..., free_params(disp_model0)...)
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
        tmpstruct = HierarchicalOptimizer(MH_model0, disp_model0, models, data, composite, logAge, metallicities, F, G, true)
        return -LogDensityProblems.logdensity_and_gradient(tmpstruct, X)[1]
    end
    function fg_mle!(F, G, X)
        # Creating structs doesn't copy data so this should be free
        tmpstruct = HierarchicalOptimizer(MH_model0, disp_model0, models, data, composite, logAge, metallicities, F, G, false)
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
    μ_map = similar(Optim.minimizer(result_map), Nbins + nparams(MH_model0) + nparams(disp_model0))
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
                                           update_params(MH_model0, @view(μ_map[Nbins+1:Nbins+nparams(MH_model0)])),
                                           update_params(disp_model0, @view(μ_map[Nbins+nparams(MH_model0)+1:end]))),
                                BFGSResult(μ_mle, σ_mle, invH_mle, result_mle,
                                           update_params(MH_model0, @view(μ_mle[Nbins+1:Nbins+nparams(MH_model0)])),
                                           update_params(disp_model0, @view(μ_mle[Nbins+nparams(MH_model0)+1:end]))) )
    
end
# For models, data that do not follow the stacked data layout (see stack_models in fitting/utilities.jl)
fit_sfh(MH_model0::AbstractMetallicityModel, disp_model0::AbstractDispersionModel, models::AbstractVector{<:AbstractMatrix{<:Number}}, data::AbstractMatrix{<:Number}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}; kws...) = fit_sfh(MH_model0, disp_model0, stack_models(models), vec(data), logAge, metallicities; kws...)



# HMC sampling routine; uses stacked data layout
# Use BFGS result to get initial position, Gaussian kinetic energy matrix
"""
    sample_sfh(bfgs_result::CompositeBFGSResult, 
               models::AbstractMatrix{<:Number},
               data::AbstractVector{<:Number},
               logAge::AbstractVector{<:Number},
               metallicities::AbstractVector{<:Number},
               Nsteps::Integer;
               ϵ::Real = 0.05, # HMC step size
               reporter = DynamicHMC.ProgressMeterReport(),
               show_convergence::Bool=true,
               rng::AbstractRNG=default_rng())
    sample_sfh(bfgs_result::CompositeBFGSResult, 
               models::AbstractVector{<:AbstractMatrix{<:Number}},
               data::AbstractMatrix{<:Number},
               logAge::AbstractVector{<:Number},
               metallicities::AbstractVector{<:Number},
               Nsteps::Integer;
               ϵ::Real = 0.05, # HMC step size
               reporter = DynamicHMC.ProgressMeterReport(),
               show_convergence::Bool=true,
               rng::AbstractRNG=default_rng())

Takes the SFH fitting result in `bfgs_result` and uses it to initialize the Hamiltonian Monte Carlo (HMC) sampler from DynamicHMC.jl to sample `Nsteps` independent draws from the posterior.

The primary method signature uses flattened formats for `models` and `data`. See the notes for the flattened call signature of [`StarFormationHistories.composite!`](@ref) for more details, as well as [`stack_models`](@ref StarFormationHistories.stack_models) that facilitates rearranging the `models` into this flattened format.

# Arguments
 - `models, data, logAge, metallicities` are as in [`fit_sfh`](@ref).
 - `Nsteps` is the number of Monte Carlo samples you want to draw.

# Keyword Arguments
 - `ϵ` is the HMC step size. Convergence of the HMC samples is checked after sampling and if a convergence warning is issued, you should decrease this value.
 - `reporter` is a valid reporter type from DynamicHMC.jl, either `NoProgressReport`, `ProgressMeterReport` for a basic progress meter, or `LogProgressReport` for more detailed reporting.
 - `show_convergence` if `true`, will send sample convergence statistics to the default display.
 - `rng` is a `Random.AbstractRNG` sampler instance that will be used when generating the random samples.

# Returns
A `NamedTuple` with two elements:
 - `posterior_matrix` is a `Matrix` with dimensions `(npar, Nsteps)` where `npar` is the number of fitting variables in the problem and is `npar = length(bfgs_result.mle.μ)`. Each column is one independent sample.
 - `tree_statistics` contains convergence statistics that can be viewed with `DynamicHMC.Diagnostics.summarize_tree_statistics`.

# See also
 - [`tsample_sfh`(@ref StarFormationHistories.tsample_sfh) for multi-threaded version.
"""
function sample_sfh(bfgs_result::CompositeBFGSResult, 
                    models::AbstractMatrix{S},
                    data::AbstractVector{<:Number},
                    logAge::AbstractVector{<:Number},
                    metallicities::AbstractVector{<:Number},
                    Nsteps::Integer;
                    ϵ::Real = 0.05, # HMC step size
                    reporter = DynamicHMC.ProgressMeterReport(),
                    show_convergence::Bool=true,
                    # composite::AbstractVector{<:Number}=similar(data, S),
                    rng::AbstractRNG=default_rng()) where {S <: Number}

    # Will use MLE for best-fit values, MAP for invH
    MAP, MLE = bfgs_result.map, bfgs_result.mle
    # Best-fit free parameter values from optimization in transformed fitting variables
    x0 = Optim.minimizer(MLE.result)
    # Best-fit all parameters (fixed included)
    μ = MLE.μ
    MH_model, disp_model = MLE.MH_model, MLE.disp_model
    
    # Get transformation parameters
    tf = (transforms(MH_model)..., transforms(disp_model)...)
    free = SVector(free_params(MH_model)..., free_params(disp_model)...)

    composite = similar(data, S)
    # Setup structs to pass to DynamicHMC.mcmc
    instance = HierarchicalOptimizer(MH_model, disp_model, models, data, composite, logAge, metallicities,
                            true, similar(x0), true)
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
    Nbins = length(μ) - nparams(MH_model) - nparams(disp_model)
    # Get indices into μ corresponding to free parameters
    row_idxs = vcat(1:Nbins, (Nbins+1:Nbins+length(free))[free])
    exptransform_samples!(result.posterior_matrix, μ[row_idxs], tf[free], free[free])

    # Now we need to expand posterior_samples to include fixed parameters as well
    if false in free
        samples = similar(result.posterior_matrix, (length(μ), Nsteps))
        samples[row_idxs, :] .= result.posterior_matrix
        # Now write in fixed parameters
        par = (values(fittable_params(MH_model))..., values(fittable_params(disp_model))...)
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
sample_sfh(bfgs_result::CompositeBFGSResult, models::AbstractVector{<:AbstractMatrix{<:Number}}, data::AbstractMatrix{<:Number}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}, Nsteps::Integer) = sample_sfh(bfgs_result, stack_models(models), vec(data), logAge, metallicities, Nsteps)

# factor ~2.7 reduced runtime for 1 -> 4 threads
# no real performance hit when nthreads=1 either
# performance poorer on m2 with 8 threads than 4;
# use of the 4 efficiency cores bottlenecks computation
# because the load is statically balanced
"""
    tsample_sfh(bfgs_result::CompositeBFGSResult, 
                models::AbstractMatrix{<:Number},
                data::AbstractVector{<:Number},
                logAge::AbstractVector{<:Number},
                metallicities::AbstractVector{<:Number},
                Nsteps::Integer;
                ϵ::Real = 0.05, # HMC step size
                reporter = DynamicHMC.ProgressMeterReport(),
                show_convergence::Bool=true,
                rng::AbstractRNG=default_rng())
    tsample_sfh(bfgs_result::CompositeBFGSResult, 
                models::AbstractVector{<:AbstractMatrix{<:Number}},
                data::AbstractMatrix{<:Number},
                logAge::AbstractVector{<:Number},
                metallicities::AbstractVector{<:Number},
                Nsteps::Integer;
                ϵ::Real = 0.05, # HMC step size
                reporter = DynamicHMC.ProgressMeterReport(),
                show_convergence::Bool=true,
                rng::AbstractRNG=default_rng())

Multi-threaded version of [`sample_sfh`](@ref); see that method's documentation for details. The requested Monte Carlo samples `Nsteps` are split equally between Julia threads and are combined before being returned. As work is divided statically and equally amongst available threads, and this function blocks until all threads are returned, this function's runtime is limited by the slowest available thread. In architectures with inhomogenous cores, e.g., "performance" and "efficiency" cores as used in Apple M-series chips, this function will often perform better when limiting Julia threads to the number of available performance cores (e.g., using 4 threads on Apple M2 which has 4 performance cores and 4 efficiency cores by starting Julia with `julia -t 4`).  
"""
function tsample_sfh(bfgs_result::CompositeBFGSResult, 
                     models::AbstractMatrix{S},
                     data::AbstractVector{<:Number},
                     logAge::AbstractVector{<:Number},
                     metallicities::AbstractVector{<:Number},
                     Nsteps::Integer;
                     ϵ::Real = 0.05, # HMC step size
                     reporter = DynamicHMC.ProgressMeterReport(),
                     show_convergence::Bool=true,
                     # composite::AbstractVector{<:Number}=similar(data, S),
                     rng::AbstractRNG=default_rng()) where {S <: Number}

    Nthreads = Threads.nthreads()
    @assert Nsteps ≥ Nthreads "`tsample_sfh` requires you request at least as many samples as available threads (`Nsteps > Threads.nthreads`)."
    # Will use MLE for best-fit values, MAP for invH
    MAP, MLE = bfgs_result.map, bfgs_result.mle
    # Best-fit free parameter values from optimization in transformed fitting variables
    x0 = Optim.minimizer(MLE.result)
    # Best-fit all parameters (fixed included)
    μ = MLE.μ
    MH_model, disp_model = MLE.MH_model, MLE.disp_model
    
    # Get transformation parameters
    tf = (transforms(MH_model)..., transforms(disp_model)...)
    free = SVector(free_params(MH_model)..., free_params(disp_model)...)

    # Set up places to write results into
    posterior_matrices = Matrix{eltype(μ)}(undef, length(μ), Nsteps)
    tree_statistics = Vector{DynamicHMC.TreeStatisticsNUTS}(undef, Nsteps)
    # # Calculate number of steps to take in each thread, accounting for uneven remainder
    # # This can be replaced by Iterators.partition
    # tsteps = repeat([Nsteps ÷ Nthreads], Nthreads)
    # tsteps[end] += Nsteps % Nthreads
    # cum_tsteps = cumsum(tsteps)
    # idxs_all = vcat([range(1,tsteps[1])],
    #                 [range(cum_tsteps[i-1]+1,cum_tsteps[i]) for i in 2:(Nthreads-1)],
    #                 [range(cum_tsteps[end-1]+1, Nsteps)])
    # cld is integer division rounded up = div(x, y, RoundUp)
    idxs_all = collect(Iterators.partition(1:Nsteps, cld(Nsteps, Nthreads)))

    # Disable BLAS threading
    bthreads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    
    Threads.@threads for i in 1:Nthreads
        composite = similar(data, S)
        idxs = idxs_all[i]
        # Setup structs to pass to DynamicHMC.mcmc
        instance = HierarchicalOptimizer(MH_model, disp_model, models, data, composite, logAge, metallicities,
                                         true, similar(x0), true)
        # The call signature for the kinetic energy is κ = DynamicHMC.GaussianKineticEnergy(M⁻¹),
        # where M is the mass matrix (e.g., equation 5.5 in "Handbook of Markov Chain Monte Carlo").
        # As explained in section 5.4.1 of that text, on pg 134, if you have an estimate for the covariance
        # matrix of the fitting variables Σ (in our case, the inverse Hessian), you can improve the efficiency
        # of the HMC sampling by setting M⁻¹ to Σ, which is what we do here.
        warmup_state = DynamicHMC.initialize_warmup_state(rng, instance;
                                                          q = x0, # Initial position vector
                                                          κ = DynamicHMC.GaussianKineticEnergy(MAP.invH), # Kinetic energy
                                                          ϵ = ϵ) # HMC step size
        # Only use reporter on first thread
        ireporter = i == 1 ? reporter : DynamicHMC.NoProgressReport()
        sampling_logdensity = DynamicHMC.SamplingLogDensity(rng, instance, DynamicHMC.NUTS(), ireporter)
        
        # Sample
        result = DynamicHMC.mcmc(sampling_logdensity, length(idxs), warmup_state)
        # result = DynamicHMC.mcmc(sampling_logdensity, tsteps[i], warmup_state)
        # Write into shared output array
        # if i == 1
        #     idxs = range(1, tsteps[1])
        # elseif i == Nthreads
        #     idxs = range(cum_tsteps[end-1]+1, Nsteps)
        # else
        #     idxs = range(cum_tsteps[i-1]+1, cum_tsteps[i])
        # end
        posterior_matrices[:, idxs] .= result.posterior_matrix
        tree_statistics[idxs] .= result.tree_statistics
    end
    result = (posterior_matrix = posterior_matrices, tree_statistics = tree_statistics)
    BLAS.set_num_threads(bthreads)

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
    Nbins = length(μ) - nparams(MH_model) - nparams(disp_model)
    # Get indices into μ corresponding to free parameters
    row_idxs = vcat(1:Nbins, (Nbins+1:Nbins+length(free))[free])
    exptransform_samples!(result.posterior_matrix, μ[row_idxs], tf[free], free[free])

    # Now we need to expand posterior_samples to include fixed parameters as well
    if false in free
        samples = similar(result.posterior_matrix, (length(μ), Nsteps))
        samples[row_idxs, :] .= result.posterior_matrix
        # Now write in fixed parameters
        par = (values(fittable_params(MH_model))..., values(fittable_params(disp_model))...)
        for i in 1:length(free)
            if ~free[i] # if parameter is fixed,
                samples[Nbins+i, :] .= par[i]
            end
        end
        result = (posterior_matrix = samples, tree_statistics = result.tree_statistics)
    end
    return result
end
# function tsample_sfh(bfgs_result::CompositeBFGSResult, 
#                      models::AbstractMatrix{S},
#                      data::AbstractVector{<:Number},
#                      logAge::AbstractVector{<:Number},
#                      metallicities::AbstractVector{<:Number},
#                      Nsteps::Integer;
#                      ϵ::Real = 0.05, # HMC step size
#                      reporter = DynamicHMC.ProgressMeterReport(),
#                      show_convergence::Bool=true,
#                      # composite::AbstractVector{<:Number}=similar(data, S),
#                      rng::AbstractRNG=default_rng()) where {S <: Number}

#     Nthreads = Threads.nthreads()
#     @assert Nsteps ≥ Nthreads "`tsample_sfh` requires you request at least as many samples as available threads (`Nsteps > Threads.nthreads`)."
#     # Will use MLE for best-fit values, MAP for invH
#     MAP, MLE = bfgs_result.map, bfgs_result.mle
#     # Best-fit free parameter values from optimization in transformed fitting variables
#     x0 = Optim.minimizer(MLE.result)
#     # Best-fit all parameters (fixed included)
#     μ = MLE.μ
#     MH_model, disp_model = MLE.MH_model, MLE.disp_model
    
#     # Get transformation parameters
#     tf = (transforms(MH_model)..., transforms(disp_model)...)
#     free = SVector(free_params(MH_model)..., free_params(disp_model)...)

#     # Set up places to write results into
#     posterior_matrices = Matrix{eltype(μ)}(undef, length(μ), Nsteps)
#     tree_statistics = Vector{DynamicHMC.TreeStatisticsNUTS}(undef, Nsteps)
#     # # Calculate number of steps to take in each thread, accounting for uneven remainder
#     # tsteps = repeat([Nsteps ÷ Nthreads], Nthreads)
#     # tsteps[end] += Nsteps % Nthreads
#     # cum_tsteps = cumsum(tsteps)
#     # # idxs_all = vcat([range(1,tsteps[1])],
#     # #                 [range(cum_tsteps[i-1]+1,cum_tsteps[i]) for i in 2:(Nthreads-1)],
#     # #                 [range(cum_tsteps[end-1]+1, Nsteps)])

#     # Disable BLAS threading
#     bthreads = BLAS.get_num_threads()
#     BLAS.set_num_threads(1)

#     # This chunking method works but results in very low length per MC chain,
#     # which is suboptimal for statistical purposes. Sticking with one chain
#     # per thread for now, although it is slower on efficiency cores.
#     # Implemented based on https://discourse.julialang.org/t/multithreading-with-shared-memory-caches/100194/2
#     # Break your work into chunks
#     # More chunks per thread has lower overhead but worse load balancing
#     chunks_per_thread = Nsteps % Nthreads
#     chunks = Iterators.partition(1:Nsteps, chunks_per_thread)

#     # Map over the chunks, creating an array of spawned tasks
#     tasks = map(chunks) do chunk
#         Threads.@spawn begin
#             composite = similar(data, S)
#             # Setup structs to pass to DynamicHMC.mcmc
#             instance = HierarchicalOptimizer(MH_model, disp_model, models, data, composite, logAge, metallicities,
#                                              true, similar(x0), true)
#             # The call signature for the kinetic energy is κ = DynamicHMC.GaussianKineticEnergy(M⁻¹),
#             # where M is the mass matrix (e.g., equation 5.5 in "Handbook of Markov Chain Monte Carlo").
#             # As explained in section 5.4.1 of that text, on pg 134, if you have an estimate for the covariance
#             # matrix of the fitting variables Σ (in our case, the inverse Hessian), you can improve the efficiency
#             # of the HMC sampling by setting M⁻¹ to Σ, which is what we do here.
#             warmup_state = DynamicHMC.initialize_warmup_state(rng, instance;
#                                                               q = x0, # Initial position vector
#                                                               κ = DynamicHMC.GaussianKineticEnergy(MAP.invH), # Kinetic energy
#                                                               ϵ = ϵ) # HMC step size
#             # Only use reporter on first thread
#             # ireporter = i == 1 ? reporter : DynamicHMC.NoProgressReport()
#             # sampling_logdensity = DynamicHMC.SamplingLogDensity(rng, instance, DynamicHMC.NUTS(), ireporter)
#             sampling_logdensity = DynamicHMC.SamplingLogDensity(rng, instance, DynamicHMC.NUTS(),
#                                                                 DynamicHMC.NoProgressReport())
            
#             # Sample
#             return DynamicHMC.mcmc(sampling_logdensity, length(chunk), warmup_state)
#         end
#     end
#     # Now we fetch all the results from the spawned tasks
#     results = fetch.(tasks)
#     result = (posterior_matrix = reduce(hcat, i[1] for i in results),
#               tree_statistics = reduce(vcat, i[2] for i in results))
#     # result = (posterior_matrix = posterior_matrices, tree_statistics = tree_statistics)
#     BLAS.set_num_threads(bthreads)

#     # Test convergence
#     tree_stats = DynamicHMC.Diagnostics.summarize_tree_statistics(result.tree_statistics)
#     show_convergence && display(tree_stats)
#     if tree_stats.a_mean < 0.8
#         @warn "Acceptance ratio for samples less than 80%, recommend re-running with smaller step size ϵ."
#     end
#     if tree_stats.termination_counts.divergence > (0.1 * Nsteps)
#         @warn "More than 10% of samples diverged, recommend re-running with smaller step size ϵ."
#     end

#     # Transform samples
#     Nbins = length(μ) - nparams(MH_model) - nparams(disp_model)
#     # Get indices into μ corresponding to free parameters
#     row_idxs = vcat(1:Nbins, (Nbins+1:Nbins+length(free))[free])
#     exptransform_samples!(result.posterior_matrix, μ[row_idxs], tf[free], free[free])

#     # Now we need to expand posterior_samples to include fixed parameters as well
#     if false in free
#         samples = similar(result.posterior_matrix, (length(μ), Nsteps))
#         samples[row_idxs, :] .= result.posterior_matrix
#         # Now write in fixed parameters
#         par = (values(fittable_params(MH_model))..., values(fittable_params(disp_model))...)
#         for i in 1:length(free)
#             if ~free[i] # if parameter is fixed,
#                 samples[Nbins+i, :] .= par[i]
#             end
#         end
#         result = (posterior_matrix = samples, tree_statistics = result.tree_statistics)
#     end
#     return result
# end
