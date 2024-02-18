# Gradient-based optimization for SFH given a fixed input age-metallicity relation, expressed as a series of relative weights that are applied per-template. For each unique entry in logAge, the sum of all relative weights for isochrones with that logAge but *any* metallicity must equal 1.

"""
    fixed_amr(models::AbstractVector{T},
              data::AbstractMatrix{<:Number},
              logAge::AbstractVector{<:Number},
              metallicities::AbstractVector{<:Number},
              relweights::AbstractVector{<:Number};
              relweightsmin::Number=0, 
              x0=construct_x0_mdf(logAge, convert(S,log10(13.7e9))),
              kws...) where {S <: Number, T <: AbstractMatrix{S}}

Method that fits a linear combination of the provided Hess diagrams `models` to the observed Hess diagram `data`, under an externally-imposed age-metallicity relation (AMR) and/or metallicity distribution function (MDF). As such, a number of coefficients equal to `length(unique(logAge))` are returned; that is, only one coefficient is derived per unique entry in `logAge`.

# Arguments
 - `models::AbstractVector{<:AbstractMatrix{<:Number}}` is a vector of equal-sized matrices that represent the template Hess diagrams for the simple stellar populations that compose the observed Hess diagram.
 - `data::AbstractMatrix{<:Number}` is the Hess diagram for the observed data.
 - `logAge::AbstractVector{<:Number}` is the vector containing the effective ages of the stellar populations used to create the templates in `models`, in units of `log10(age [yr])`. For example, if a population has an age of 1 Myr, its entry in `logAge` should be `log10(10^6) = 6.0`.
 - `metallicities::AbstractVector{<:Number}` is the vector containing the effective metallicities of the stellar populations used to create the templates in `models`. This is most commonly a logarithmic abundance like [M/H] or [Fe/H], but you could use a linear abundance like the metal mass fraction Z if you wanted to. There are some notes on the [Wikipedia](https://en.wikipedia.org/wiki/Metallicity) that might be useful.
 - `relweights::AbstractVector{<:Number}` is a vector of length equal to that of `models` which contains the relative weights to apply to each model Hess diagram resulting from an externally-imposed age-metallicity relation and/or metallicity distribution function. Additional details on how to create these weights is provided in the notes below and in the online documentation.

# Keyword Arguments
 - `relweightsmin` truncates the input list of `models` based on the provided `relweights`, providing a speedup at the cost of precision by removing `models` that contribute least to the overall composite model. By default, no truncation of the input is performed and all provided `models` are used in the fit. We recommend this only be increased when fitting performance begins to impact workflow (e.g., when running massive Monte Carlo experiments). See [`StarFormationHistories.truncate_relweights`](@ref) for implementation details. 
 - `x0` is the vector of initial guesses for the stellar mass coefficients per unique entry in `logAge`. You should basically always be calculating and passing this keyword argument. We provide [`StarFormationHistories.construct_x0_mdf`](@ref) to prepare `x0` assuming constant star formation rate, which is typically a good initial guess. 
Other `kws...` are passed to `Optim.options` to set things like convergence criteria for the optimization.

# Notes
 - All metallicity-related weighting of the `models` is assumed to be captured in the provided `relweights` vector, which has the same length as the `logAge`, `metallicities`, and `models` vectors. Each entry in `relweights` is assumed to be a relative weight for the corresponding `model`. For example, for the model Hess diagram `models[i]`, with log10(age [yr]) = `logAge[i]` and metallicity `metallicities[i]`, the relative weight due to the model's age and metallicity `w(logAge[i], metallicities[i])` is assumed to be `relweights[i]`. The sum of all `relweights` for each unique entry in `logAge` should be 1; i.e., the following condition should be met: `all( sum(relweights[logAge .== la]) ≈ 1 for la in unique(logAge))`. If this is not the case, this function will issue a warning and attempt to renormalize `relweights` by mutating the vector in place. More information on preparation of the `relweights` for input to this method is provided in our online documentation. 
 - This function is designed to work best with a "grid" of stellar models, defined by the outer product of `N` unique entries in `logAge` and `M` unique entries in `metallicities`. See the examples for more information on usage.
"""
function fixed_amr(models::AbstractMatrix{S},
                   data::AbstractVector{<:Number},
                   logAge::AbstractVector{<:Number},
                   metallicities::AbstractVector{<:Number},
                   relweights::AbstractVector{<:Number};
                   relweightsmin::Number=0, # By default, do not truncate input model template list
                   x0=construct_x0_mdf(logAge, convert(S,log10(13.7e9))),
                   kws...) where S <: Number

    composite = Vector{S}(undef,length(data)) # Scratch matrix for storing complex Hess model
    unique_logAge = unique(logAge)
    @assert length(x0) == length(unique_logAge)
    @assert size(models,2) == length(logAge) == length(metallicities) == length(relweights)
    @assert all(x -> x ≥ 0, relweights) # All relative weights must be \ge 0
    @assert relweightsmin >= 0 # By definition relweightsmin must be greater than or equal to 0
    
    # Identify which of the provided models are significant enough
    # to be included in the fitting on the basis of the provided `relweightsmin`.
    if relweightsmin != 0
        keep_idx = truncate_relweights(relweightsmin, relweights, logAge) # Method defined below
        models = models[:,keep_idx]
        logAge = logAge[keep_idx]
        metallicities = metallicities[keep_idx]
        relweights = relweights[keep_idx]
    end

    # Loop through all unique logAge entries and ensure sum over relweights = 1
    for la in unique_logAge
        good = findall( logAge .== la )
        goodsum = sum(relweights[good])
        if !(goodsum ≈ 1)
            # Don't warn if relweightsmin != 0 as it will ALWAYS need to renormalize in this case
            if relweightsmin == 0
                @warn "The relative weights for logAge="*string(la)*" provided to `fixed_amr` do not sum to 1 and will be renormalized in place. This warning is suppressed for additional values of logAge." maxlog=1
            end
            relweights[good] ./= goodsum
        end
    end

    # Compute the index masks for each unique entry in logAge so we can
    # construct the full coefficients vector when evaluating the likelihood
    idxlogAge = [findall( ==(i), logAge) for i in unique_logAge]

    # Perform logarithmic transformation on the provided x0 for all SFH variables
    x0 = log.(x0)
    # Make scratch array for assessing transformations on fitting variables
    x = similar(x0)
    # Make scratch array for holding full gradient from ∇loglikelihood
    fullG = Vector{eltype(x0)}(undef, size(models,2))
    # Make scratch array for holding full coefficient vector
    coeffs = similar(fullG)
    
    # These closures don't seem to hurt performance much
    function fg_map!(F, G, xvec)
        # Transform the provided logarithmic SFH coefficients
        x .= exp.(xvec)
        # Expand the SFH coefficients into per-model coefficients
        for (i, idxs) in enumerate(idxlogAge)
            @inbounds coeffs[idxs] .= relweights[idxs] .* x[i]
        end
        # Construct the composite model
        composite!( composite, coeffs, models )
        logL = loglikelihood(composite, data) # Need to do this before ∇loglikelihood! because it will overwrite composite
        logL += sum(xvec) # This is the Jacobian correction
        if (F != nothing) & (G != nothing) # Optim.optimize wants objective and gradient
            @assert axes(G) == axes(x)
            # Calculate the ∇loglikelihood with respect to model coefficients
            ∇loglikelihood!(fullG, composite, models, data)
            # Now need to do the transformation to the per-logage `x` variables
            # rather than per-model coefficients
            for (i, idxs) in enumerate(idxlogAge)
                @inbounds G[i] = -sum( fullG[j] * coeffs[j] for j in idxs ) - 1
            end
            return -logL
        elseif F != nothing # Optim.optimize wants only objective
            return -logL
        end
    end

    function fg_mle!(F, G, xvec)
        # Transform the provided logarithmic SFH coefficients
        x .= exp.(xvec)
        # Expand the SFH coefficients into per-model coefficients
        for (i, idxs) in enumerate(idxlogAge)
            @inbounds coeffs[idxs] .= relweights[idxs] .* x[i]
        end
        # Construct the composite model
        composite!( composite, coeffs, models )
        logL = loglikelihood(composite, data) # Need to do this before ∇loglikelihood! because it will overwrite composite
        if (F != nothing) & (G != nothing) # Optim.optimize wants objective and gradient
            @assert axes(G) == axes(x)
            # Calculate the ∇loglikelihood with respect to model coefficients
            ∇loglikelihood!(fullG, composite, models, data)
            # Now need to do the transformation to the per-logage `x` variables
            # rather than per-model coefficients
            for (i, idxs) in enumerate(idxlogAge)
                # This is correct but we can reduce
                # G[i] = -sum( fullG[j] * coeffs[j] / x[i] for j in idxs ) * x[i]
                @inbounds G[i] = -sum( fullG[j] * coeffs[j] for j in idxs )
            end
            return -logL
        elseif F != nothing # Optim.optimize wants only objective
            return -logL
        end
    end

    # The InitialStatic(1.0,true) alphaguess helps to regularize the optimization and 
    # makes it less sensitive to initial x0.
    bfgs_struct = Optim.BFGS(; alphaguess=LineSearches.InitialStatic(1.0,true), linesearch=LineSearches.HagerZhang())
    
    # The extended trace will contain the BFGS estimate of the inverse Hessian, aka the
    # covariance matrix, which we can use to make parameter uncertainty estimates
    bfgs_options = Optim.Options(; allow_f_increases=true, store_trace=true, extended_trace=true, kws...)
    
    # Calculate results
    result_map = Optim.optimize(Optim.only_fg!( fg_map! ), x0, bfgs_struct, bfgs_options)
    result_mle = Optim.optimize(Optim.only_fg!( fg_mle! ), Optim.minimizer(result_map), bfgs_struct, bfgs_options)
    
    # Transform the resulting variables
    μ_map = exp.(copy( Optim.minimizer(result_map) ))
    μ_mle = exp.(copy( Optim.minimizer(result_mle) ))

    # Estimate parameter uncertainties from the inverse Hessian approximation
    σ_map = sqrt.(diag(Optim.trace(result_map)[end].metadata["~inv(H)"]))
    σ_mle = sqrt.(diag(Optim.trace(result_mle)[end].metadata["~inv(H)"]))

    return (map = (μ = μ_map, σ = σ_map, invH = Optim.trace(result_map)[end].metadata["~inv(H)"], result = result_map),
            mle = (μ = μ_mle, σ = σ_mle, invH = Optim.trace(result_mle)[end].metadata["~inv(H)"], result = result_mle))
    
end
fixed_amr(models::AbstractVector{<:AbstractMatrix{<:Number}}, data::AbstractMatrix{<:Number}, args...; kws...) = fixed_amr(stack_models(models), vec(data), args...; kws...)

"""
    keep_idx::Vector{Int} = truncate_relweights(relweightsmin::Number, relweights::AbstractVector{<:Number}, logAge::AbstractVector{<:Number})

Method to truncate an isochrone grid with log10(age [yr]) values `logAge` and relative weights `relweights` due to an age-metallicity relation to only include models with `relweights` greater than `relweightsmin` times the maximum relative weight for each unique entry in `logAge`. The input vectors are the same as those for [`StarFormationHistories.fixed_amr`](@ref), which includes more information. Returns a vector of the indices into `relweights` and `logAge` of the isochrone models whose relative weights are significant given the provided `relweightsmin`.

# Examples
When using a fixed input age-metallicity relation as enabled by, for example, [`StarFormationHistories.fixed_amr`](@ref), only the star formation rate (or total stellar mass) coefficients need to be fit, as the metallicity distribution is no longer a free parameter in the model. As such, the relative weights of each model with identical `logAge` but different `metallicities` only need to be computed once at the start of the optimization. As the metallicity distribution is not a free parameter, it is also possible to truncate the list of models to only those that contribute significantly to the final composite model to improve runtime performance. That is what this method does.

A simple isochrone grid will be two-dimensional, encompassing age and metallicity. Consider a subset of the model grid with the same age such that `unique(logAge) = [10.0]` but a series of different metallicities, `metallicities = -2.5:0.25:0`. If we model the metallicity distribution function for this age as having a mean [M/H] of -2.0 and a Gaussian spread of 0.2 dex, then the relative weights of these models can be approximated as 

```julia
import Distributions: Normal, pdf
metallicities = -2.5:0.25:0
relweights = pdf.(Normal(-2.0, 0.2), metallicities)
relweights ./= sum(relweights) # Normalize the relative weights to unity sum
```

```
11-element Vector{Float64}:
 0.021919934465195145
 0.2284109622221623
 0.4988954088848224
 0.2284109622221623
 0.021919934465195145
 0.0004409368867815243
 1.8592101580561089e-6
 1.6432188478108614e-9
 3.0442281937632026e-13
 1.1821534989089337e-17
 9.622444440364979e-23
```

Several of these models with very low relative weights are unlikely to contribute significantly to the final composite model. We can select out only the significant ones with, say, relative weights greater than 10% of the maximum as `StarFormationHistories.truncate_relweights(0.1, relweights, fill(10.0,length(metallicities)))` which will return indices into `relweights` whose values are greater than `0.1 * maximum(relweights) = 0.04988954088848224`,

```
3-element Vector{Int64}:
 2
 3
 4
```

which correspond to `relweights[2,3,4] = [ 0.2284109622221623, 0.4988954088848224, 0.2284109622221623 ]`. If we use only these 3 templates in the fit, instead of the original 11, we will achieve a speedup of almost 4x with a minor loss in precision which, in most cases, will be less than the numerical uncertainties on the individual star formation rate parameters. However, as fits of these sort are naturally quite fast, we recommend use of this type of truncation only in applications where many fits are required (e.g., Monte Carlo experiments). For most applications, this level of optimization is not necessary.
"""
function truncate_relweights(relweightsmin::Number,
                             relweights::AbstractVector{<:Number},
                             logAge::AbstractVector{<:Number})
    @assert length(relweights) == length(logAge)
    relweightsmin == 0 && return collect(eachindex(relweights, logAge)) # short circuit for relweightsmin = 0
    keep_idx = Vector{Int}[] # Vector of vectors of integer indices
    for la in unique(logAge)
        good = findall(logAge .== la) # Select models with correct logAge
        tmp_relweights = relweights[good]
        max_relweight = maximum(tmp_relweights) # Find maximum relative weight for this set of models
        high_weights = findall(tmp_relweights .>= (relweightsmin * max_relweight))
        push!(keep_idx, good[high_weights])
    end
    return reduce(vcat, keep_idx)
end
