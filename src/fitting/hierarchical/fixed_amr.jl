# Gradient-based optimization for SFH given a fixed input age-metallicity relation, expressed as a series of relative weights that are applied per-template. For each unique entry in logAge, the sum of all relative weights for isochrones with that logAge but *any* metallicity must equal 1.

"""
    fixed_amr(models::AbstractVector{T},
              data::AbstractMatrix{<:Number},
              logAge::AbstractVector{<:Number},
              metallicities::AbstractVector{<:Number},
              relweights::AbstractVector{<:Number};
              composite=Matrix{S}(undef,size(data)),
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
 - `composite` is the working matrix that will be used to store the composite Hess diagram model during computation; must be of the same size as the templates contained in `models` and the observed Hess diagram `data`.
 - `x0` is the vector of initial guesses for the stellar mass coefficients per unique entry in `logAge`. You should basically always be calculating and passing this keyword argument. We provide [`StarFormationHistories.construct_x0_mdf`](@ref) to prepare `x0` assuming constant star formation rate, which is typically a good initial guess. 
Other `kws...` are passed to `Optim.options` to set things like convergence criteria for the optimization.

# Notes
 - All metallicity-related weighting of the `models` is assumed to be captured in the provided `relweights` vector, which has the same length as the `logAge`, `metallicities`, and `models` vectors. Each entry in `relweights` is assumed to be a relative weight for the corresponding `model`. For example, for the model Hess diagram `models[i]`, with log10(age [yr]) = `logAge[i]` and metallicity `metallicities[i]`, the relative weight due to the model's age and metallicity `w(logAge[i], metallicities[i])` is assumed to be `relweights[i]`. The sum of all `relweights` for each unique entry in `logAge` should be 1; i.e., the following condition should be met: `all( sum(relweights[logAge .== la]) ≈ 1 for la in unique(logAge))`. If this is not the case, this function will issue a warning and attempt to renormalize `relweights` by mutating the vector in place. More information on preparation of the `relweights` for input to this method is provided in our online documentation. 
 - This function is designed to work best with a "grid" of stellar models, defined by the outer product of `N` unique entries in `logAge` and `M` unique entries in `metallicities`. See the examples for more information on usage.
"""
function fixed_amr(models::AbstractVector{T},
                   data::AbstractMatrix{<:Number},
                   logAge::AbstractVector{<:Number},
                   metallicities::AbstractVector{<:Number},
                   relweights::AbstractVector{<:Number};
                   composite=Matrix{S}(undef,size(data)),
                   x0=construct_x0_mdf(logAge, convert(S,log10(13.7e9))),
                   kws...) where {S <: Number, T <: AbstractMatrix{S}}

    unique_logAge = unique(logAge)
    @assert length(x0) == length(unique_logAge)
    @assert length(models) == length(logAge) == length(metallicities) == length(relweights)
    @assert all(x -> x ≥ 0, relweights) # All relative weights must be \ge 0

    # Loop through all unique logAge entries and ensure sum over relweights = 1
    # warned_norm = false
    for la in unique_logAge
        good = findall( logAge .== la )
        goodsum = sum(relweights[good])
        if !(goodsum ≈ 1)
            # if !warned_norm
            @warn "The relative weights for logAge="*string(la)*" provided to `fixed_amr` do not sum to 1 and will be renormalized in place. This warning is suppressed for additional values of logAge." maxlog=1
                # warned=true
            # end
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
    fullG = Vector{eltype(x0)}(undef, length(models))
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
                # This is correct but we can reduce
                # G[i] = -sum( fullG[j] * coeffs[j] / x[i] for j in idxs ) * x[i] - 1
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
