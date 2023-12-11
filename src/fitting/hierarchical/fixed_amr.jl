# Gradient-based optimization for SFH given a fixed input age-metallicity relation, expressed as a series of relative weights that are applied per-template. For each unique entry in logAge, the sum of all relative weights for isochrones with that logAge but *any* metallicity must equal 1.

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
    @assert length(logAge) == length(metallicities)
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
