"""
    coeffs = calculate_coeffs_logamr(variables::AbstractVector{<:Number}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}, [, α::Number, β::Number, σ::Number]; MH_func=StarFormationHistories.MH_from_Z, max_logAge::Number=maximum(logAge))

Calculates per-model stellar mass coefficients `coeffs` from the fitting parameters of [`StarFormationHistories.fit_templates_logamr`](@ref) and [`StarFormationHistories.hmc_sample_logamr`](@ref). The `variables` returned by these functions is of length `length(unique(logAge))+3`. The first `length(logAge)` entries are stellar mass coefficients, one per unique entry in `logAge`. The final three elements are α, β, and σ defining a metallicity evolution such that the mean metal mass fraction Z for element `i` of `unique(logAge)` is `μ_Z[i] = α * (exp10(max_logAge) - exp10(unique(logAge)[i])) / 1e9 + β`. This is converted to a mean metallicity in [M/H] via the provided callable keyword argument `MH_func` which defaults to [`StarFormationHistories.MH_from_Z`](@ref). The individual weights per each isochrone are then determined via Gaussian weighting with the above mean [M/H] and the provided `σ` in dex. The provided `metallicities` vector should be in [M/H]. 

# Notes
 - Physically, the metal mass fraction `Z` must always be positive. Under the above model, this means α and β must be positive. With σ being a Gaussian width, it too must be positive. 
 - 
"""
function calculate_coeffs_logamr(variables::AbstractVector{<:Number}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}, α::Number, β::Number, σ::Number; MH_func=MH_from_Z, max_logAge=maximum(logAge))
    @assert (α > 0) & (β > 0) & (σ > 0)
    if maximum(logAge) > max_logAge
        @warn "We recommend that the keyword argument `max_logAge` to `StarFormationHistories.calculate_coeffs_logamr` be set equal to or greater than the maximum of the `logAge` argument. The provided `max_logAge` is less than `maximum(logAge)`, such that it is possible the metal mass fraction may become negative in the model, which would be unphysical."
    end
    S = promote_type(eltype(variables), eltype(logAge), eltype(metallicities), typeof(α), typeof(β), typeof(σ))
    # Compute the coefficients on each model template given the `variables` and the MDF
    unique_logAge = unique(logAge)
    max_age = exp10(max_logAge) # Lookback time in yr at which Z = β
    @assert length(variables) == length(unique_logAge)
    coeffs = Vector{S}(undef,length(logAge))
    norm_vals = Vector{S}(undef,length(unique_logAge))
    for i in eachindex(unique_logAge)
        la = unique_logAge[i]
        μZ = α * (max_age - exp10(la)) / 1e9 + β # Find the mean Z of this age bin
        # Test that the mean Z is greater than 0
        if μZ < 0 # Keep this for now
            throw(DomainError(μZ, "The provided coefficients to `calculate_coeffs_logamr` resulted in a mean metal mass fraction Z for `logAge` entry "*string(la)*" less than 0."))
        end
        idxs = findall( ==(la), logAge) # Find all entries that match this logAge
        μMH = MH_func(μZ) # Calculate mean metallicity from mean Z
        tmp_coeffs = [_gausspdf(metallicities[j], μMH, σ) for j in idxs] # Calculate relative weights; _gausspdf from linear_amr.jl
        A = sum(tmp_coeffs)
        norm_vals[i] = A
        # Make sure sum over tmp_coeffs equals 1 and write to coeffs
        coeffs[idxs] .= tmp_coeffs .* variables[i] ./ A
    end
    return coeffs
end
calculate_coeffs_logamr(variables, logAge, metallicities; kws...) =
    calculate_coeffs_logamr(view(variables,firstindex(variables):lastindex(variables)-3),
                         logAge, metallicities, variables[end-2], variables[end-1], variables[end]; kws...)



# Calculate loglikelihood and gradient with respect to SFR parameters and logarithmic AMR parameters
@inline function fg_logamr!(F, G, variables::AbstractVector{<:Number}, models::AbstractVector{T}, data::AbstractMatrix{<:Number}, composite::AbstractMatrix{<:Number}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}, max_age::Number, MH_func, MH_deriv_Z) where T <: AbstractMatrix{<:Number}
    # `variables` should have length `length(unique(logAge)) + 3`; coeffs for each unique
    # entry in logAge, plus α and β to define the MDF at fixed logAge and σ to define Gaussian width
    @assert axes(data) == axes(composite)
    S = promote_type(eltype(variables), eltype(eltype(models)), eltype(eltype(data)), eltype(composite), eltype(logAge), eltype(metallicities), typeof(max_age))
    # Compute the coefficients on each model template given the `variables` and the MDF
    α, β, σ = variables[end-2], variables[end-1], variables[end]
    unique_logAge = unique(logAge)
    max_logAge = log10(max_age) + 9
    # Calculate the per-template coefficents and normalization values
    coeffs = calculate_coeffs_logamr(view(variables,firstindex(variables):lastindex(variables)-3), logAge, metallicities, α, β, σ; MH_func=MH_func, max_logAge=max_logAge)

    # Fill the composite array with the equivalent of sum( coeffs .* models )
    # composite = sum( coeffs .* models )
    # return -loglikelihood(composite, data)
    composite!(composite, coeffs, models)
    logL = loglikelihood(composite, data) # Need to do this before ∇loglikelihood! because it will overwrite composite
    if (F != nothing) & (G != nothing) # Optim.optimize wants objective and gradient
        @assert axes(G) == axes(variables)
        # Calculate the ∇loglikelihood with respect to model coefficients; we will need all of these
        # fullG = [ ∇loglikelihood(models[i], composite, data) for i in axes(models,1) ]
        fullG = Vector{eltype(G)}(undef,length(models))
        ∇loglikelihood!(fullG, composite, models, data)
        # Now need to do the transformation to the `variables` rather than model coefficients
        G[end-2] = zero(eltype(G))
        G[end-1] = zero(eltype(G))
        G[end] = zero(eltype(G))
        dZdβ = 1 # Derivative of metal mass fraction with respect to β
        for i in axes(G,1)[begin:end-3] 
            la = unique_logAge[i]
            age = exp10(la) / 1e9 # the / 1e9 makes α the slope in MH/Gyr, improves convergence
            # meanZ = α * age + β # Mean metal mass fraction
            meanZ = α * (max_age - age) + β # Mean metal mass fraction
            μ = MH_func(meanZ) # Find the mean metallicity [M/H] of this age bin
            idxs = findall( ==(la), logAge) # Find all entries that match this logAge
            tmp_coeffs = [_gausspdf(metallicities[j], μ, σ) for j in idxs] # Calculate relative weights
            A = sum(tmp_coeffs)
            # This should be correct for any MDF model at fixed logAge
            @inbounds G[i] = -sum( fullG[j] * coeffs[j] / variables[i] for j in idxs )
            # βsum = sum( ((metallicities[idxs[j]]-μ) * tmp_coeffs[j]) for j in eachindex(idxs))
            βsum = sum( ((metallicities[idxs[j]]-μ) * tmp_coeffs[j]) for j in eachindex(idxs))
            dAdμ = -sum( fullG[idxs[j]] * variables[i] *
                ( ((metallicities[idxs[j]]-μ) * tmp_coeffs[j]) - tmp_coeffs[j] / A * βsum )
                         for j in eachindex(idxs)) / A / σ^2
            dμdZ = MH_deriv_Z(meanZ) # Derivative of MH_func(Z) with respect to Z
            dZdα = (max_age - age) # Derivative of metal mass fraction with respect to α
            dLdβ = dAdμ * dμdZ * dZdβ
            dLdα = dLdβ * dZdα
            σsum = sum( tmp_coeffs[j] *
                (metallicities[idxs[j]]-μ)^2 / σ^3 for j in eachindex(idxs))
            dLdσ = -sum( fullG[idxs[j]] * variables[i] *
                (tmp_coeffs[j] * (metallicities[idxs[j]]-μ)^2 / σ^3 -
                tmp_coeffs[j] / A * σsum) for j in eachindex(idxs)) / A
            G[end-2] += dLdα
            G[end-1] += dLdβ
            G[end] += dLdσ
        end
        return -logL
    elseif F != nothing # Optim.optimize wants only objective
        return -logL
    end
end

function fit_templates_logamr(models::AbstractVector{T},
                              data::AbstractMatrix{<:Number},
                              logAge::AbstractVector{<:Number},
                              metallicities::AbstractVector{<:Number};
                              composite = Matrix{S}(undef,size(data)),
                              x0 = vcat(construct_x0_mdf(logAge, convert(S,log10(13.7e9))),
                                        [1e-4, 5e-5, 0.2]),
                              MH_func = MH_from_Z,
                              MH_deriv_Z = dMH_dZ,
                              max_logAge = maximum(logAge),
                              kws...) where {S <: Number, T <: AbstractMatrix{S}}
    unique_logage = unique(logAge)
    max_age = exp10(max_logAge) / 1e9 # Lookback time at which to normalize β in Gyr
    @assert length(x0) == length(unique_logage)+3
    # All variables must be positive  since Z = α * (max_age - age) + β with age being
    # positive lookback time, α must always be positive for an increasing AMR.
    # Perform logarithmic transformation on the provided x0 for all variables.
    x0 = log.(x0)
    # Make scratch array for assessing transformations
    x = similar(x0)
    # Define wrapper function to pass to Optim.only_fg!
    # It looks like you don't need the Jacobian correction to arrive at the maximum likelihood
    # result, and if you remove the Jacobian corrections it actually converges to the
    # non-log-transformed case. However, the uncertainty estimates from the inverse Hessian
    # don't seem reliable without the Jacobian corrections.
    function fg_logamr!_map(F, G, xvec)
        # All variables are log-transformed
        x .= exp.(xvec) 
        logL = fg_logamr!(F, G, x, models, data, composite,
                          logAge, metallicities, max_age, MH_func, MH_deriv_Z)
        # Appy Jacobian correction to logL; α should be the same as
        # D[Log[x],x] = D[Log[-x],x] = 1/x, Log[1/x] = -Log[x]
        logL -= sum( xvec ) 
        # Add the Jacobian correction for every element of G
        G .= G .* x .- 1
        return logL
    end
    function fg_logamr!_mle(F, G, xvec)
        # All variables are log-transformed
        x .= exp.(xvec) 
        logL = fg_logamr!(F, G, x, models, data, composite,
                          logAge, metallicities, max_age, MH_func, MH_deriv_Z)
        G .= G .* x 
        return logL
    end
    # Set up options for the optimization
    # The InitialStatic(1.0,true) alphaguess helps to regularize the optimization and 
    # makes it less sensitive to initial x0.
    bfgs_struct = Optim.BFGS(; alphaguess=LineSearches.InitialStatic(1.0,true),
                             linesearch=LineSearches.HagerZhang())
    # The extended trace will contain the BFGS estimate of the inverse Hessian, aka the
    # covariance matrix, which we can use to make parameter uncertainty estimates
    bfgs_options = Optim.Options(; allow_f_increases=true, store_trace=true, extended_trace=true, kws...)
    # Calculate result
    result_map = Optim.optimize(Optim.only_fg!( fg_logamr!_map ), x0, bfgs_struct, bfgs_options)
    result_mle = Optim.optimize(Optim.only_fg!( fg_logamr!_mle ), Optim.minimizer(result_map), bfgs_struct, bfgs_options)
    # Transform the resulting variables
    μ_map = exp.(deepcopy( Optim.minimizer(result_map) ))
    μ_mle = exp.(deepcopy( Optim.minimizer(result_mle) ))

    # Estimate parameter uncertainties from the inverse Hessian approximation
    σ_map = sqrt.(diag(Optim.trace(result_map)[end].metadata["~inv(H)"]))
    σ_mle = sqrt.(diag(Optim.trace(result_mle)[end].metadata["~inv(H)"]))
    # Need to account for the logarithmic transformation
    σ_map .*= μ_map
    σ_mle .*= μ_mle
    # return (map = LogTransformMDFResult(μ_map, σ_map, Optim.trace(result_map)[end].metadata["~inv(H)"], result_map),
    #         mle = LogTransformMDFResult(μ_mle, σ_mle, Optim.trace(result_mle)[end].metadata["~inv(H)"], result_mle))
    return (map = (μ = μ_map, σ = σ_map, result = result_map),
            mle = (μ = μ_mle, σ = σ_mle, result = result_mle))
end

# function hmc_sample_logamr()


include("fixed_log_amr.jl")
