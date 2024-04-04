# """
#     coeffs = calculate_coeffs_logamr(variables::AbstractVector{<:Number}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number} [, α::Number, β::Number, σ::Number]; MH_func=StarFormationHistories.MH_from_Z)

# Calculates per-model stellar mass coefficients `coeffs` from the fitting parameters of [`StarFormationHistories.fit_templates_logamr`](@ref) and [`StarFormationHistories.hmc_sample_logamr`](@ref). The `variables` returned by these functions is of length `length(unique(logAge))+3`. The first `length(logAge)` entries are stellar mass coefficients, one per unique entry in `logAge`. The final three elements are α, β, and σ defining a metallicity evolution such that the mean metal mass fraction Z for element `i` of `unique(logAge)` is `μ_Z[i] = α * exp10(unique(logAge)[i]) / 1e9 + β`. This is converted to a mean metallicity in [M/H] via the provided callable keyword argument `MH_func` which defaults to [`StarFormationHistories.MH_from_Z`](@ref). The individual weights per each isochrone are then determined via Gaussian weighting with the above mean [M/H] and the provided `σ` in dex. The provided `metallicities` vector should be in [M/H]. 

# # Notes
#  - If the provided AMR coefficients cause the mean metal mass fraction Z to be negative for any entry in `logAge`, this function will throw an error as a negative Z cannot be converted to an [M/H]. 
# """
function calculate_coeffs_logamr(variables::AbstractVector{<:Number}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}, α::Number, β::Number, σ::Number; MH_func=MH_from_Z)
    S = promote_type(eltype(variables), eltype(logAge), eltype(metallicities), typeof(α), typeof(β), typeof(σ))
    # Compute the coefficients on each model template given the `variables` and the MDF
    unique_logAge = unique(logAge)
    @assert length(variables) == length(unique_logAge)
    coeffs = Vector{S}(undef,length(logAge))
    norm_vals = Vector{S}(undef,length(unique_logAge))
    for i in eachindex(unique_logAge)
        la = unique_logAge[i]
        μZ = α * exp10(la) / 1e9 + β # Find the mean Z of this age bin
        # Test that the mean Z is greater than 0
        if μZ < 0
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
@inline function fg_log_amr!(F, G, variables::AbstractVector{<:Number}, models::AbstractVector{T}, data::AbstractMatrix{<:Number}, composite::AbstractMatrix{<:Number}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}, MH_func, MH_deriv_Z) where T <: AbstractMatrix{<:Number}
    # `variables` should have length `length(unique(logAge)) + 3`; coeffs for each unique
    # entry in logAge, plus α and β to define the MDF at fixed logAge and σ to define Gaussian width
    @assert axes(data) == axes(composite)
    S = promote_type(eltype(variables), eltype(eltype(models)), eltype(eltype(data)), eltype(composite), eltype(logAge), eltype(metallicities))
    # Compute the coefficients on each model template given the `variables` and the MDF
    α, β, σ = variables[end-2], variables[end-1], variables[end]
    unique_logAge = unique(logAge)
    # Calculate the per-template coefficents and normalization values
    coeffs = calculate_coeffs_logamr(view(variables,firstindex(variables):lastindex(variables)-3), logAge, metallicities, α, β, σ; MH_func=MH_func)

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
            meanZ = α * age + β # Mean metal mass fraction
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
            dZdα = age # Derivative of metal mass fraction with respect to α
            dLdβ = dAdμ * dμdZ * dZdβ
            dLdα = dLdβ * age
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

# function fit_templates_logamr()

# function hmc_sample_logamr()


include("fixed_log_amr.jl")
