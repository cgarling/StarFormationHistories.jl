# """
#     coeffs = calculate_coeffs_logamr(variables::AbstractVector{<:Number}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number} [, α::Number, β::Number, σ::Number]; MH_func=StarFormationHistories.MH_from_Z)

# Calculates per-model stellar mass coefficients `coeffs` from the fitting parameters of [`StarFormationHistories.fit_templates_logamr`](@ref) and [`StarFormationHistories.hmc_sample_logamr`](@ref). The `variables` returned by these functions is of length `length(unique(logAge))+3`. The first `length(logAge)` entries are stellar mass coefficients, one per unique entry in `logAge`. The final three elements are α, β, and σ defining a metallicity evolution such that the mean metal mass fraction Z for element `i` of `unique(logAge)` is `μ_Z[i] = α * exp10(unique(logAge)[i]) / 1e9 + β`. This is converted to a mean metallicity in [M/H] via the provided callable keyword argument `MH_func` which defaults to [`StarFormationHistories.MH_from_Z`](@ref). The individual weights per each isochrone are then determined via Gaussian weighting with the above mean [M/H] and the provided `σ` in dex.

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


# function fit_templates_logamr()

# function hmc_sample_logamr()


include("fixed_log_amr.jl")
