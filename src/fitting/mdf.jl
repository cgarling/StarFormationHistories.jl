# Ways to derive integrated metallicity distribution functions from
# the results of SFH fits.
"""
    (unique_MH, mass_mdf) =
    mdf_amr(coeffs::AbstractVector{<:Number},
            logAge::AbstractVector{<:Number},
            metallicities::AbstractVector{<:Number})

Calculates the mass-weighted metallicity distribution function given a set of *stellar mass coefficients* `coeffs` for stellar populations with logarithmic ages `logAge=log10(age [yr])` and metallicities given by `metallicities`. This is calculated as

```math
P_j = \\frac{ \\sum_k r_{j,k} \\, [\\text{M} / \\text{H}]_k}{\\sum_{j,k} r_{j,k} \\, [\\text{M} / \\text{H}]_k}
```

where ``r_{j,k}`` are the elements of `coeffs` where ``j`` indexes over unique entries in `logAge` and ``k`` indexes over unique entries in `metallicities.` This is the same nomenclature used in the [the documentation on constrained metallicity evolutions](@ref metal_evo_intro).

# Examples
```jldoctest; setup = :(import StarFormationHistories: mdf_amr)
julia> mdf_amr([1.0, 2.0, 1.0], [10, 10, 10], [-2, -1.5, -1])
([-2.0, -1.5, -1.0], [0.25, 0.5, 0.25])
```
"""
function mdf_amr(coeffs::AbstractVector{<:Number}, # Stellar mass coefficients
                 logAge::AbstractVector{<:Number},
                 metallicities::AbstractVector{<:Number})

    @assert length(coeffs) == length(logAge) == length(metallicities)
    # Now, loop through and sum all the coeffs for each unique metallicity
    # This will form an (unnormalized) mass-weighted MDF.
    unique_MH = unique(metallicities)
    mass_mdf = [sum(coeffs[idx]) for idx in (findall( ==(i), metallicities) for i in unique_MH)]
    mass_mdf ./= sum(mass_mdf) # Normalize to sum probability = 1
    return unique_MH, mass_mdf
end

# function mdf_amr(stellar_masses::AbstractVector{<:Number},
#                  logAge::AbstractVector{<:Number},
#                  metallicities::AbstractVector{<:Number},
#                  relweights::AbstractVector{<:Number};
#                  relweightsmin::Number=0)

#     unique_logAge = unique(logAge)
#     unique_MH = unique(metallicities)
#     @assert length(stellar_masses) == length(unique_logAge) # one SFR / stellar mass coefficient per unique logAge
#     @assert length(logAge) == length(metallicities) == length(relweights)
#     @assert all(x -> x ≥ 0, relweights) # All relative weights must be \ge 0
#     @assert relweightsmin >= 0 # By definition relweightsmin must be greater than or equal to 0

#     # Identify which of the provided models are significant enough
#     # to be included in the fitting on the basis of the provided `relweightsmin`.
#     if relweightsmin != 0
#         keep_idx = truncate_relweights(relweightsmin, relweights, logAge) # Method defined below
#         models = models[keep_idx]
#         logAge = logAge[keep_idx]
#         metallicities = metallicities[keep_idx]
#         relweights = relweights[keep_idx]
#     end

#     # Loop through all unique logAge entries and ensure sum over relweights = 1
#     for la in unique_logAge
#         good = findall( logAge .== la )
#         goodsum = sum(relweights[good])
#         if !(goodsum ≈ 1)
#             # Don't warn if relweightsmin != 0 as it will ALWAYS need to renormalize in this case
#             if relweightsmin == 0
#                 @warn "The relative weights for logAge="*string(la)*" provided to `fixed_amr` do not sum to 1 and will be renormalized in place. This warning is suppressed for additional values of logAge." maxlog=1
#             end
#             relweights[good] ./= goodsum
#         end
#     end

#     # Make scratch array for holding full coefficient vector
#     coeffs = similar(logAge)

#     # Compute the index masks for each unique entry in logAge so we can
#     # construct the full coefficients vector when evaluating the likelihood
#     idxlogAge = [findall( ==(i), logAge) for i in unique_logAge]

#     # Expand the SFH coefficients into per-model coefficients
#     for (i, idxs) in enumerate(idxlogAge)
#         @inbounds coeffs[idxs] .= relweights[idxs] .* stellar_masses[i]
#     end

#     # Now, loop through and sum all the coeffs for each unique metallicity
#     # This will form an (unnormalized) mass-weighted MDF.
#     mass_mdf = [sum(coeffs[idx]) for idx in (findall( ==(i), metallicities) for i in unique_MH)]
#     return unique_MH, mass_mdf

# end
