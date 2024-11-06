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

where ``r_{j,k}`` are the elements of `coeffs` where ``j`` indexes over unique entries in `logAge` and ``k`` indexes over unique entries in `metallicities.` This is the same nomenclature used in the [the documentation on constrained metallicity evolutions](@ref metal_evo_intro). The return values are sorted so that `unique_MH` is in increasing order.

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
    p = sortperm(unique_MH) # Return in sorted order of unique_MH
    return unique_MH[p], mass_mdf[p]
end


