"""
    x0::Vector = construct_x0_mdf(logAge::AbstractVector{T},
                                  [ cum_sfh, ]
                                  T_max::Number;
                                  normalize_value::Number = one(T)) where T <: Number

Generates a vector of initial stellar mass normalizations for input to [`fit_sfh`](@ref) and similar methods with a total stellar mass of `normalize_value`. The `logAge` vector must contain the `log10(Age [yr])` of each isochrone that you are going to input as models. If `cum_sfh` is not provided, a constant star formation rate is assumed. For the purposes of computing the constant star formation rate, the provided `logAge` are treated as left-bin edges, with the final right-bin edge being `T_max`, which has units of Gyr. For example, you might have `logAge=[6.6, 6.7, 6.8]` in which case a final logAge of 6.9 would give equal bin widths (in log-space). In this case you would set `T_max = exp10(6.9) / 1e9 ≈ 0.0079` so that the width of the final bin for the star formation rate calculation has the same `log10(Age [yr])` step as the other bins.

A desired cumulative SFH vector `cum_sfh::AbstractVector{<:Number}` can be provided as the second argument, which should correspond to a lookback time vector `unique(logAge)`. You can also provide `cum_sfh` as a length-2 indexable (e.g., a length-2 `Vector{Vector{<:Number}}`) with the first element containing a list of `log10(Age [yr])` values and the second element containing the cumulative SFH values at those values. This cumulative SFH is then interpolated onto the `logAge` provided in the first argument. This method should be used when you want to define the cumulative SFH on a different age grid from the `logAge` you provide in the first argument. The examples below demonstrate these use cases.

The difference between this function and [`StarFormationHistories.construct_x0`](@ref) is that this function generates an `x0` vector that is of length `length(unique(logage))` (that is, a single normalization factor for each unique entry in `logAge`) while [`StarFormationHistories.construct_x0`](@ref) returns an `x0` vector that is of length `length(logAge)`; that is, a normalization factor for every entry in `logAge`. The order of the coefficients is such that the coefficient `x[i]` corresponds to the entry `unique(logAge)[i]`. 

# Notes

# Examples -- Constant SFR
```jldoctest; setup = :(import StarFormationHistories: construct_x0_mdf, construct_x0)
julia> isapprox( construct_x0_mdf([9.0, 8.0, 7.0], 10.0; normalize_value=5.0),
                 [4.504504504504504, 0.4504504504504504, 0.04504504504504504] )
true

julia> isapprox( construct_x0_mdf(repeat([9.0, 8.0, 7.0, 8.0]; inner=3), 10.0; normalize_value=5.0),
                 [4.504504504504504, 0.4504504504504504, 0.04504504504504504] )
true

julia> isapprox( construct_x0_mdf(repeat([9.0, 8.0, 7.0, 8.0]; outer=3), 10.0; normalize_value=5.0),
                 construct_x0([9.0, 8.0, 7.0], 10.0; normalize_value=5.0) )
true
```

# Examples -- Input Cumulative SFH defined on same `logAge` grid
```jldoctest; setup = :(import StarFormationHistories: construct_x0_mdf)
julia> isapprox( construct_x0_mdf([9.0, 8.0, 7.0], [0.9009, 0.99099, 1.0], 10.0; normalize_value=5.0),
                 [4.5045, 0.4504, 0.0450]; atol=1e-3 )
true

julia> isapprox( construct_x0_mdf([9.0, 8.0, 7.0], [0.1, 0.5, 1.0], 10.0; normalize_value=5.0),
                 [0.5, 2.0, 2.5] )
true

julia> isapprox( construct_x0_mdf([7.0, 8.0, 9.0], [1.0, 0.5, 0.1], 10.0; normalize_value=5.0),
                 [2.5, 2.0, 0.5] )
true
```

# Examples -- Input Cumulative SFH with separate `logAge` grid
```jldoctest; setup = :(import StarFormationHistories: construct_x0_mdf)
julia> isapprox( construct_x0_mdf([9.0, 8.0, 7.0],
                                  [[9.0, 8.0, 7.0], [0.9009, 0.99099, 1.0]], 10.0; normalize_value=5.0),
                 construct_x0_mdf([9.0, 8.0, 7.0], [0.9009, 0.99099, 1.0], 10.0; normalize_value=5.0) )
true

julia> isapprox( construct_x0_mdf([9.0, 8.0, 7.0],
                                  [[9.0, 8.5, 8.25, 7.0], [0.9009, 0.945945, 0.9887375, 1.0]], 10.0; normalize_value=5.0),
                 construct_x0_mdf([9.0, 8.0, 7.0], [0.9009, 0.99099, 1.0], 10.0; normalize_value=5.0) )
true
```
"""
function construct_x0_mdf(logAge::AbstractVector{T}, T_max::Number; normalize_value::Number=one(T)) where T <: Number
    @argcheck log10(T_max) + 9 > maximum(logAge)
    min_logAge = minimum(logAge)
    max_logAge = log10(T_max) + 9 # T_max in units of Gyr
    sfr = normalize_value / (exp10(max_logAge) - exp10(min_logAge)) # Average SFR / yr
    unique_logAge = unique(logAge)
    idxs = sortperm(unique_logAge)
    sorted_ul = vcat(unique_logAge[idxs], max_logAge)
    dt = diff(exp10.(sorted_ul))
    return [ begin
                idx = findfirst(Base.Fix1(==, sorted_ul[i]), unique_logAge) # findfirst(==(sorted_ul[i]), unique_logAge)
                sfr * dt[idx]
             end for i in eachindex(unique_logAge) ]
end

function construct_x0_mdf(logAge::AbstractVector{<:Number}, cum_sfh::AbstractVector{T},
                          T_max::Number; normalize_value::Number=one(T)) where T <: Number
    @argcheck minimum(cum_sfh) ≥ zero(T)
    if !isapprox(maximum(cum_sfh), one(T))
        @warn "Maximum of `cum_sfh` argument is $(maximum(cum_sfh)) which is not approximately equal to 1."
    end
    @argcheck log10(T_max) + 9 > maximum(logAge)
    max_logAge = log10(T_max) + 9 # T_max in units of Gyr
    unique_logAge = unique(logAge)
    @argcheck length(unique_logAge) == length(cum_sfh) "`length(unique(logAge))` not equal to `length(cum_sfh)`."
    
    idxs = sortperm(unique_logAge)
    sorted_cum_sfh = vcat(cum_sfh[idxs], zero(T))
    # Test that cum_sfh is properly monotonic
    @argcheck(all(sorted_cum_sfh[i] ≤ sorted_cum_sfh[i-1] for i in eachindex(sorted_cum_sfh)[2:end]),
            "Provided `cum_sfh` must be monotonically increasing as `logAge` decreases.")
    sorted_ul = vcat(unique_logAge[idxs], max_logAge)
    return [ begin
                idx = findfirst(Base.Fix1(==, sorted_ul[i]), unique_logAge) # findfirst(==(sorted_ul[i]), unique_logAge)
                (normalize_value * (sorted_cum_sfh[idx] - sorted_cum_sfh[idx+1]))
             end for i in eachindex(unique_logAge) ]
end

# For providing cum_sfh as a length-2 indexable with cum_sfh[1] = log(age) values
# and cum_sfh[2] = cumulative SFH values at those log(age) values.
# This interpolates the provided pair onto the provided logAge in first argument.
function construct_x0_mdf(logAge::AbstractVector{T}, cum_sfh_vec,
                          T_max::Number; normalize_value::Number=one(T)) where T <: Number
    @argcheck(length(cum_sfh_vec) == 2,
            "`cum_sfh` must either be a vector of numbers or vector containing two vectors that \
             define a cumulative SFH with a different log(age) discretization than the provided \
             `logAge` argument.")
    # Extract cum_sfh info and concatenate with T_max where cum_sfh=0 by definition
    @argcheck(maximum(last(cum_sfh_vec)) ≤ one(T),
            "Maximum of cumulative SFH must be less than or equal to one when passing a custom \
             cumulative SFH to `construct_x0_mdf`.")
    idxs = sortperm(first(cum_sfh_vec))
    cum_sfh_la = vcat(first(cum_sfh_vec)[idxs], log10(T_max) + 9)
    cum_sfh_in = vcat(last(cum_sfh_vec)[idxs], zero(T))
    # Construct interpolant and evaluate at unique(logAge)
    itp = extrapolate(interpolate((cum_sfh_la,), cum_sfh_in, Gridded(Linear())), Flat())
    cum_sfh = itp.(unique(logAge))
    # Feed into above method
    return construct_x0_mdf(logAge, cum_sfh, T_max; normalize_value=normalize_value)
end
