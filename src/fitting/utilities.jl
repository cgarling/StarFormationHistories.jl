"
    stack_models(models::AbstractVector{<:AbstractMatrix{<:Number}}) = reduce(hcat,map(vec,models))
Transforms a vector of matrices into a single matrix, with each matrix from `models` being transcribed into a single column in the output matrix. This data layout enables more efficient calculations in some of our internal functions like [`composite!`](@ref StarFormationHistories.composite!) and [`∇loglikelihood!`](@ref StarFormationHistories.∇loglikelihood!).

# Examples
```julia-repl
julia> stack_models([rand(5,5) for i in 1:10])
25×10 Matrix{Float64}:
...
```
"
stack_models(models::AbstractVector{<:AbstractMatrix{<:Number}}) = reduce(hcat,map(vec,models)) # mapreduce(vec, hcat, models)

"""
    x0::typeof(logage) = construct_x0(logAge::AbstractVector{T},
                                      T_max::Number;
                                      normalize_value::Number=one(T)) where T <: Number

Generates a vector of initial stellar mass normalizations for input to [`fit_templates`](@ref) or [`hmc_sample`](@ref) with a total stellar mass of `normalize_value` such that the implied star formation rate is constant across the provided `logAge` vector that contains the `log10(Age [yr])` of each isochrone that you are going to input as models. For the purposes of computing the constant star formation rate, the provided `logAge` are treated as left-bin edges, with the final right-bin edge being `T_max`, which has units of Gyr. For example, you might have `logAge=[6.6, 6.7, 6.8]` in which case a final logAge of 6.9 would give equal bin widths. In this case you would set `T_max = exp10(6.9) / 1e9 ≈ 0.0079` so that the width of the final bin for the star formation rate calculation has the same `log10(Age [yr])` step as the other bins.

# Examples
```julia-repl
julia> x0 = construct_x0(repeat([7.0,8.0,9.0],3), 10.0; normalize_value=5.0)
9-element Vector{Float64}: ...

julia> sum(x0)
4.99... # Close to `normalize_value`.
```
"""
function construct_x0(logAge::AbstractVector{T}, T_max::Number; normalize_value::Number=one(T)) where T <: Number
    min_log, max_log = extrema(logAge)
    max_logAge = log10(T_max) + 9 # T_max in units of Gyr
    @assert max_logAge > max_log # max_logAge must be greater than maximum(logAge)
    unique_logAge = vcat(sort!(unique(logAge)), max_logAge)
    sfr = normalize_value / (exp10(max_logAge) - exp10(min_log)) # Average SFR / yr
    num_ages = [count(logAge .== la) for la in unique_logAge] # number of entries per unique
    dt = [exp10(unique_logAge[i+1]) - exp10(unique_logAge[i]) for i in 1:length(unique_logAge)-1]
    result = similar(logAge)
    for i in eachindex(logAge, result)
        la = logAge[i]
        idx = findfirst( x -> x==la, unique_logAge )
        result[i] = sfr * dt[idx] / num_ages[idx]
    end
    return result
end

"""
    (unique_logAge, cum_sfh, sfr, mean_MH) =
        calculate_cum_sfr(coeffs::AbstractVector,
                          logAge::AbstractVector,
                          MH::AbstractVector,
                          T_max::Number;
                          normalize_value=1,
                          sorted::Bool=false)

Calculates cumulative star formation history, star formation rates, and mean metallicity evolution as functions of `logAge = log10(age [yr])`.

# Arguments
 - `coeffs::AbstractVector` is a vector of stellar mass coefficients such as those returned by [`fit_templates`](@ref), for example. Actual stellar mass in stellar population `j` is `coeffs[j] * normalize_value`.
 - `logAge::AbstractVector` is a vector giving the `log10(age [yr])` of the stellar populations corresponding to the provided `coeffs`. For the purposes of calculating star formation rates, these are assumed to be left-bin edges.
 - `MH::AbstractVector` is a vector giving the metallicities of the stellar populations corresponding to the provided `coeffs`.
 - `T_max::Number` is the rightmost final bin edge for calculating star formation rates. For example, you might have `logAge=[6.6, 6.7, 6.8]` in which case a final logAge of 6.9 would give equal bin widths. In this case you would set `T_max = exp10(6.9) / 1e9 ≈ 0.0079` so that the width of the final bin for the star formation rate calculation has the same `log10(Age [yr])` step as the other bins.

# Keyword Arguments
 - `normalize_value` is a multiplicative prefactor to apply to all the `coeffs`; same as the keyword in [`partial_cmd_smooth`](@ref).
 - `sorted::Bool` is either `true` or `false` and signifies whether to assume `logAge` is sorted.

# Returns
 - `unique_logAge::Vector` is essentially `unique(sort(logAge))` and provides the x-values you would plot the other returned vectors against.
 - `cum_sfh::Vector` is the normalized cumulative SFH implied by the provided `coeffs`. This is ~1 at the most recent time in `logAge` and decreases as `logAge` increases.
 - `sfr::Vector` gives the star formation rate across each bin in `unique_logAge`. If `coeffs .* normalize_value` are in units of solar masses, then `sfr` is in units of solar masses per year.
 - `mean_MH::Vector` gives the stellar-mass-weighted mean metallicity of the stellar population as a function of `unique_logAge`. 
"""
function calculate_cum_sfr(coeffs::AbstractVector, logAge::AbstractVector, MH::AbstractVector, T_max::Number; normalize_value=1, sorted::Bool=false)
    @assert axes(coeffs) == axes(logAge) == axes(MH)
    max_logAge = log10(T_max) + 9 # T_max in units of Gyr
    @assert max_logAge > maximum(logAge)
    coeffs = coeffs .* normalize_value # Transform the coefficients to proper stellar masses
    mstar_total = sum(coeffs) # Calculate the total stellar mass of the model
    # Calculate the stellar mass per time bin by summing over the different MH at each logAge
    if ~sorted # If we aren't sure that logAge is sorted, we sort. 
        idx = sortperm(logAge)
        logAge = logAge[idx]
        coeffs = coeffs[idx]
        MH = MH[idx] 
    end
    unique_logAge = unique(logAge)
    dt = diff( vcat(exp10.(unique_logAge), exp10(max_logAge)) )
    mstar_arr = Vector{eltype(coeffs)}(undef, length(unique_logAge))
    mean_mh_arr = zeros(promote_type(eltype(MH),eltype(coeffs)), length(unique_logAge))
    for i in eachindex(unique_logAge)
        Mstar_tmp = zero(eltype(mstar_arr))
        mh_tmp = Vector{eltype(MH)}(undef,0)
        coeff_tmp = Vector{eltype(coeffs)}(undef,0)
        for j in eachindex(logAge)
            if unique_logAge[i] == logAge[j]
                Mstar_tmp += coeffs[j]
                push!(mh_tmp, MH[j])
                push!(coeff_tmp, coeffs[j])
            end
        end
        mstar_arr[i] = Mstar_tmp
        coeff_sum = sum(coeff_tmp)
        if coeff_sum == 0
            if i == 1
                mean_mh_arr[i] = mean(mh_tmp)
            else
                mean_mh_arr[i] = mean_mh_arr[i-1]
            end
        else
            mean_mh_arr[i] = sum( mh_tmp .* coeff_tmp ) / sum(coeff_tmp) # mean(mh_tmp)
        end
    end
    cum_sfr_arr = cumsum(reverse(mstar_arr)) ./ mstar_total
    reverse!(cum_sfr_arr)
    return unique_logAge, cum_sfr_arr, mstar_arr ./ dt, mean_mh_arr
end
