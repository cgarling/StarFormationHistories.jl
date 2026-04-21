"
    stack_models(models::AbstractVector{<:AbstractMatrix{<:Number}})
Transforms a vector of matrices into a single matrix, with each matrix from `models` being transcribed into a single column in the output matrix. This data layout enables more efficient calculations in some of our internal functions like [`composite!`](@ref StarFormationHistories.composite!) and [`∇loglikelihood!`](@ref StarFormationHistories.∇loglikelihood!). This function is just `reduce(hcat, map(vec, models))`.

# Examples
```julia-repl
julia> stack_models([rand(5,5) for i in 1:10])
25×10 Matrix{Float64}:
...
```
"
stack_models(models::AbstractVector{<:AbstractMatrix{<:Number}}) =
    reduce(hcat, map(vec, models)) # mapreduce(vec, hcat, models)

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
function construct_x0(logAge::AbstractVector{T}, T_max::Number;
                      normalize_value::Number=one(T)) where T <: Number
    @argcheck log10(T_max) + 9 > maximum(logAge)
    min_logAge = minimum(logAge)
    max_logAge = log10(T_max) + 9 # T_max in units of Gyr
    unique_logAge = vcat(sort!(unique(logAge)), max_logAge)
    sfr = normalize_value / (exp10(max_logAge) - exp10(min_logAge)) # Average SFR / yr
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
    renormalize_x0(data::AbstractVector{<:Number},
                   models::AbstractMatrix{<:Number},
                   x0::AbstractVector{<:Number},
                   full_coeffs::AbstractVector{<:Number} = x0) -> x0_renormalized
    renormalize_x0(data::AbstractMatrix{<:Number},
                   models::AbstractVector{<:AbstractMatrix{<:Number}},
                   x0::AbstractVector{<:Number}) -> x0_renormalized

Renormalizes the overall scale of the coefficient vector `x0` so that the composite
Hess diagram model best matches the total counts in `data`, and returns the
renormalized `x0_renormalized`. The composite model is ``m_i = \\sum_j r_j \\, c_{i,j}``
(see [`StarFormationHistories.composite!`](@ref)) where ``r_j`` are the coefficients.

The optimal overall scale factor ``\\alpha`` is derived analytically by maximizing the
Poisson log-likelihood with respect to a uniform rescaling of all coefficients
``r_j \\to \\alpha \\, r_j``, yielding

```math
\\alpha^* = \\frac{\\sum_i n_i}{\\sum_i m_i}
```

where ``n_i`` is bin ``i`` of the observed Hess diagram `data` and ``m_i`` is the
composite model evaluated at the proposed `x0`. The renormalized coefficients are then
`x0_renormalized = x0 .* α`. This preserves the relative proportions between all
elements of `x0` while adjusting the overall normalization to best match `data`.

The optional fourth argument `full_coeffs` is the *expanded* per-model coefficient
vector derived from `x0`. When `x0` is a per-unique-age coefficient vector (as used by
[`fit_sfh`](@ref) and [`fixed_amr`](@ref)) rather than a per-model coefficient vector,
`full_coeffs` should be the result of [`calculate_coeffs`](@ref) applied to `x0`.
The composite is then computed from `full_coeffs` while `x0` is scaled. In the default
case `full_coeffs = x0`, which is appropriate when `x0` directly indexes the `models`
(as in [`fit_templates`](@ref) and similar functions).

The second call signature accepts `models` as a vector of matrices and `data` as a
matrix and converts them to the flattened format before dispatching to the primary
method.

# Examples
```jldoctest
julia> models = [rand(5,5) for i in 1:3];

julia> true_x0 = [1.0, 2.0, 3.0];

julia> data = sum(true_x0 .* models);  # noise-free data

julia> x0 = true_x0 .* 0.5;  # deliberately wrong normalization

julia> x0_ren = StarFormationHistories.renormalize_x0(data, models, x0);

julia> sum(x0_ren .* models) ≈ data  # total counts match
true
```
"""
function renormalize_x0(data::AbstractVector{<:Number},
                        models::AbstractMatrix{<:Number},
                        x0::AbstractVector{<:Number},
                        full_coeffs::AbstractVector{<:Number} = x0)
    S = promote_type(eltype(data), eltype(models), eltype(full_coeffs))
    composite = Vector{S}(undef, length(data))
    composite!(composite, full_coeffs, models)
    composite_sum = sum(composite)
    iszero(composite_sum) && return copy(x0)  # guard against degenerate case
    α = sum(data) / composite_sum
    return x0 .* α
end
# Convenience wrapper for the vector-of-matrices call signature
renormalize_x0(data::AbstractMatrix{<:Number},
               models::AbstractVector{<:AbstractMatrix{<:Number}},
               x0::AbstractVector{<:Number},
               full_coeffs::AbstractVector{<:Number} = x0) =
    renormalize_x0(vec(data), stack_models(models), x0, full_coeffs)

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
 - `T_max::Number` is the rightmost final bin edge for calculating star formation rates in units of Gyr. For example, you might have `logAge=[6.6, 6.7, 6.8]` in which case a final logAge of 6.9 would give equal bin widths. In this case you would set `T_max = exp10(6.9) / 1e9 ≈ 0.0079` so that the width of the final bin for the star formation rate calculation has the same `log10(Age [yr])` step as the other bins.

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
    @argcheck axes(coeffs) == axes(logAge) == axes(MH)
    @argcheck log10(T_max) + 9 > maximum(logAge)
    max_logAge = log10(T_max) + 9 # T_max in units of Gyr
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

"""
    (cum_sfh, sfr, mean_MH, samples) = 
        cum_sfr_quantiles(result::Union{CompositeBFGSResult,BFGSResult},
                          logAge::AbstractVector{<:Number},
                          MH::AbstractVector{<:Number}, T_max::Number,
                          Nsamples::Integer, q; kws...)

Draws `Nsamples` independent star formation histories from the solution `result` and calculates quantiles `q` across the samples in each unique time bin (i.e., `unique(logAge)`) for the cumulative star formation histories, star formation rates, and mean metallicities. Also returns the drawn samples.

# Arguments
 - `result::Union{CompositeBFGSResult,BFGSResult}` is a BFGS result object as returned by [`fit_sfh`](@ref), for example, whose contents will be used to sample random, independent star formation histories. If this is a `CompositeBFGSResult` containing both the MLE and MAP solutions, then the MLE solution is used for the best-fit values (SFRs and metallicity parameters) and the MAP solution is used for the uncertainty estimate.
 - `logAge::AbstractVector` is a vector giving the `log10(age [yr])` of the stellar populations that were used to derive `result`. For the purposes of calculating star formation rates, these are assumed to be left-bin edges.
 - `MH::AbstractVector` is a vector giving the metallicities of the stellar populations that were used to derive `result`.
 - `T_max::Number` is the rightmost final bin edge for calculating star formation rates in units of Gyr. For example, you might have `logAge=[6.6, 6.7, 6.8]` in which case a final logAge of 6.9 would give equal bin widths. In this case you would set `T_max = exp10(6.9) / 1e9 ≈ 0.0079` so that the width of the final bin for the star formation rate calculation has the same `log10(Age [yr])` step as the other bins.
 - `Nsamples::Integer` is the number of random, independent star formation histories to draw from `result` to use when calculating quantiles.
  - `q` are the quantiles you wish to have calculated. Can be one number (i.e., `0.5` for median), or most iterable types (e.g., `(0.16, 0.5, 0.84)` for median and 1-σ range).

# Keyword Arguments
 - `kws...` are passed to [`calculate_cum_sfr`](@ref StarFormationHistories.calculate_cum_sfr), see that method's documentation for more information.

# Returns
 - `cum_sfh::Matrix` has size `(length(unique(logAge)), length(q))` and contains the normalized cumulative SFH. This is ~1 at the most recent time in `logAge` and decreases as `logAge` increases.
 - `sfr::Matrix` has size `(length(unique(logAge)), length(q))` and gives the star formation rate across each bin in `unique(logAge)`.
 - `mean_MH::Matrix` has size `(length(unique(logAge)), length(q))` and gives the stellar-mass-weighted mean metallicity of the stellar population as a function of `unique(logAge)`.
 - `samples::Matrix` has size `(length(unique(logAge)), Nsamples)` and contains the unique samples drawn from `result` that were used to derive the quantiles.

# Examples
```julia-repl
julia> result = fit_sfh(...)
CompositeBFGSResult{...} ...

julia> q = (0.16, 0.5, 0.84) # quantiles we want

julia> result = cum_sfr_quantiles(result, logAge, MH, 13.7, 10_000, q);

julia> length(result) == 4
true

julia> all(size(result[i]) == (length(unique(logAge)), length(q)) for i in 1:3)
true

julia> size(result[4], 2) == 10_000
```
"""
function cum_sfr_quantiles(result::CompositeBFGSResult, logAge::AbstractVector{<:Number},
                           MH::AbstractVector{<:Number}, T_max::Number, Nsamples::Integer, q;
                           kws...)
    MH_model, disp_model = result.mle.MH_model, result.mle.disp_model
    samples = rand(result, Nsamples)
    return _cum_sfr_quantiles(samples, MH_model, disp_model, logAge, MH, T_max, q; kws...)
end
function cum_sfr_quantiles(result::BFGSResult, logAge::AbstractVector{<:Number},
                           MH::AbstractVector{<:Number}, T_max::Number, Nsamples::Integer, q;
                           kws...)
    MH_model, disp_model = result.MH_model, result.disp_model
    samples = rand(result, Nsamples)
    return _cum_sfr_quantiles(samples, MH_model, disp_model, logAge, MH, T_max, q; kws...)
end
function _cum_sfr_quantiles(samples, MH_model, disp_model, logAge, MH, T_max, q; kws...)
    # Get number of unique time bins
    npar_MH_model = nparams(MH_model)
    npar_disp_model = nparams(disp_model)
    Nbins = size(samples, 1) - npar_MH_model - npar_disp_model
    Nsamples = size(samples, 2)
    
    # Allocate matrices to accumulate cumulative SFHs, SFRs, and <[M/H]> for
    # all samples
    cum_sfh_mat = Matrix{Float64}(undef, Nsamples, Nbins)
    sfrs_mat = similar(cum_sfh_mat)
    mean_mh_mat = similar(cum_sfh_mat)
    good = fill(true, Nsamples) # Vector to track good samples; bad samples are set to false
    Threads.@threads for i in 1:Nsamples
        r = view(samples, :, i)
        new_MH_model = update_params(MH_model, @view(r[Nbins+1:end-npar_disp_model]))
        new_disp_model = update_params(disp_model, @view(r[end-npar_disp_model+1:end]))
        tmp_coeffs = calculate_coeffs(new_MH_model, new_disp_model, @view(r[begin:Nbins]),
                                      logAge, MH)
        if any(!isfinite, tmp_coeffs)
            good[i] = false
            continue
        end
        # We are doing a lot of extra work in calculate_cum_sfr
        # that we could do once here, but it would require a bespoke implementation
        _, mdf_1, mdf_2, mdf_3 = calculate_cum_sfr(tmp_coeffs, logAge, MH, T_max; kws...)
        cum_sfh_mat[i,:] .= mdf_1
        sfrs_mat[i,:] .= mdf_2
        mean_mh_mat[i,:] .= mdf_3
    end
    if count(good) != Nsamples
        @info "$(count(good)) / $Nsamples samples were valid."
    end

    # Allocate matrices to accumulate quantiles
    cum_sfh_q = Matrix{Float64}(undef, Nbins, length(q))
    sfrs_q = similar(cum_sfh_q)
    mean_mh_q = similar(cum_sfh_q)
    # Calculate quantiles on samples
    Threads.@threads for i in 1:Nbins
        cum_sfh_q[i,:] .= quantile(view(cum_sfh_mat, good, i), q)
        sfrs_q[i,:] .= quantile(view(sfrs_mat, good, i), q)
        mean_mh_q[i,:] .= quantile(view(mean_mh_mat, good, i), q)
    end
    # cum_sfh_quantiles = [quantile(row, q) for row in eachrow(cum_sfh)]
    # cum_sfh_quantiles = [SVector(quantile(row, q)) for row in eachrow(cum_sfh)]
    # cum_sfh_quantiles = tups_to_mat([quantile(row, q) for row in eachrow(cum_sfh)])
    # return cum_sfh_quantiles
    return (cum_sfh = cum_sfh_q, sfrs = sfrs_q, mean_mh = mean_mh_q, samples = samples)
end


"""
    tau_interp(unique_logAge, max_logAge, cum_sfh)
Returns an interpolator for the lookback time (in Gyr) at which a given fraction of the total stellar mass of a galaxy formed. This is a helper function for [`tau`](@ref StarFormationHistories.tau).
"""
function tau_interp(unique_logAge, max_logAge, cum_sfh)
    @argcheck length(unique_logAge) == length(cum_sfh) "length(unique_logAge) != length(cum_sfh)"
    @argcheck maximum(unique_logAge) < max_logAge "`max_logAge` must be greater than the maximum of `unique_logAge`."
    if !issorted(cum_sfh)
        cum_sfh = reverse(cum_sfh)
        unique_logAge = reverse(unique_logAge)
    end

    !issorted(cum_sfh) && error("`cum_sfh` must be sorted in ascending or descending order.")
    !issorted(unique_logAge; rev=true) && error("`unique_logAge` must be sorted in the same order as `cum_sfh`.")

    if first(cum_sfh) != 0
        cum_sfh = vcat(0.0, cum_sfh)
    end

    if first(unique_logAge) != max_logAge
        unique_logAge = vcat(max_logAge, unique_logAge)
    end

    cum_sfh = cum_sfh ./ maximum(cum_sfh) # Ensure cum_sfh is normalized to 1
    # Convert logAge to lookback time in Gyr and interpolate
    deduplicate_knots!(cum_sfh; move_knots=true) # Deduplicate cum_sfh values to avoid issues with interpolation
    itp = interpolate((cum_sfh,), exp10.(unique_logAge) ./ 1e9, Gridded(Linear()))
    return itp
end

"""
    tau(τ, unique_logAge, max_logAge, cum_sfh [, lower, upper])

Returns the lookback time (in Gyr) at which `τ` percent of the total stellar mass of a galaxy formed.
See [`tau_interp`](@ref StarFormationHistories.tau_interp) for the underlying interpolator function.

# Arguments
 - `τ`: Fraction of a galaxy's total stellar mass for which you wish to know *when* the galaxy had that much mass (e.g., 0.50 for `τ_50`). 
 - `unique_logAge` should contain the list of unique `log10(age [yr])` for which the SFH solution was derived.
 - `max_logAge` should be the maximum `log10(age [yr])` that you wish to consider -- this sets when the stellar mass of the galaxy was 0.
 - `cum_sfh` is an array of the cumulative stellar mass of the galaxy corresponding to the lookback times in `unique_logAge`; this can be calculated with [`calculate_cum_sfr`](@ref StarFormationHistories.calculate_cum_sfr) or [`cum_sfr_quantiles`](@ref StarFormationHistories.cum_sfr_quantiles).

# Optional Arguments
 - `lower` is an array of the cumulative stellar mass of the galaxy at some lower confidence level (e.g., 16% for 1-σ); if provided, the function will also return the lower bound on `t`.
 - `upper` is an array of the cumulative stellar mass of the galaxy at some higher confidence level (e.g., 84% for 1-σ); if provided, the function will also return the upper bound on `t`.

# Returns
 - `t::Number` the lookback time in Gyr when the galaxy's stellar mass was `τ` times its total birth stellar mass.
If `lower` and `upper` are provided, a tuple of three numbers is returned: `(t_lower, t_mle, t_upper)`.

# Examples
First we will verify that `tau` returns the correct lookback time when the requested `τ` value corresponds exactly to a value in `cum_sfh`:

```jldoctest tau
julia> unique_logAge = [8.0, 8.5, 9.0, 9.5, 10.0];

julia> max_logAge = 10.13;

julia> cum_sfh = [1.0, 0.8, 0.5, 0.2, 0.1];

julia> StarFormationHistories.tau(0.5, unique_logAge, max_logAge, cum_sfh) # Exact result
1.0
```

Now we will show an example where `τ` does not correspond exactly to a value in `cum_sfh`, in which case the function will interpolate between the two nearest values in `cum_sfh` to return the lookback time corresponding to the requested `τ`. The interpolation is performed in linear age space.

```jldoctest tau
julia> cum_sfh = [1.0, 0.8, 0.4, 0.2, 0.1]; # Interpolation between log(age) = 8.5 and 9.0

julia> StarFormationHistories.tau(0.5, unique_logAge, max_logAge, cum_sfh) ≈ 0.8290569415042095
true
```

We can also pass multiple `τ` values at once:

```jldoctest tau
julia> StarFormationHistories.tau([0.5, 0.75], unique_logAge, max_logAge, cum_sfh) ≈ [0.8290569415042095, 0.40169929526473325]
true
```

This will also work with reversed `cum_sfh` and `unique_logAge`:

```jldoctest tau
julia> StarFormationHistories.tau(0.5, reverse(unique_logAge), max_logAge, reverse(cum_sfh)) ≈ 0.8290569415042095
true
```

Now we can pass in lower and upper estimates on the cumulative stellar mass to get confidence intervals on `τ`; the result is returned
as a size `(length(τ), 3)` matrix where the first column is the lower bound on `τ`, the second column is the best estimate of `τ`, and the third column is the upper bound on `τ`:

```jldoctest tau
julia> upper = [1.0, 0.85, 0.6, 0.3, 0.15];

julia> lower = [1.0, 0.75, 0.3, 0.1, 0.05];

julia> StarFormationHistories.tau(0.5, unique_logAge, max_logAge, cum_sfh, lower, upper) ≈ [0.6961012293408169  0.8290569415042095  1.7207592200561264]
true
```

And calling with a vector of `τ` values also works in this case:

```jldoctest tau
julia> StarFormationHistories.tau([0.5, 0.75], unique_logAge, max_logAge, cum_sfh, lower, upper) ≈ [0.6961012293408169  0.8290569415042095  1.7207592200561264; 0.31622776601683794  0.40169929526473325  0.5897366596101027]
true
```
"""
function tau(τ, unique_logAge, max_logAge, cum_sfh)
    itp = tau_interp(unique_logAge, max_logAge, cum_sfh)
    return itp.(τ) # Broadcast the interpolator over τ
end
function tau(τ, unique_logAge, max_logAge, cum_sfh, lower, upper)
    itp_lower = tau_interp(unique_logAge, max_logAge, lower)
    itp = tau_interp(unique_logAge, max_logAge, cum_sfh)
    itp_upper = tau_interp(unique_logAge, max_logAge, upper)
    t = itp.(τ)
    t_lower = itp_lower.(τ)
    t_upper = itp_upper.(τ)
    # Package into a (length(τ), 3) matrix where the first column is the lower bound, the second column is the best estimate, and the third column is the upper bound
    return reduce(hcat, (t_lower, t, t_upper))
end

"""
    tau(result, τ, logAge, MH, max_logAge; Nsamples=10_000, q=(0.16, 0.5, 0.84), kws...)

Returns the lookback time (in Gyr) at which `τ` percent of the total stellar mass of a galaxy formed with upper and lower uncertainty estimates. Uses the solution `result` to draw samples of the cumulative star formation history and then calculates quantiles across those samples.

# Arguments
 - `result::Union{CompositeBFGSResult, BFGSResult}` is a BFGS result object as returned by [`fit_sfh`](@ref), for example, whose contents will be used to sample random, independent star formation histories via [`cum_sfr_quantiles`](@ref StarFormationHistories.cum_sfr_quantiles).
 - `τ`: Fraction of a galaxy's total stellar mass for which you wish to know *when* the galaxy had that much mass (e.g., 0.50 for `τ_50`). Can also be a vector of multiple `τ` values (e.g., `[0.5, 0.75, 0.9]`). 
 - `logAge::AbstractVector` is a vector giving the log10(age [yr]) of the stellar populations that were used to derive `result`. For the purposes of calculating star formation rates, these are assumed to be left-bin edges.
 - `MH::AbstractVector` is a vector giving the metallicities of the stellar populations that were used to derive `result`.
 - `max_logAge` should be the maximum `log10(age [yr])` that you wish to consider -- this sets when the stellar mass of the galaxy was 0.

# Keyword Arguments
 - `Nsamples::Integer` is the number of random, independent star formation histories to draw from `result` to use when calculating quantiles.
 - `q` is passed to [`cum_sfr_quantiles`](@ref StarFormationHistories.cum_sfr_quantiles) to calculate quantiles on the cumulative star formation history samples. This must be a length 3 tuple of quantiles (e.g., `(0.16, 0.5, 0.84)`) to get lower, best, and upper estimates on `τ`. It is not recommended that you change this.
 - `kws...` are passed to [`calculate_cum_sfr`](@ref StarFormationHistories.calculate_cum_sfr), see that method's documentation for more information.

# Returns
 - `t::Matrix`, a size `(length(τ), 3)` matrix where the first column is the lower bound on `τ`, the second column is the best estimate of `τ`, and the third column is the upper bound on `τ`.
"""
function tau(result::Union{BFGSResult, CompositeBFGSResult}, τ, logAge, MH, max_logAge; Nsamples::Int=10_000, q=(0.16, 0.5, 0.84), kws...)
    @argcheck length(q) == 3 "q must be a tuple of three quantiles (e.g., (0.16, 0.5, 0.84)) to get lower, best, and upper estimates on `τ`."
    # Use cum_sfr_quantiles to draw samples and calculate quantiles for the cumulative SFH
    T_max = exp10(max_logAge) / 1e9
    _result = cum_sfr_quantiles(result, logAge, MH, T_max, Nsamples, q; kws...)
    cum_sfh_mat = _result[1]
    return tau(τ, unique(logAge), max_logAge, cum_sfh_mat[:,2], cum_sfh_mat[:,1], cum_sfh_mat[:,3])
end
