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
    Threads.@threads for i in 1:Nsamples
        r = view(samples, :, i)
        new_MH_model = update_params(MH_model, @view(r[Nbins+1:end-npar_disp_model]))
        new_disp_model = update_params(disp_model, @view(r[end-npar_disp_model+1:end]))
        tmp_coeffs = calculate_coeffs(new_MH_model, new_disp_model, @view(r[begin:Nbins]),
                                      logAge, MH)
        # We are doing a lot of extra work in calculate_cum_sfr
        # that we could do once here, but it would require a bespoke implementation
        _, mdf_1, mdf_2, mdf_3 = calculate_cum_sfr(tmp_coeffs, logAge, MH, T_max; kws...)
        cum_sfh_mat[i,:] .= mdf_1
        sfrs_mat[i,:] .= mdf_2
        mean_mh_mat[i,:] .= mdf_3
    end

    # Allocate matrices to accumulate quantiles
    cum_sfh_q = Matrix{Float64}(undef, Nbins, length(q))
    sfrs_q = similar(cum_sfh_q)
    mean_mh_q = similar(cum_sfh_q)
    # Calculate quantiles on samples
    Threads.@threads for i in 1:Nbins
        cum_sfh_q[i,:] .= quantile(view(cum_sfh_mat, :, i), q)
        sfrs_q[i,:] .= quantile(view(sfrs_mat, :, i), q)
        mean_mh_q[i,:] .= quantile(view(mean_mh_mat, :, i), q)
    end
    # cum_sfh_quantiles = [quantile(row, q) for row in eachrow(cum_sfh)]
    # cum_sfh_quantiles = [SVector(quantile(row, q)) for row in eachrow(cum_sfh)]
    # cum_sfh_quantiles = tups_to_mat([quantile(row, q) for row in eachrow(cum_sfh)])
    # return cum_sfh_quantiles
    return (cum_sfh = cum_sfh_q, sfrs = sfrs_q, mean_mh = mean_mh_q, samples = samples)

end
# function cum_sfr_quantiles(result::CompositeBFGSResult, logAge::AbstractVector{<:Number},
#                            MH::AbstractVector{<:Number}, T_max::Number, Nsamples::Integer, q;
#                            kws...)
#     MLE, MAP = result.mle, result.map
#     MH_model, disp_model = MLE.MH_model, MLE.disp_model
    
#     # Get number of unique time bins
#     npar_MH_model = nparams(MLE.MH_model)
#     npar_disp_model = nparams(MLE.disp_model)
#     Nbins = length(MLE.μ) - npar_MH_model - npar_disp_model

#     # Generate samples
#     samples = rand(result, Nsamples)
    
#     # Allocate matrices to accumulate cumulative SFHs, SFRs, and <[M/H]> for
#     # all samples
#     cum_sfh_mat = Matrix{Float64}(undef, Nsamples, Nbins)
#     sfrs_mat = similar(cum_sfh_mat)
#     mean_mh_mat = similar(cum_sfh_mat)
#     Threads.@threads for i in 1:Nsamples
#         r = view(samples, :, i)
#         new_MH_model = update_params(MH_model, @view(r[Nbins+1:end-npar_disp_model]))
#         new_disp_model = update_params(disp_model, @view(r[end-npar_disp_model+1:end]))
#         tmp_coeffs = calculate_coeffs(new_MH_model, new_disp_model, @view(r[begin:Nbins]),
#                                       logAge, MH)
#         # We are doing a lot of extra work in calculate_cum_sfr
#         # that we could do once here, but it would require a bespoke implementation
#         _, mdf_1, mdf_2, mdf_3 = calculate_cum_sfr(tmp_coeffs, logAge, MH, T_max; kws...)
#         cum_sfh_mat[i,:] .= mdf_1
#         sfrs_mat[i,:] .= mdf_2
#         mean_mh_mat[i,:] .= mdf_3
#     end

#     # Allocate matrices to accumulate quantiles
#     cum_sfh_q = Matrix{Float64}(undef, Nbins, length(q))
#     sfrs_q = similar(cum_sfh_q)
#     mean_mh_q = similar(cum_sfh_q)
#     # Calculate quantiles on samples
#     Threads.@threads for i in 1:Nbins
#         cum_sfh_q[i,:] .= quantile(view(cum_sfh_mat, :, i), q)
#         sfrs_q[i,:] .= quantile(view(sfrs_mat, :, i), q)
#         mean_mh_q[i,:] .= quantile(view(mean_mh_mat, :, i), q)
#     end
#     # cum_sfh_quantiles = [quantile(row, q) for row in eachrow(cum_sfh)]
#     # cum_sfh_quantiles = [SVector(quantile(row, q)) for row in eachrow(cum_sfh)]
#     # cum_sfh_quantiles = tups_to_mat([quantile(row, q) for row in eachrow(cum_sfh)])
#     # return cum_sfh_quantiles
#     return (cum_sfh = cum_sfh_q, sfrs = sfrs_q, mean_mh = mean_mh_q, samples = samples)

# end
