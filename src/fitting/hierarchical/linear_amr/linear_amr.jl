# Linear age-metallicity relation with constant Gaussian spread σ

"""
    x0::Vector = construct_x0_mdf(logAge::AbstractVector{T},
                                  [ cum_sfh, ]
                                  T_max::Number;
                                  normalize_value::Number = one(T)) where T <: Number

Generates a vector of initial stellar mass normalizations for input to [`StarFormationHistories.fit_templates_mdf`](@ref) or [`StarFormationHistories.hmc_sample_mdf`](@ref) with a total stellar mass of `normalize_value`. The `logAge` vector must contain the `log10(Age [yr])` of each isochrone that you are going to input as models. If `cum_sfh` is not provided, a constant star formation rate is assumed. For the purposes of computing the constant star formation rate, the provided `logAge` are treated as left-bin edges, with the final right-bin edge being `T_max`, which has units of Gyr. For example, you might have `logAge=[6.6, 6.7, 6.8]` in which case a final logAge of 6.9 would give equal bin widths (in log-space). In this case you would set `T_max = exp10(6.9) / 1e9 ≈ 0.0079` so that the width of the final bin for the star formation rate calculation has the same `log10(Age [yr])` step as the other bins.

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
    minlog, maxlog = extrema(logAge)
    max_logAge = log10(T_max) + 9 # T_max in units of Gyr
    @assert max_logAge > maxlog   # max_logAge has to be greater than the maximum of logAge vector
    sfr = normalize_value / (exp10(max_logAge) - exp10(minlog)) # Average SFR / yr
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
    cmin, cmax = extrema(cum_sfh)
    @assert cmin ≥ zero(T)
    !isapprox(cmax, one(T)) && @warn "Maximum of `cum_sfh` argument is $cmax which is not approximately equal to 1."
    maximum(cum_sfh) 
    minlog, maxlog = extrema(logAge)
    max_logAge = log10(T_max) + 9 # T_max in units of Gyr
    @assert max_logAge > maxlog   # max_logAge has to be greater than the maximum of logAge vector
    unique_logAge = unique(logAge)
    @assert length(unique_logAge) == length(cum_sfh)
    
    idxs = sortperm(unique_logAge)
    sorted_cum_sfh = vcat(cum_sfh[idxs], zero(T))
    # Test that cum_sfh is properly monotonic
    @assert(all(sorted_cum_sfh[i] ≤ sorted_cum_sfh[i-1] for i in eachindex(sorted_cum_sfh)[2:end]),
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
    @assert(length(cum_sfh_vec) == 2,
            "`cum_sfh` must either be a vector of numbers or vector containing two vectors that \
             define a cumulative SFH with a different log(age) discretization than the provided \
             `logAge` argument.")
    # Extract cum_sfh info and concatenate with T_max where cum_sfh=0 by definition
    @assert(maximum(last(cum_sfh_vec)) ≤ one(T),
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

"""
    LogTransformMDFσResult(μ::AbstractVector{<:Number},
                           σ::AbstractVector{<:Number},
                           invH::AbstractMatrix{<:Number},
                           result)

Type for containing the maximum likelihood estimate (MLE) and maximum a posteriori (MAP) results from [`fit_templates_mdf`](@ref) for fixed `σ`. The fitted coefficients are available in the `μ` field. Estimates of the standard errors are available in the `σ` field. These have both been transformed from the native logarithmic fitting space into natural units (i.e., stellar mass or star formation rate). The linear age-metallicity relation parameters `α` (slope [dex/Gyr]) and `β` (intercept at `T_max` [dex]) are available in the second-to-last and last elements of `μ` and `σ`, respectively. 

`invH` contains the estimated inverse Hessian of the likelihood / posterior at the maximum point in the logarithmic fitting units. `result` is the full result object returned by the optimization routine.

This type is implemented as a subtype of `Distributions.Sampleable{Multivariate, Continuous}` to enable sampling from an estimate of the likelihood / posterior distribution. We approximate the distribution as a multivariate Gaussian in the native (logarithmically transformed) fitting variables with covariance matrix `invH` and means `log.(μ)`. We find this approximation is good for the MAP result but less robust for the MLE. You can obtain `N::Integer` samples from the distribution by `rand(R, N)` where `R` is an instance of this type; this will return a size `(length(μ)+2) x N` matrix, or fail if `invH` is not positive definite.

# Examples
```julia-repl
julia> result = fit_templates_mdf(models, data, model_logAge, model_MH, 0.3);

julia> typeof(result.map)
StarFormationHistories.LogTransformMDFσResult{...}

julia> size(rand(result.map, 3)) == (length(models)+2,3)
true
```
"""
struct LogTransformMDFσResult{S <: AbstractVector{<:Number},
                              T <: AbstractVector{<:Number},
                              U <: AbstractMatrix{<:Number},
                              V} <: Sampleable{Multivariate, Continuous}
    μ::S
    σ::T
    invH::U
    result::V
end
Base.length(result::LogTransformMDFσResult) = length(result.μ)
function _rand!(rng::AbstractRNG, result::LogTransformMDFσResult, x::Union{AbstractVector{T}, DenseMatrix{T}}) where T <: Real
    dist = MvNormal(Optim.minimizer(result.result), Hermitian(result.invH))
    _rand!(rng, dist, x)
    for i in axes(x,1)[begin:end-1]
        for j in axes(x,2)
            x[i,j] = exp(x[i,j])
        end
    end
    return x
end

_gausspdf(x,μ,σ) = exp( -((x-μ)/σ)^2 / 2 )  # Unnormalized, 1-D Gaussian PDF
# _gausstest(x,age,α,β,σ) = inv(σ) * exp( -((x-(α*age+β))/σ)^2 / 2 )
# ForwardDiff.derivative(x -> _gausstest(-1.0, 1e9, -1e-10, x, 0.2), -0.4) = -2.74
# _dgaussdβ(x,age,α,β,σ) = (μ = α*age+β; (x-μ) * exp( -((x-μ)/σ)^2 / 2 ) * inv(σ)^3)
# _dgaussdβ(-1.0,1e9,-1e-10,-0.4,0.2) = -2.74

"""
    calculate_coeffs_mdf(variables::AbstractVector{<:Number},
                         logAge::AbstractVector{<:Number},
                         metallicities::AbstractVector{<:Number},
                         T_max::Number
                         α::Number,
                         β::Number,
                         σ::Number,)
    calculate_coeffs_mdf(variables::AbstractVector{<:Number},
                         logAge::AbstractVector{<:Number},
                         metallicities::AbstractVector{<:Number},
                         T_max::Number)

Calculates per-model stellar mass coefficients `coeffs` from the fitting parameters of [`StarFormationHistories.fit_templates_mdf`](@ref) and [`StarFormationHistories.hmc_sample_mdf`](@ref). The `variables` returned by these functions is of length `length(unique(logAge))+3`. The first `length(logAge)` entries are stellar mass coefficients, one per unique entry in `logAge`. The final three elements are α, β, and σ defining a metallicity evolution such that the mean for element `i` of `unique(logAge)` is `μ[i] = α * (T_max - exp10(unique(logAge)[i]) / 1e9) + β`. The individual weights per each isochrone are then determined via Gaussian weighting with the above mean and the provided `σ`. The second call signature can be used on samples that include α, β, and σ.

# Examples
```jldoctest; setup = :(import StarFormationHistories: calculate_coeffs_mdf)
julia> calculate_coeffs_mdf([1,1], [7,7,8,8], [-2,-1,-2,-1], 12, 0.05, -2.0, 0.2) ≈ [ 0.07673913563377144, 0.9232608643662287, 0.08509904500701986, 0.9149009549929802 ]
true
```
"""
function calculate_coeffs_mdf(variables::AbstractVector{<:Number}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}, T_max::Number, α::Number, β::Number, σ::Number) # =exp10(maximum(logAge))/1e9)
    S = promote_type(eltype(variables), eltype(logAge), eltype(metallicities), typeof(α), typeof(β), typeof(σ), typeof(T_max))
    # Compute the coefficients on each model template given the `variables` and the MDF
    unique_logAge = unique(logAge)
    @assert length(variables) == length(unique_logAge)
    coeffs = Vector{S}(undef,length(logAge))
    norm_vals = Vector{S}(undef,length(unique_logAge))
    for i in eachindex(unique_logAge)
        la = unique_logAge[i]
        μ = α * (T_max - exp10(la) / 1e9) + β # Find the mean metallicity of this age bin
        idxs = findall( ==(la), logAge) # Find all entries that match this logAge
        tmp_coeffs = [_gausspdf(metallicities[j], μ, σ) for j in idxs] # Calculate relative weights
        A = sum(tmp_coeffs)
        norm_vals[i] = A
        # Make sure sum over tmp_coeffs equals 1 and write to coeffs
        coeffs[idxs] .= tmp_coeffs .* variables[i] ./ A
    end
    return coeffs
end
calculate_coeffs_mdf(variables, logAge, metallicities, T_max) =
    calculate_coeffs_mdf(view(variables,firstindex(variables):lastindex(variables)-3),
                         logAge, metallicities, T_max, variables[end-2], variables[end-1], variables[end])



# variables[begin:end-2] are stellar mass coefficients
# variables[end-1] is the slope of the age-MH relation in [MH] / [10Gyr; (lookback)], e.g. -1.0
# variables[end] is the intercept of the age-MH relation in MH at present-day, e.g. -0.4
@inline function fg_mdf_fixedσ!(F, G, variables::AbstractVector{<:Number}, models::Union{AbstractMatrix{<:Number}, AbstractVector{<:AbstractMatrix{<:Number}}}, data::Union{AbstractVector{<:Number}, AbstractMatrix{<:Number}}, composite::Union{AbstractVector{<:Number}, AbstractMatrix{<:Number}}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}, T_max::Number, σ::Number)
    # `variables` should have length `length(unique(logAge)) + 2`; coeffs for each unique
    # entry in logAge, plus α and β to define the MDF at fixed logAge
    @assert axes(data) == axes(composite)
    S = promote_type(eltype(variables), eltype(eltype(models)), eltype(eltype(data)), eltype(composite), eltype(logAge), eltype(metallicities))
    # Compute the coefficients on each model template given the `variables` and the MDF
    α, β = variables[end-1], variables[end]
    unique_logAge = unique(logAge)
    coeffs = calculate_coeffs_mdf(view(variables,firstindex(variables):lastindex(variables)-2), logAge, metallicities, T_max, α, β, σ)
    # Fill the composite array with the equivalent of sum( coeffs .* models )
    # composite = sum( coeffs .* models )
    # return -loglikelihood(composite, data)
    composite!(composite, coeffs, models)
    logL = loglikelihood(composite, data)
    if (F != nothing) & (G != nothing) # Optim.optimize wants objective and gradient
        @assert axes(G) == axes(variables)
        # Calculate the ∇loglikelihood with respect to model coefficients; we will need all of these
        fullG = Vector{eltype(G)}(undef, length(coeffs)) # length(models))
        ∇loglikelihood!(fullG, composite, models, data)
        # Now need to do the transformation to the `variables` rather than model coefficients
        G[end-1] = zero(eltype(G))
        G[end] = zero(eltype(G))
        for i in axes(G,1)[begin:end-2] 
            la = unique_logAge[i]
            age = exp10(la) / 1e9 # the / 1e9 makes α the slope in MH/Gyr, improves convergence
            μ = α * (T_max - age) + β # Find the mean metallicity of this age bin
            idxs = findall( ==(la), logAge) # Find all entries that match this logAge
            tmp_coeffs = [_gausspdf(metallicities[j], μ, σ) for j in idxs] # Calculate relative weights
            A = sum(tmp_coeffs)
            # This should be correct for any MDF model at fixed logAge
            @inbounds G[i] = -sum( fullG[j] * coeffs[j] / variables[i] for j in idxs )
            βsum = sum( ((metallicities[idxs[j]]-μ) * tmp_coeffs[j]) for j in eachindex(idxs))
            dLdβ = -sum( fullG[idxs[j]] * variables[i] *
                ( ((metallicities[idxs[j]]-μ) * tmp_coeffs[j]) - tmp_coeffs[j] / A * βsum )
                         for j in eachindex(idxs)) / A / σ^2
            dLdα = dLdβ * (T_max - age)
            G[end-1] += dLdα
            G[end] += dLdβ
        end
        return -logL
    elseif F != nothing # Optim.optimize wants only objective
        return -logL
    end
end

# for fixed σ
function fit_templates_mdf(models::AbstractMatrix{S},
                           data::AbstractVector{<:Number},
                           logAge::AbstractVector{<:Number},
                           metallicities::AbstractVector{<:Number},
                           T_max::Number,
                           σ::Number;
                           x0=vcat(construct_x0_mdf(logAge, convert(S,13.7)), [0.05, -2.0]),
                           kws...) where {S <: Number}
    unique_logage = unique(logAge)
    @assert length(x0) == length(unique_logage)+2
    @assert size(models,1) == length(data)
    @assert size(models,2) == length(logAge) == length(metallicities)
    composite = Vector{S}(undef,length(data)) # Scratch matrix for storing complex Hess model
    # Perform logarithmic transformation on the provided x0 for all variables except β
    x0 = copy(x0) # We don't actually want to modify x0 externally to this program, so copy
    for i in eachindex(x0)[begin:end-1]
        x0[i] = log(x0[i])
    end
    # Make scratch array for assessing transformations
    x = similar(x0)
    # Define wrapper function to pass to Optim.only_fg!
    # It looks like you don't need the Jacobian correction to arrive at the maximum likelihood
    # result, and if you remove the Jacobian corrections it actually converges to the non-log-transformed case.
    # However, the uncertainty estimates from the inverse Hessian don't seem reliable without the
    # Jacobian corrections.
    function fg_mdf_fixedσ!_map(F, G, xvec)
        for i in eachindex(xvec)[begin:end-1] # These are the per-logage stellar mass coefficients and α
            x[i] = exp(xvec[i])
        end
        x[end] = xvec[end] # β

        logL = fg_mdf_fixedσ!(F, G, x, models, data, composite, logAge, metallicities, T_max, σ)
        logL -= sum( @view xvec[begin:end-1] ) # this is the Jacobian correction
        # Add the Jacobian correction for every element of G except β (x[end-1])
        for i in eachindex(G)[begin:end-1]
            G[i] = G[i] * x[i] - 1
        end
        return logL
    end
    function fg_mdf_fixedσ!_mle(F, G, xvec)
        for i in eachindex(xvec)[begin:end-1] # These are the per-logage stellar mass coefficients and α
            x[i] = exp(xvec[i])
        end
        x[end] = xvec[end] # β

        logL = fg_mdf_fixedσ!(F, G, x, models, data, composite, logAge, metallicities, T_max, σ)
        for i in eachindex(G)[begin:end-1]
            G[i] = G[i] * x[i]
        end
        return logL
    end
    # The InitialStatic(1.0,true) alphaguess helps to regularize the optimization and 
    # makes it less sensitive to initial x0.
    bfgs_struct = Optim.BFGS(; alphaguess=LineSearches.InitialStatic(1.0,true), linesearch=LineSearches.HagerZhang())
    # The extended trace will contain the BFGS estimate of the inverse Hessian, aka the
    # covariance matrix, which we can use to make parameter uncertainty estimates
    bfgs_options = Optim.Options(; allow_f_increases=true, store_trace=true, extended_trace=true, kws...)
    # Calculate results
    result_map = Optim.optimize(Optim.only_fg!( fg_mdf_fixedσ!_map ), x0, bfgs_struct, bfgs_options)
    result_mle = Optim.optimize(Optim.only_fg!( fg_mdf_fixedσ!_mle ), Optim.minimizer(result_map), bfgs_struct, bfgs_options)
    # Transform the resulting variables
    μ_map = deepcopy( Optim.minimizer(result_map) )
    μ_mle = deepcopy( Optim.minimizer(result_mle) )
    for i in eachindex(μ_map,μ_mle)[begin:end-1]
        μ_map[i] = exp(μ_map[i])
        μ_mle[i] = exp(μ_mle[i])
    end
    # Estimate parameter uncertainties from the inverse Hessian approximation
    σ_map = sqrt.(diag(Optim.trace(result_map)[end].metadata["~inv(H)"]))
    σ_mle = sqrt.(diag(Optim.trace(result_mle)[end].metadata["~inv(H)"]))
    # Need to account for the logarithmic transformation
    for i in eachindex(σ_map,σ_mle)[begin:end-1]
        σ_map[i] = μ_map[i] * σ_map[i]
        σ_mle[i] = μ_mle[i] * σ_mle[i]
    end
    return (map = LogTransformMDFσResult(μ_map, σ_map, Optim.trace(result_map)[end].metadata["~inv(H)"], result_map),
            mle = LogTransformMDFσResult(μ_mle, σ_mle, Optim.trace(result_mle)[end].metadata["~inv(H)"], result_mle))
    # return (map = (μ = μ_map, σ = σ_map, result = result_map),
    #         mle = (μ = μ_mle, σ = σ_mle, result = result_mle))
end
fit_templates_mdf(models::AbstractVector{<:AbstractMatrix{<:Number}}, data::AbstractMatrix{<:Number}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}, T_max::Number, σ::Number; kws...) = fit_templates_mdf(stack_models(models), vec(data), logAge, metallicities, T_max, σ; kws...)

"""
    LogTransformMDFResult(μ::AbstractVector{<:Number},
                          σ::AbstractVector{<:Number},
                          invH::AbstractMatrix{<:Number},
                          result)

Type for containing the maximum likelihood estimate (MLE) and maximum a posteriori (MAP) results from [`fit_templates_mdf`](@ref) when freely fitting `σ`. The fitted coefficients are available in the `μ` field. Estimates of the standard errors are available in the `σ` field. These have both been transformed from the native logarithmic fitting space into natural units (i.e., stellar mass or star formation rate). The linear age-metallicity relation parameters `α` (slope [dex/Gyr]) and `β` (intercept at `T_max` [dex]) are available in the third-to-last and second-to-last elements of `μ` and `σ`, respectively. The static Gaussian width of the MDF at fixed age is provided in the last element of `μ` and `σ`. 

`invH` contains the estimated inverse Hessian of the likelihood / posterior at the maximum point in the logarithmic fitting units. `result` is the full result object returned by the optimization routine.

This type is implemented as a subtype of `Distributions.Sampleable{Multivariate, Continuous}` to enable sampling from an estimate of the likelihood / posterior distribution. We approximate the distribution as a multivariate Gaussian in the native (logarithmically transformed) fitting variables with covariance matrix `invH` and means `log.(μ)`. We find this approximation is good for the MAP result but less robust for the MLE. You can obtain `N::Integer` samples from the distribution by `rand(R, N)` where `R` is an instance of this type; this will return a size `(length(μ)+3) x N` matrix, or fail if `invH` is not positive definite.

# Examples
```julia-repl
julia> result = fit_templates_mdf(models, data, model_logAge, model_MH);

julia> typeof(result.map)
StarFormationHistories.LogTransformMDFσResult{...}

julia> size(rand(result.map, 3)) == (length(models)+3,3)
true
```
"""
struct LogTransformMDFResult{S <: AbstractVector{<:Number},
                             T <: AbstractVector{<:Number},
                             U <: AbstractMatrix{<:Number},
                             V} <: Sampleable{Multivariate, Continuous}
    μ::S
    σ::T
    invH::U
    result::V
end
Base.length(result::LogTransformMDFResult) = length(result.μ)
function _rand!(rng::AbstractRNG, result::LogTransformMDFResult, x::Union{AbstractVector{T}, DenseMatrix{T}}) where T <: Real
    dist = MvNormal(Optim.minimizer(result.result), Hermitian(result.invH))
    _rand!(rng, dist, x)
    for i in axes(x,1)[begin:end-2] # Only β = x[end-1] is not log-transformed
        for j in axes(x,2)
            x[i,j] = exp(x[i,j])
        end
    end
    # Transform the σ
    for j in axes(x,2)
        x[end,j] = exp(x[end,j])
    end
    return x
end

# fits σ as free parameter
@inline function fg_mdf!(F, G, variables::AbstractVector{<:Number}, models::Union{AbstractMatrix{<:Number}, AbstractVector{<:AbstractMatrix{<:Number}}}, data::Union{AbstractVector{<:Number}, AbstractMatrix{<:Number}}, composite::Union{AbstractVector{<:Number}, AbstractMatrix{<:Number}}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}, T_max::Number)
    # `variables` should have length `length(unique(logAge)) + 3`; coeffs for each unique
    # entry in logAge, plus α and β to define the MDF at fixed logAge and σ to define Gaussian width
    @assert axes(data) == axes(composite)
    S = promote_type(eltype(variables), eltype(eltype(models)), eltype(eltype(data)), eltype(composite), eltype(logAge), eltype(metallicities), typeof(T_max))
    # Compute the coefficients on each model template given the `variables` and the MDF
    α, β, σ = variables[end-2], variables[end-1], variables[end]
    unique_logAge = unique(logAge)
    # Calculate the per-template coefficents and normalization values
    coeffs = calculate_coeffs_mdf(view(variables,firstindex(variables):lastindex(variables)-3), logAge, metallicities, T_max, α, β, σ)

    # Fill the composite array with the equivalent of sum( coeffs .* models )
    # composite = sum( coeffs .* models )
    # return -loglikelihood(composite, data)
    composite!(composite, coeffs, models)
    logL = loglikelihood(composite, data) # Need to do this before ∇loglikelihood! because it will overwrite composite
    if (F != nothing) & (G != nothing) # Optim.optimize wants objective and gradient
        @assert axes(G) == axes(variables)
        # Calculate the ∇loglikelihood with respect to model coefficients; we will need all of these
        fullG = Vector{eltype(G)}(undef, length(coeffs)) # length(models))
        ∇loglikelihood!(fullG, composite, models, data)
        # Now need to do the transformation to the `variables` rather than model coefficients
        G[end-2] = zero(eltype(G))
        G[end-1] = zero(eltype(G))
        G[end] = zero(eltype(G))
        for i in axes(G,1)[begin:end-3] 
            la = unique_logAge[i]
            age = exp10(la) / 1e9 # the / 1e9 makes α the slope in MH/Gyr, improves convergence
            μ = α * (T_max - age) + β # Find the mean metallicity of this age bin
            idxs = findall( ==(la), logAge) # Find all entries that match this logAge
            tmp_coeffs = [_gausspdf(metallicities[j], μ, σ) for j in idxs] # Calculate relative weights
            A = sum(tmp_coeffs)
            # This should be correct for any MDF model at fixed logAge
            @inbounds G[i] = -sum( fullG[j] * coeffs[j] / variables[i] for j in idxs )
            βsum = sum( ((metallicities[idxs[j]]-μ) * tmp_coeffs[j]) for j in eachindex(idxs))
            dLdβ = -sum( fullG[idxs[j]] * variables[i] *
                ( ((metallicities[idxs[j]]-μ) * tmp_coeffs[j]) - tmp_coeffs[j] / A * βsum )
                         for j in eachindex(idxs)) / A / σ^2
            dLdα = dLdβ * (T_max - age)
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

"""
    fit_templates_mdf(models::AbstractVector{<:AbstractMatrix{S}},
                      data::AbstractMatrix{<:Number},
                      logAge::AbstractVector{<:Number},
                      metallicities::AbstractVector{<:Number},
                      T_max::Number
                      [, σ::Number];
                      x0 = vcat(construct_x0_mdf(logAge, convert(S,13.7)),
                                [0.05, -2.0, 0.2]),
                      kws...) where {S <: Number}
    fit_templates_mdf(models::AbstractMatrix{S},
                      data::AbstractVector{<:Number},
                      logAge::AbstractVector{<:Number},
                      metallicities::AbstractVector{<:Number},
                      T_max::Number
                      [, σ::Number];
                      x0 = vcat(construct_x0_mdf(logAge, convert(S,13.7)),
                                [0.05, -2.0, 0.2]),
                      kws...) where {S <: Number}

Method that fits a linear combination of the provided Hess diagrams `models` to the observed Hess diagram `data`, constrained to have a linear age-metallicity relation with the mean metallicity of element `i` of `unique(logAge)` being `μ[i] = α * (T_max - exp10(unique(logAge)[i]) / 1e9) + β`. `α` is therefore a slope in the units of `metallicities` per Gyr, and `β` is the mean metallicity value of stars being born at a lookback time of `T_max`, which has units of Gyr. Individual weights for each isochrone template are then determined via Gaussian weighting with the above mean and the standard deviation `σ`, which can either be fixed or fit.

This function is designed to work best with a "grid" of stellar models, defined by the outer product of `N` unique entries in `logAge` and `M` unique entries in `metallicities`. See the examples for more information on usage.

The second call signature supports the flattened formats for `models` and `data`. See the notes for the flattened call signature of [`StarFormationHistories.composite!`](@ref) for more details.

# Arguments
 - `models` are the template Hess diagrams for the simple stellar populations that compose the observed Hess diagram. 
 - `data` is the Hess diagram for the observed data. 
 - `logAge::AbstractVector{<:Number}` is the vector containing the effective ages of the stellar populations used to create the templates in `models`, in units of `log10(age [yr])`. For example, if a population has an age of 1 Myr, its entry in `logAge` should be `log10(10^6) = 6.0`.
 - `metallicities::AbstractVector{<:Number}` is the vector containing the effective metallicities of the stellar populations used to create the templates in `models`. This is most commonly a logarithmic abundance like [M/H] or [Fe/H], but you could use a linear abundance like the metal mass fraction Z if you wanted to. There are some notes on the [Wikipedia](https://en.wikipedia.org/wiki/Metallicity) that might be useful.
 - `T_max::Number` is the time at which the age-metallicity relation has a value of `\beta` in Gyr. For example, if the oldest stellar populations in your isochrone grid are 12 Gyr old, you could set `T_max = 12.0`. 

# Optional Arguments
 - If provided, `σ::Number` is the fixed width of the Gaussian the defines the metallicity distribution function (MDF) at fixed `logAge`. If this argument is omitted, `σ` will be a free parameter in the fit. 

# Keyword Arguments
 - `x0` is the vector of initial guesses for the stellar mass coefficients per *unique* entry in `logAge`, plus the variables that define the metallicity evolution model. You should basically always be calculating and passing this keyword argument. We provide [`StarFormationHistories.construct_x0_mdf`](@ref) to prepare the first part of `x0` assuming constant star formation rate, which is typically a good initial guess. You then have to concatenate that result with an initial guess for the metallicity evolution parameters. For example, `x0=vcat(construct_x0_mdf(logAge, 13.7; normalize_value=1e4), [0.05,-2.0,0.2])`, where `logAge` is a valid argument for this function (see above), and the initial guesses on the parameters are `[α, β, σ] = [0.05, -2.0, 0.2]`. If the provided `metallicities` are, for example, [M/H] values, then this mean metallicity evolution is μ(t) [dex] = 0.05 [dex/Gyr] * (T_max - t) [Gyr] - 2.0 [dex], and at fixed time, the metallicity distribution function is Gaussian with mean μ(t) and standard deviation σ. If you provide `σ` as an optional argument, then you should not include an entry for it in `x0`.
 - Other `kws...` are passed to `Optim.options` to set things like convergence criteria for the optimization.

# Returns
 - This function returns an object (say, `result`) of similar structure to the object returned by [`fit_templates`](@ref). Specifically, this method will return a `NamedTuple` with entries `result.mle` and `result.map` for the maximum likelihood and maximum a posteriori estimates, respectively. If you provide a fixed `σ`, those objects will be instances of [`StarFormationHistories.LogTransformMDFσResult`](@ref). If you allow `σ` to be freely fit, those objects will be instances of [`StarFormationHistories.LogTransformMDFResult`](@ref). Both of these types support sampling via, e.g., `rand(result.map, 10)`. 

# Notes
 - `α` and `σ` are optimized under a logarithmic transformation, so they are constrained to be positive. `β` is not and may be negative. This method also uses the `BFGS` method from `Optim.jl` internally just like [`fit_templates`](@ref); please see the notes section of that method. 
"""
function fit_templates_mdf(models::AbstractMatrix{S},
                           data::AbstractVector{<:Number},
                           logAge::AbstractVector{<:Number},
                           metallicities::AbstractVector{<:Number},
                           T_max::Number;
                           x0 = vcat(construct_x0_mdf(logAge, convert(S,13.7)), [0.05, -2.0, 0.2]),
                           kws...) where {S <: Number}
    unique_logage = unique(logAge)
    @assert length(x0) == length(unique_logage)+3
    @assert size(models,1) == length(data)
    @assert size(models,2) == length(logAge) == length(metallicities)
    composite = Vector{S}(undef,length(data)) # Scratch matrix for storing complex Hess model
    # Perform logarithmic transformation on the provided x0 for all variables except β
    x0 = copy(x0) # We don't actually want to modify x0 externally to this program, so copy
    for i in eachindex(x0)[begin:end-2]
        x0[i] = log(x0[i])
    end
    x0[end] = log(x0[end])
    # Make scratch array for assessing transformations
    x = similar(x0)
    # Define wrapper function to pass to Optim.only_fg!
    # It looks like you don't need the Jacobian correction to arrive at the maximum likelihood
    # result, and if you remove the Jacobian corrections it actually converges to the non-log-transformed case.
    # However, the uncertainty estimates from the inverse Hessian don't seem reliable without the
    # Jacobian corrections.
    function fg_mdf!_map(F, G, xvec)
        for i in eachindex(xvec)[begin:end-2] # These are the per-logage stellar mass coefficients and α
            x[i] = exp(xvec[i])
        end
        x[end-1] = xvec[end-1]  # β
        x[end] = exp(xvec[end]) # σ
        logL = fg_mdf!(F, G, x, models, data, composite, logAge, metallicities, T_max)
        logL -= sum( @view xvec[begin:end-2] ) + xvec[end] # This is the Jacobian correction
        # Add the Jacobian correction for every element of G except β (x[end-1])
        for i in eachindex(G)[begin:end-2]
            G[i] = G[i] * x[i] - 1
        end
        G[end] = G[end] * x[end] - 1
        return logL
    end
    function fg_mdf!_mle(F, G, xvec)
        for i in eachindex(xvec)[begin:end-2] # These are the per-logage stellar mass coefficients and α
            x[i] = exp(xvec[i])
        end
        x[end-1] = xvec[end-1]  # β
        x[end] = exp(xvec[end]) # σ
        logL = fg_mdf!(F, G, x, models, data, composite, logAge, metallicities, T_max)
        for i in eachindex(G)[begin:end-2]
            G[i] = G[i] * x[i]
        end
        G[end] = G[end] * x[end]
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
    result_map = Optim.optimize(Optim.only_fg!( fg_mdf!_map ), x0, bfgs_struct, bfgs_options)
    result_mle = Optim.optimize(Optim.only_fg!( fg_mdf!_mle ), Optim.minimizer(result_map), bfgs_struct, bfgs_options)
    # Transform the resulting variables
    μ_map = deepcopy( Optim.minimizer(result_map) )
    μ_mle = deepcopy( Optim.minimizer(result_mle) )
    for i in eachindex(μ_map,μ_mle)[begin:end-2]
        μ_map[i] = exp(μ_map[i])
        μ_mle[i] = exp(μ_mle[i])
    end
    μ_map[end] = exp(μ_map[end])
    μ_mle[end] = exp(μ_mle[end])

    # Estimate parameter uncertainties from the inverse Hessian approximation
    σ_map = sqrt.(diag(Optim.trace(result_map)[end].metadata["~inv(H)"]))
    σ_mle = sqrt.(diag(Optim.trace(result_mle)[end].metadata["~inv(H)"]))
    # Need to account for the logarithmic transformation
    for i in eachindex(σ_map,σ_mle)[begin:end-2]
        σ_map[i] = μ_map[i] * σ_map[i]
        σ_mle[i] = μ_mle[i] * σ_mle[i]
    end
    σ_map[end] = μ_map[end] * σ_map[end]
    σ_mle[end] = μ_mle[end] * σ_mle[end]
    return (map = LogTransformMDFResult(μ_map, σ_map, Optim.trace(result_map)[end].metadata["~inv(H)"], result_map),
            mle = LogTransformMDFResult(μ_mle, σ_mle, Optim.trace(result_mle)[end].metadata["~inv(H)"], result_mle))
    # return (map = (μ = μ_map, σ = σ_map, result = result_map),
    #         mle = (μ = μ_mle, σ = σ_mle, result = result_mle))
end
fit_templates_mdf(models::AbstractVector{<:AbstractMatrix{<:Number}}, data::AbstractMatrix{<:Number}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}, T_max::Number; kws...) = fit_templates_mdf(stack_models(models), vec(data), logAge, metallicities, T_max; kws...)

# We can even use the inv(H) = covariance matrix estimate to draw samples to compare to HMC
# import Distributions: MvNormal
# result1, std1, fr = fit_templates_mdf(mdf_templates, h.weights, mdf_template_logAge, mdf_template_MH; x0=vcat(construct_x0_mdf(mdf_template_logAge, 13.7; normalize_value=1e4),[-0.1,-0.5,0.3]))
# corner.corner(permutedims(rand(MvNormal(result1,LinearAlgebra.Hermitian(fr.trace[end].metadata["~inv(H)"])),10000)[end-2:end,:]))
# Can we also use this inv(H) estimate as input to HMC? I think that's roughly the M matrix.

######################################################################################
## HMC with MDF

struct HMCModelMDF{T,S,V,W,X,Y,Z}
    models::T
    composite::S
    data::V
    logAge::W
    metallicities::X
    G::Y
    T_max::Z
end

# This model will return loglikelihood and gradient
LogDensityProblems.capabilities(::Type{<:HMCModelMDF}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(problem::HMCModelMDF) = length(problem.G)

function LogDensityProblems.logdensity_and_gradient(problem::HMCModelMDF, xvec)
    composite = problem.composite
    models = problem.models
    data = problem.data
    dims = length(models)
    logAge = problem.logAge
    metallicities = problem.metallicities
    G = problem.G
    T_max = problem.T_max
    @assert axes(G) == axes(xvec)
    # Transform the provided x
    # β (xvec[end-1]) is the only variables that are not log-transformed
    x = similar(xvec)
    for i in eachindex(xvec)[begin:end-2] # These are the per-logage stellar mass coefficients and α
        x[i] = exp(xvec[i])
    end
    x[end-1] = xvec[end-1]  # β
    x[end] = exp(xvec[end]) # σ

    # fg_mdf! returns -logL and fills G with -∇logL so we need to negate the signs.
    # println(x)
    logL = -fg_mdf!(true, G, x, models, data, composite, logAge, metallicities, T_max)
    logL += sum(view(xvec,firstindex(xvec):lastindex(xvec)-2)) + xvec[end] # this is the Jacobian correction
    ∇logL = -G
    # Add the Jacobian correction for every element of ∇logL except β (x[end-1])
    for i in eachindex(∇logL)[begin:end-2]
        ∇logL[i] = ∇logL[i] * x[i] + 1
    end
    ∇logL[end] = ∇logL[end] * x[end] + 1
    return logL, ∇logL
end

# Version with just one chain; no threading
"""
    hmc_sample_mdf(models::AbstractVector{T},
                   data::AbstractMatrix{<:Number},
                   logAge::AbstractVector{<:Number},
                   metallicities::AbstractVector{<:Number},
                   T_max::Number,
                   nsteps::Integer;
                   composite=Matrix{S}(undef,size(data)),
                   rng::Random.AbstractRNG=Random.default_rng(),
                   kws...) where {S <: Number, T <: AbstractMatrix{S}}

Method to sample the posterior of the star formation history coefficients constrained to have a linear age-metallicity relation with the mean metallicity of element `i` of `unique(logAge)` being `μ[i] = α * (T_max - exp10(unique(logAge)[i]) / 1e9) + β`. `α` is therefore a slope in the units of `metallicities` per Gyr, and `β` is the mean metallicity value of stars born at lookback time `T_max` which has units of Gyr. Individual weights for each isochrone template are then determined via Gaussian weighting with the above mean and the standard deviation `σ`, which can either be fixed or fit. This method is essentially an analog of [`StarFormationHistories.fit_templates_mdf`](@ref) that samples the posterior rather than using optimization methods to find the maximum likelihood estimate. This method uses the No-U-Turn sampler as implemented in [DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl), which is a form of dynamic Hamiltonian Monte Carlo.
 
This function is designed to work best with a "grid" of stellar models, defined by the outer product of `N` unique entries in `logAge` and `M` unique entries in `metallicities`. See the examples for more information on usage.

# Arguments
 - `models::AbstractVector{<:AbstractMatrix{<:Number}}` is a vector of equal-sized matrices that represent the template Hess diagrams for the simple stellar populations that compose the observed Hess diagram.
 - `data::AbstractMatrix{<:Number}` is the Hess diagram for the observed data.
 - `logAge::AbstractVector{<:Number}` is the vector containing the effective ages of the stellar populations used to create the templates in `models`, in units of `log10(age [yr])`. For example, if a population has an age of 1 Myr, its entry in `logAge` should be `log10(10^6) = 6.0`.
 - `metallicities::AbstractVector{<:Number}` is the vector containing the effective metallicities of the stellar populations used to create the templates in `models`. This is most commonly a logarithmic abundance like [M/H] or [Fe/H], but you could use a linear abundance like the metal mass fraction Z if you wanted to. There are some notes on the [Wikipedia](https://en.wikipedia.org/wiki/Metallicity) that might be useful.
 - `T_max::Number` is the time at which the age-metallicity relation has a value of `\beta` in Gyr. For example, if the oldest stellar populations in your isochrone grid are 12 Gyr old, you could set `T_max = 12.0`. 
 - `nsteps::Integer` is the number of samples to draw per chain.

# Optional Arguments (NOT YET IMPLEMENTED)
 - `nchains::Integer`: If this argument is not provided, this method will return a single chain. If this argument is provided, it will sample `nchains` chains using all available threads and will return a vector of the individual chains. If `nchains` is set, `composite` must be a vector of matrices containing a working matrix for each chain. 

# Keyword Arguments
 - `composite` is the working matrix (or vector of matrices, if the argument `nchains` is provided) that will be used to store the composite Hess diagram model during computation; must be of the same size as the templates contained in `models` and the observed Hess diagram `data`.
 - `rng::Random.AbstractRNG` is the random number generator that will be passed to DynamicHMC.jl. If `nchains` is provided this method will attempt to sample in parallel, requiring a thread-safe `rng` such as that provided by `Random.default_rng()`. 
All other keyword arguments `kws...` will be passed to `DynamicHMC.mcmc_with_warmup` or `DynamicHMC.mcmc_keep_warmup` depending on whether `nchains` is provided.

# Returns (NEEDS UPDATED)
 - If `nchains` is not provided, returns a `NamedTuple` as summarized in DynamicHMC.jl's documentation. In short, the matrix of samples can be extracted and transformed as `exp.( result.posterior_matrix )`. Statistics about the chain can be obtained with `DynamicHMC.Diagnostics.summarize_tree_statistics(result.tree_statistics)`; you want to see a fairly high acceptance rate (>0.5) and the majority of samples having termination criteria being "turning." See DynamicHMC.jl's documentation for more information.
 - If `nchains` *is* provided, returns a vector of length `nchains` of the same `NamedTuple`s described above. The samples from each chain in the returned vector can be stacked to a single `(nsamples, nchains, length(models))` matrix with `DynamicHMC.stack_posterior_matrices(result)`.
"""
function hmc_sample_mdf(models::Union{AbstractVector{<:AbstractMatrix{S}}, AbstractMatrix{S}},
                        data::Union{AbstractVector{<:Number}, AbstractMatrix{<:Number}},
                        logAge::AbstractVector{<:Number},
                        metallicities::AbstractVector{<:Number},
                        T_max::Number,
                        nsteps::Integer;
                        composite=Array{S,ndims(data)}(undef,size(data)),
                        rng::AbstractRNG=default_rng(),
                        kws...) where {S <: Number}
    @assert length(logAge) == length(metallicities)
    @assert size(data) == size(composite)
    instance = HMCModelMDF( models, composite, data, logAge, metallicities,
                            Vector{S}(undef, length(unique(logAge)) + 3), T_max )
    return DynamicHMC.mcmc_with_warmup(rng, instance, nsteps; kws...)
    # return DynamicHMC.mcmc_keep_warmup(rng, instance, nsteps; kws...)
end

include("fixed_lamr_distance_mcmc.jl")
include("fixed_linear_amr.jl")

# unique_logAge = range(6.6, 10.1; step=0.1)
# unique_MH = range(-2.2, 0.3; step=0.1)
# template_logAge = repeat(unique_logAge; inner=length(unique_MH))
# template_MH = repeat(unique_MH; outer=length(unique_logAge))
# models = [rand(99,99) for i in 1:length(template_logAge)]
# coeffs = rand(length(template_logAge))
# data = sum( coeffs .* models )
# variables = ones(length(unique_logAge)+2)
# C = similar(data)
# G = rand(length(unique_logAge)+2)
# variables[end] = -0.4 # Intercept at present-day
# variables[end-1] = -1.103700353306591e-10 # Slope in MH/yr, with yr being in terms of lookback 
# tmpans = SFH.fg2!(true, G, variables, models, data, C, template_logAge, template_MH, 0.2)
# println(sum(tmpans[1:length(unique_MH)]) ≈ 1) # This should be true if properly normalized
# This checks that the mean MH is correct for the first unique logAge
# println( isapprox(-0.4, sum( tmpans[1:length(unique_MH)] .* unique_MH ) / sum( tmpans[1:length(unique_MH)] ), atol=1e-3 ) )
# import ForwardDiff
# ForwardDiff.gradient( x-> SFH.fg2!(true, G, x, models, data, C, template_logAge, template_MH, 0.2), variables)
# FiniteDifferences is very slow but agrees with ForwardDiff
# import FiniteDifferences
# FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), x-> SFH.fg2!(true, rand(length(variables)), x, models, data, C, template_logAge, template_MH, 0.2), variables)
# G2 = similar(coeffs)
# @benchmark SFH.fg!($true, $G2, $coeffs, $models, $data, $C) # 7.6 ms
# @benchmark SFH.fg2!($true, $G, $variables, $models, $data, $C, $template_logAge, $template_MH, $0.2) # currently 8 ms
