# Linear age-metallicity relation with constant Gaussian spread σ

"""
    x0::Vector = construct_x0_mdf(logAge::AbstractVector{T}, max_logAge::Number; normalize_value::Number=one(T)) where T <: Number

Generates a vector of initial stellar mass normalizations for input to [`StarFormationHistories.fit_templates_mdf`](@ref) or [`StarFormationHistories.hmc_sample_mdf`](@ref) with a total stellar mass of `normalize_value` such that the implied star formation rate is constant across the provided `logAge` vector that contains the `log10(Age [yr])` of each isochrone that you are going to input as models. For the purposes of computing the constant star formation rate, the provided `logAge` are treated as left-bin edges, and with the final right-bin edge being `max_logAge`. For example, you might have `logAge=[6.6, 6.7, 6.8]` in which case you would want to set `max_logAge=6.9` so that the width of the final bin for the star formation rate calculation has the same `log10(Age [yr])` step as the other bins.

The difference between this function and [`StarFormationHistories.construct_x0`](@ref) is that this function generates an `x0` vector that is of length `length(unique(logage))` (that is, a single normalization factor for each unique entry in `logAge`) while [`StarFormationHistories.construct_x0`](@ref) returns an `x0` vector that is of length `length(logAge)`; that is, a normalization factor for every entry in `logAge`. The order of the coefficients is such that the coefficient `x[i]` corresponds to the entry `unique(logAge)[i]`. 

# Notes

# Examples
```julia
julia> construct_x0_mdf([9.0,8.0,7.0], 10.0; normalize_value=5.0)
3-element Vector{Float64}:
 4.504504504504504
 0.4504504504504504
 0.04504504504504504

julia> construct_x0_mdf(repeat([9.0,8.0,7.0,8.0];inner=3), 10.0; normalize_value=5.0)
3-element Vector{Float64}:
 4.504504504504504
 0.4504504504504504
 0.04504504504504504

julia> construct_x0_mdf(repeat([9.0,8.0,7.0,8.0],3), 10.0; normalize_value=5.0) ≈ construct_x0([9.0,8.0,7.0], 10.0; normalize_value=5.0)
true
```
"""
function construct_x0_mdf(logAge::AbstractVector{T}, max_logAge::Number; normalize_value::Number=one(T)) where T <: Number
    minlog, maxlog = extrema(logAge)
    @assert max_logAge > maxlog # max_logAge has to be greater than the maximum of logAge vector
    sfr = normalize_value / (exp10(max_logAge) - exp10(minlog)) # Average SFR / yr
    unique_logAge = unique(logAge)
    idxs = sortperm(unique_logAge)
    sorted_ul = vcat(unique_logAge[idxs], max_logAge)
    dt = diff( vcat(exp10.(sorted_ul), exp10(max_logAge)) )
    return [ begin
                idx = findfirst( ==(sorted_ul[i]), unique_logAge )
                sfr * dt[idx]
             end for i in eachindex(unique_logAge) ]
end

"""
    LogTransformMDFσResult(μ::AbstractVector{<:Number},
                           σ::AbstractVector{<:Number},
                           invH::AbstractMatrix{<:Number},
                           result)

Type for containing the maximum likelihood estimate (MLE) and maximum a posteriori (MAP) results from [`fit_templates_mdf`](@ref) for fixed `σ`. The fitted coefficients are available in the `μ` field. Estimates of the standard errors are available in the `σ` field. These have both been transformed from the native logarithmic fitting space into natural units (i.e., stellar mass or star formation rate).

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
    for i in axes(x,1)[begin:end-2]
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
    coeffs = calculate_coeffs_mdf(variables::AbstractVector{<:Number}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number} [, α::Number, β::Number, σ::Number])

Calculates per-model stellar mass coefficients `coeffs` from the fitting parameters of [`StarFormationHistories.fit_templates_mdf`](@ref) and [`StarFormationHistories.hmc_sample_mdf`](@ref). The `variables` returned by these functions is of length `length(unique(logAge))+3`. The first `length(logAge)` entries are stellar mass coefficients, one per unique entry in `logAge`. The final three elements are α, β, and σ defining a metallicity evolution such that the mean for element `i` of `unique(logAge)` is `μ[i] = α * exp10(unique(logAge)[i]) / 1e9 + β`. The individual weights per each isochrone are then determined via Gaussian weighting with the above mean and the provided `σ`. 
"""
function calculate_coeffs_mdf(variables::AbstractVector{<:Number}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}, α::Number, β::Number, σ::Number)
    S = promote_type(eltype(variables), eltype(logAge), eltype(metallicities), typeof(α), typeof(β), typeof(σ))
    # Compute the coefficients on each model template given the `variables` and the MDF
    unique_logAge = unique(logAge)
    @assert length(variables) == length(unique_logAge)
    coeffs = Vector{S}(undef,length(logAge))
    norm_vals = Vector{S}(undef,length(unique_logAge))
    for i in eachindex(unique_logAge)
        la = unique_logAge[i]
        μ = α * exp10(la) / 1e9 + β # Find the mean metallicity of this age bin
        idxs = findall( ==(la), logAge) # Find all entries that match this logAge
        tmp_coeffs = [_gausspdf(metallicities[j], μ, σ) for j in idxs] # Calculate relative weights
        A = sum(tmp_coeffs)
        norm_vals[i] = A
        # Make sure sum over tmp_coeffs equals 1 and write to coeffs
        coeffs[idxs] .= tmp_coeffs .* variables[i] ./ A
    end
    return coeffs
end
calculate_coeffs_mdf(variables, logAge, metallicities) =
    calculate_coeffs_mdf(view(variables,firstindex(variables):lastindex(variables)-3),
                         logAge, metallicities, variables[end-2], variables[end-1], variables[end])


"""
variables[begin:end-2] are stellar mass coefficients
variables[end-1] is the slope of the age-MH relation in [MH] / [10Gyr; (lookback)], e.g. -1.0
variables[end] is the intercept of the age-MH relation in MH at present-day, e.g. -0.4
"""
@inline function fg_mdf_fixedσ!(F, G, variables::AbstractVector{<:Number}, models::AbstractVector{T}, data::AbstractMatrix{<:Number}, composite::AbstractMatrix{<:Number}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}, σ::Number) where T <: AbstractMatrix{<:Number}
    # `variables` should have length `length(unique(logAge)) + 2`; coeffs for each unique
    # entry in logAge, plus α and β to define the MDF at fixed logAge
    @assert axes(data) == axes(composite)
    S = promote_type(eltype(variables), eltype(eltype(models)), eltype(eltype(data)), eltype(composite), eltype(logAge), eltype(metallicities))
    # Compute the coefficients on each model template given the `variables` and the MDF
    α, β = variables[end-1], variables[end]
    unique_logAge = unique(logAge)
    coeffs = calculate_coeffs_mdf(view(variables,firstindex(variables):lastindex(variables)-2), logAge, metallicities, α, β, σ)
    # Fill the composite array with the equivalent of sum( coeffs .* models )
    # composite = sum( coeffs .* models )
    # return -loglikelihood(composite, data)
    composite!(composite, coeffs, models)
    logL = loglikelihood(composite, data)
    if (F != nothing) & (G != nothing) # Optim.optimize wants objective and gradient
        @assert axes(G) == axes(variables)
        # Calculate the ∇loglikelihood with respect to model coefficients; we will need all of these
        # fullG = [ ∇loglikelihood(models[i], composite, data) for i in axes(models,1) ]
        fullG = Vector{eltype(G)}(undef,length(models))
        ∇loglikelihood!(fullG, composite, models, data)
        # Now need to do the transformation to the `variables` rather than model coefficients
        G[end-1] = zero(eltype(G))
        G[end] = zero(eltype(G))
        for i in axes(G,1)[begin:end-2] 
            la = unique_logAge[i]
            age = exp10(la) / 1e9 # the / 1e9 makes α the slope in MH/Gyr, improves convergence
            μ = α * age + β # Find the mean metallicity of this age bin
            idxs = findall( ==(la), logAge) # Find all entries that match this logAge
            tmp_coeffs = [_gausspdf(metallicities[j], μ, σ) for j in idxs] # Calculate relative weights
            A = sum(tmp_coeffs)
            # This should be correct for any MDF model at fixed logAge
            @inbounds G[i] = -sum( fullG[j] * coeffs[j] / variables[i] for j in idxs )
            βsum = sum( ((metallicities[idxs[j]]-μ) * tmp_coeffs[j]) for j in eachindex(idxs))
            dLdβ = -sum( fullG[idxs[j]] * variables[i] *
                ( ((metallicities[idxs[j]]-μ) * tmp_coeffs[j]) - tmp_coeffs[j] / A * βsum )
                         for j in eachindex(idxs)) / A / σ^2
            dLdα = dLdβ * age
            G[end-1] += dLdα
            G[end] += dLdβ
        end
        return -logL
    elseif F != nothing # Optim.optimize wants only objective
        return -logL
    end
end

function fit_templates_mdf(models::AbstractVector{T},
                           data::AbstractMatrix{<:Number},
                           logAge::AbstractVector{<:Number},
                           metallicities::AbstractVector{<:Number},
                           σ::Number;
                           composite=Matrix{S}(undef,size(data)),
                           x0=vcat(construct_x0_mdf(logAge), [-0.1, -0.5]),
                           kws...) where {S <: Number, T <: AbstractMatrix{S}}
    unique_logage = unique(logAge)
    @assert length(x0) == length(unique_logage)+2
    # Perform logarithmic transformation on the provided x0 for all variables except α and β
    for i in eachindex(x0)[begin:end-2]
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
        for i in eachindex(xvec)[begin:end-2] # These are the per-logage stellar mass coefficients
            x[i] = exp(xvec[i])
        end
        x[end-1] = xvec[end-1]  # α
        x[end] = xvec[end]      # β

        logL = fg_mdf_fixedσ!(F, G, x, models, data, composite, logAge, metallicities, σ)
        logL -= sum( @view xvec[begin:end-2] ) # this is the Jacobian correction
        # Add the Jacobian correction for every element of G except α (x[end-2]) and β (x[end-1])
        for i in eachindex(G)[begin:end-2]
            G[i] = G[i] * x[i] - 1
        end
        return logL
    end
    function fg_mdf_fixedσ!_mle(F, G, xvec)
        for i in eachindex(xvec)[begin:end-2] # These are the per-logage stellar mass coefficients
            x[i] = exp(xvec[i])
        end
        x[end-1] = xvec[end-1]  # α
        x[end] = xvec[end]      # β

        logL = fg_mdf_fixedσ!(F, G, x, models, data, composite, logAge, metallicities, σ)
        for i in eachindex(G)[begin:end-2]
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
    for i in eachindex(μ_map,μ_mle)[begin:end-2]
        μ_map[i] = exp(μ_map[i])
        μ_mle[i] = exp(μ_mle[i])
    end
    # Estimate parameter uncertainties from the inverse Hessian approximation
    σ_map = sqrt.(diag(Optim.trace(result_map)[end].metadata["~inv(H)"]))
    σ_mle = sqrt.(diag(Optim.trace(result_mle)[end].metadata["~inv(H)"]))
    # Need to account for the logarithmic transformation
    for i in eachindex(σ_map,σ_mle)[begin:end-2]
        σ_map[i] = μ_map[i] * σ_map[i]
        σ_mle[i] = μ_mle[i] * σ_mle[i]
    end
    return (map = LogTransformMDFσResult(μ_map, σ_map, Optim.trace(result_map)[end].metadata["~inv(H)"], result_map),
            mle = LogTransformMDFσResult(μ_mle, σ_mle, Optim.trace(result_mle)[end].metadata["~inv(H)"], result_mle))
    # return (map = (μ = μ_map, σ = σ_map, result = result_map),
    #         mle = (μ = μ_mle, σ = σ_mle, result = result_mle))
end

"""
    LogTransformMDFResult(μ::AbstractVector{<:Number},
                           σ::AbstractVector{<:Number},
                           invH::AbstractMatrix{<:Number},
                           result)

Type for containing the maximum likelihood estimate (MLE) and maximum a posteriori (MAP) results from [`fit_templates_mdf`](@ref) when freely fitting `σ`. The fitted coefficients are available in the `μ` field. Estimates of the standard errors are available in the `σ` field. These have both been transformed from the native logarithmic fitting space into natural units (i.e., stellar mass or star formation rate).

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
    for i in axes(x,1)[begin:end-3]
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

"""
This version of mdf fg! also fits σ
"""
@inline function fg_mdf!(F, G, variables::AbstractVector{<:Number}, models::AbstractVector{T}, data::AbstractMatrix{<:Number}, composite::AbstractMatrix{<:Number}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}) where T <: AbstractMatrix{<:Number}
    # `variables` should have length `length(unique(logAge)) + 2`; coeffs for each unique
    # entry in logAge, plus α and β to define the MDF at fixed logAge
    @assert axes(data) == axes(composite)
    S = promote_type(eltype(variables), eltype(eltype(models)), eltype(eltype(data)), eltype(composite), eltype(logAge), eltype(metallicities))
    # Compute the coefficients on each model template given the `variables` and the MDF
    α, β, σ = variables[end-2], variables[end-1], variables[end]
    unique_logAge = unique(logAge)
    # Calculate the per-template coefficents and normalization values
    coeffs = calculate_coeffs_mdf(view(variables,firstindex(variables):lastindex(variables)-3), logAge, metallicities, α, β, σ)

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
        for i in axes(G,1)[begin:end-3] 
            la = unique_logAge[i]
            age = exp10(la) / 1e9 # the / 1e9 makes α the slope in MH/Gyr, improves convergence
            μ = α * age + β # Find the mean metallicity of this age bin
            idxs = findall( ==(la), logAge) # Find all entries that match this logAge
            tmp_coeffs = [_gausspdf(metallicities[j], μ, σ) for j in idxs] # Calculate relative weights
            A = sum(tmp_coeffs)
            # This should be correct for any MDF model at fixed logAge
            @inbounds G[i] = -sum( fullG[j] * coeffs[j] / variables[i] for j in idxs )
            βsum = sum( ((metallicities[idxs[j]]-μ) * tmp_coeffs[j]) for j in eachindex(idxs))
            dLdβ = -sum( fullG[idxs[j]] * variables[i] *
                ( ((metallicities[idxs[j]]-μ) * tmp_coeffs[j]) - tmp_coeffs[j] / A * βsum )
                         for j in eachindex(idxs)) / A / σ^2
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

"""
    fit_templates_mdf(models::AbstractVector{T},
                      data::AbstractMatrix{<:Number},
                      logAge::AbstractVector{<:Number},
                      metallicities::AbstractVector{<:Number} [, σ::Number];
                      composite=Matrix{S}(undef,size(data)),
                      x0=vcat(construct_x0_mdf(logAge), [-0.1, -0.5, 0.3]),
                      kws...) where {S <: Number, T <: AbstractMatrix{S}}

Method that fits a linear combination of the provided Hess diagrams `models` to the observed Hess diagram `data`, constrained to have a linear mean metallicity evolution with the mean metallicity of element `i` of `unique(logAge)` being `μ[i] = α * exp10(unique(logAge)[i]) / 1e9 + β`. `α` is therefore a slope in the units of `metallicities` per Gyr, and `β` is the mean metallicity value of stars being born at present-day. Individual weights per each isochrone are then determined via Gaussian weighting with the above mean and the standard deviation `σ`, which can either be fixed or fit.

This function is designed to work best with a "grid" of stellar models, defined by the outer product of `N` unique entries in `logAge` and `M` unique entries in `metallicities`. See the examples for more information on usage. 

# Arguments
 - `models::AbstractVector{<:AbstractMatrix{<:Number}}` is a vector of equal-sized matrices that represent the template Hess diagrams for the simple stellar populations that compose the observed Hess diagram.
 - `data::AbstractMatrix{<:Number}` is the Hess diagram for the observed data.
 - `logAge::AbstractVector{<:Number}` is the vector containing the effective ages of the stellar populations used to create the templates in `models`, in units of `log10(age [yr])`. For example, if a population has an age of 1 Myr, its entry in `logAge` should be `log10(10^6) = 6.0`.
 - `metallicities::AbstractVector{<:Number}` is the vector containing the effective metallicities of the stellar populations used to create the templates in `models`. This is most commonly a logarithmic abundance like [M/H] or [Fe/H], but you could use a linear abundance like the metal mass fraction Z if you wanted to. There are some notes on the [Wikipedia](https://en.wikipedia.org/wiki/Metallicity) that might be useful. 

# Optional Arguments
 - If provided, `σ::Number` is the fixed width of the Gaussian the defines the metallicity distribution function (MDF) at fixed `logAge`. If this argument is omitted, `σ` will be a free parameter in the fit. 

# Keyword Arguments
 - `composite` is the working matrix that will be used to store the composite Hess diagram model during computation; must be of the same size as the templates contained in `models` and the observed Hess diagram `data`.
 - `x0` is the vector of initial guesses for the stellar mass coefficients per *unique* entry in `logAge`, plus the variables that define the metallicity evolution model. You should basically always be calculating and passing this keyword argument. We provide [`StarFormationHistories.construct_x0_mdf`](@ref) to prepare the first part of `x0` assuming constant star formation rate, which is typically a good initial guess. You then have to concatenate that result with an initial guess for the metallicity evolution parameters. For example, `x0=vcat(construct_x0_mdf(logAge; normalize_value=1e4), [-0.1,-0.5,0.3])`, where `logAge` is a valid argument for this function (see above), and the initial guesses on the parameters are `[α, β, σ] = [-0.1, -0.5, 0.3]`. If the provided `metallicities` are, for example, [M/H] values, then this mean metallicity evolution is μ(t) [dex] = -0.1 [dex/Gyr] * t [Gyr] - 0.5 [dex], and at fixed time, the metallicity distribution function is Gaussian with mean μ(t) and standard deviation σ. If you provide `σ` as an optional argument, then you should not include an entry for it in `x0`. 
Other `kws...` are passed to `Optim.options` to set things like convergence criteria for the optimization.

# Returns
 - This function returns an object (say, `result`) of similar structure to the object returned by [`fit_templates`](@ref). Specifically, this method will return a `NamedTuple` with entries `result.mle` and `result.map` for the maximum likelihood and maximum a posteriori estimates, respectively. If you provide a fixed `σ`, those objects will be instances of [`StarFormationHistories.LogTransformMDFσResult`](@ref). If you allow `σ` to be freely fit, those objects will be instances of [`StarFormationHistories.LogTransformMDFResult`](@ref). Both of these types support sampling via, e.g., `rand(result.map, 10)`. 

# Notes
 - `α` and `β` are not optimized under a logarithmic transform, but `σ` is since it must be positive. This method also uses the `BFGS` method from `Optim.jl` internally just like [`fit_templates`](@ref); please see the notes section of that method. 
"""
function fit_templates_mdf(models::AbstractVector{T},
                           data::AbstractMatrix{<:Number},
                           logAge::AbstractVector{<:Number},
                           metallicities::AbstractVector{<:Number};
                           composite=Matrix{S}(undef,size(data)),
                           x0=vcat(construct_x0_mdf(logAge), [-0.1, -0.5, 0.3]),
                           kws...) where {S <: Number, T <: AbstractMatrix{S}}
    unique_logage = unique(logAge)
    @assert length(x0) == length(unique_logage)+3
    # Perform logarithmic transformation on the provided x0 for all variables except α and β
    for i in eachindex(x0)[begin:end-3]
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
        for i in eachindex(xvec)[begin:end-3] # These are the per-logage stellar mass coefficients
            x[i] = exp(xvec[i])
        end
        x[end-2] = xvec[end-2]  # α
        x[end-1] = xvec[end-1]  # β
        x[end] = exp(xvec[end]) # σ
        logL = fg_mdf!(F, G, x, models, data, composite, logAge, metallicities)
        logL -= sum( @view xvec[begin:end-3] ) + xvec[end] # This is the Jacobian correction
        # Add the Jacobian correction for every element of G except α (x[end-2]) and β (x[end-1])
        for i in eachindex(G)[begin:end-3]
            G[i] = G[i] * x[i] - 1
        end
        G[end] = G[end] * x[end] - 1
        return logL
    end
    function fg_mdf!_mle(F, G, xvec)
        for i in eachindex(xvec)[begin:end-3] # These are the per-logage stellar mass coefficients
            x[i] = exp(xvec[i])
        end
        x[end-2] = xvec[end-2]  # α
        x[end-1] = xvec[end-1]  # β
        x[end] = exp(xvec[end]) # σ
        logL = fg_mdf!(F, G, x, models, data, composite, logAge, metallicities)
        for i in eachindex(G)[begin:end-3]
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
    for i in eachindex(μ_map,μ_mle)[begin:end-3]
        μ_map[i] = exp(μ_map[i])
        μ_mle[i] = exp(μ_mle[i])
    end
    μ_map[end] = exp(μ_map[end])
    μ_mle[end] = exp(μ_mle[end])

    # Estimate parameter uncertainties from the inverse Hessian approximation
    σ_map = sqrt.(diag(Optim.trace(result_map)[end].metadata["~inv(H)"]))
    σ_mle = sqrt.(diag(Optim.trace(result_mle)[end].metadata["~inv(H)"]))
    # Need to account for the logarithmic transformation
    for i in eachindex(σ_map,σ_mle)[begin:end-3]
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

# We can even use the inv(H) = covariance matrix estimate to draw samples to compare to HMC
# import Distributions: MvNormal
# result1, std1, fr = fit_templates_mdf(mdf_templates, h.weights, mdf_template_logAge, mdf_template_MH; x0=vcat(construct_x0_mdf(mdf_template_logAge; normalize_value=1e4),[-0.1,-0.5,0.3]))
# corner.corner(permutedims(rand(MvNormal(result1,LinearAlgebra.Hermitian(fr.trace[end].metadata["~inv(H)"])),10000)[end-2:end,:]))
# Can we also use this inv(H) estimate as input to HMC? I think that's roughly the M matrix.

######################################################################################
## HMC with MDF

struct HMCModelMDF{T,S,V,W,X,Y}
    models::T
    composite::S
    data::V
    logAge::W
    metallicities::X
    G::Y
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
    @assert axes(G) == axes(xvec)
    # Transform the provided x
    # α and β, which are xvec[end-2] and xvec[end-1], are the only variables that are not log-transformed
    x = similar(xvec)
    for i in eachindex(xvec)[begin:end-3] # These are the per-logage stellar mass coefficients
        x[i] = exp(xvec[i])
    end
    x[end-2] = xvec[end-2]  # α
    x[end-1] = xvec[end-1]  # β
    x[end] = exp(xvec[end]) # σ

    # fg_mdf! returns -logL and fills G with -∇logL so we need to negate the signs.
    logL = -fg_mdf!(true, G, x, models, data, composite, logAge, metallicities)
    logL += sum(view(xvec,firstindex(xvec):lastindex(xvec)-3)) + xvec[end] # this is the Jacobian correction
    ∇logL = -G
    # Add the Jacobian correction for every element of ∇logL except α (x[end-2]) and β (x[end-1])
    for i in eachindex(∇logL)[begin:end-3]
        ∇logL[i] = ∇logL[i] * x[i] + 1
    end
    ∇logL[end] = ∇logL[end] * x[end] + 1
    return logL, ∇logL
end

# Version with just one chain; no threading
function hmc_sample_mdf(models::AbstractVector{T}, data::AbstractMatrix{<:Number}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}, nsteps::Integer; composite=Matrix{S}(undef,size(data)), rng::AbstractRNG=default_rng(), kws...) where {S <: Number, T <: AbstractMatrix{S}}
    @assert length(logAge) == length(metallicities)
    instance = HMCModelMDF( models, composite, data, logAge, metallicities,
                            Vector{S}(undef, length(unique(logAge)) + 3) )
    return DynamicHMC.mcmc_with_warmup(rng, instance, nsteps; kws...)
end

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
