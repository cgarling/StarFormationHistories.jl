"""
    calculate_coeffs_logamr(variables::AbstractVector{<:Number},
                            logAge::AbstractVector{<:Number},
                            metallicities::AbstractVector{<:Number}
                            [, α::Number, β::Number, σ::Number];
                            MH_func = StarFormationHistories.MH_from_Z,
                            max_logAge = maximum(logAge))

Calculates per-model stellar mass coefficients `coeffs` from the fitting parameters of [`StarFormationHistories.fit_templates_logamr`](@ref) and [`StarFormationHistories.hmc_sample_logamr`](@ref). The `variables` returned by these functions is of length `length(unique(logAge))+3`. The first `length(logAge)` entries are stellar mass coefficients, one per unique entry in `logAge`. The final three elements are α, β, and σ defining a metallicity evolution such that the mean metal mass fraction Z for element `i` of `unique(logAge)` is `μ_Z[i] = α * (exp10(max_logAge) - exp10(unique(logAge)[i])) / 1e9 + β`. This is converted to a mean metallicity in [M/H] via the provided callable keyword argument `MH_func` which defaults to [`StarFormationHistories.MH_from_Z`](@ref). The individual weights per each isochrone are then determined via Gaussian weighting with the above mean [M/H] and the provided `σ` in dex. The provided `metallicities` vector should be in [M/H]. 

# Notes
 - Physically, the metal mass fraction `Z` must always be positive. Under the above model, this means α and β must be greater than or equal to 0. With σ being a Gaussian width, it must be positive.
 - If you manually set the keyword argument `max_logAge` to something lower than the maximum of the `logAge` argument you provide, a warning will be raised which may be ignored if it does not result in any of the mean metal mass fractions Z being less than 0 for any of the provided `logAge`.
 - An error will be thrown if the provided age-metallicity relation variables (α, β) and `max_logAge` keyword argument result in a mean metal mass fraction less than 0 for any time in the provided `logAge` vector. 
"""
function calculate_coeffs_logamr(variables::AbstractVector{<:Number}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}, α::Number, β::Number, σ::Number; MH_func=MH_from_Z, max_logAge=maximum(logAge))
    @assert (α >= 0) & (β >= 0) & (σ > 0)
    if maximum(logAge) > max_logAge
        @warn "We recommend that the keyword argument `max_logAge` to `StarFormationHistories.calculate_coeffs_logamr` be set equal to or greater than the maximum of the `logAge` argument. The provided `max_logAge` is less than `maximum(logAge)`, such that it is possible the metal mass fraction may become negative in the model, which would be unphysical."
    end
    S = promote_type(eltype(variables), eltype(logAge), eltype(metallicities), typeof(α), typeof(β), typeof(σ))
    # Compute the coefficients on each model template given the `variables` and the MDF
    unique_logAge = unique(logAge)
    max_age = exp10(max_logAge) # Lookback time in yr at which Z = β
    @assert length(variables) == length(unique_logAge)
    coeffs = Vector{S}(undef,length(logAge))
    norm_vals = Vector{S}(undef,length(unique_logAge))
    for i in eachindex(unique_logAge)
        la = unique_logAge[i]
        μZ = α * (max_age - exp10(la)) / 1e9 + β # Find the mean Z of this age bin
        # Test that the mean Z is greater than 0
        if μZ < 0 # Keep this for now
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
@inline function fg_logamr!(F, G, variables::AbstractVector{<:Number}, models::Union{AbstractMatrix{<:Number}, AbstractVector{<:AbstractMatrix{<:Number}}}, data::Union{AbstractVector{<:Number}, AbstractMatrix{<:Number}}, composite::Union{AbstractVector{<:Number}, AbstractMatrix{<:Number}}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}, max_age::Number, MH_func, MH_deriv_Z)
    # `variables` should have length `length(unique(logAge)) + 3`; coeffs for each unique
    # entry in logAge, plus α and β to define the MDF at fixed logAge and σ to define Gaussian width
    @assert axes(data) == axes(composite)
    S = promote_type(eltype(variables), eltype(eltype(models)), eltype(eltype(data)), eltype(composite), eltype(logAge), eltype(metallicities), typeof(max_age))
    # Compute the coefficients on each model template given the `variables` and the MDF
    α, β, σ = variables[end-2], variables[end-1], variables[end]
    unique_logAge = unique(logAge)
    max_logAge = log10(max_age) + 9
    # Calculate the per-template coefficents and normalization values
    coeffs = calculate_coeffs_logamr(view(variables,firstindex(variables):lastindex(variables)-3), logAge, metallicities, α, β, σ; MH_func=MH_func, max_logAge=max_logAge)

    # Fill the composite array with the equivalent of sum( coeffs .* models )
    # composite = sum( coeffs .* models )
    # return -loglikelihood(composite, data)
    composite!(composite, coeffs, models)
    logL = loglikelihood(composite, data) # Need to do this before ∇loglikelihood! because it will overwrite composite
    if (F != nothing) & (G != nothing) # Optim.optimize wants objective and gradient
        @assert axes(G) == axes(variables)
        # Calculate the ∇loglikelihood with respect to model coefficients; we will need all of these
        fullG = Vector{eltype(G)}(undef,length(coeffs)) # length(models))
        ∇loglikelihood!(fullG, composite, models, data)
        # Now need to do the transformation to the `variables` rather than model coefficients
        G[end-2] = zero(eltype(G))
        G[end-1] = zero(eltype(G))
        G[end] = zero(eltype(G))
        dZdβ = 1 # Derivative of metal mass fraction with respect to β
        for i in axes(G,1)[begin:end-3] 
            la = unique_logAge[i]
            age = exp10(la) / 1e9 # the / 1e9 makes α the slope in MH/Gyr, improves convergence
            # meanZ = α * age + β # Mean metal mass fraction
            meanZ = α * (max_age - age) + β # Mean metal mass fraction
            μ = MH_func(meanZ) # Find the mean metallicity [M/H] of this age bin
            idxs = findall( ==(la), logAge) # Find all entries that match this logAge
            tmp_coeffs = [_gausspdf(metallicities[j], μ, σ) for j in idxs] # Calculate relative weights
            A = sum(tmp_coeffs)
            # This should be correct for any MDF model at fixed logAge
            @inbounds G[i] = -sum( fullG[j] * coeffs[j] / variables[i] for j in idxs )
            βsum = sum( ((metallicities[idxs[j]]-μ) * tmp_coeffs[j]) for j in eachindex(idxs))
            dAdμ = -sum( fullG[idxs[j]] * variables[i] *
                ( ((metallicities[idxs[j]]-μ) * tmp_coeffs[j]) - tmp_coeffs[j] / A * βsum )
                         for j in eachindex(idxs)) / A / σ^2
            dμdZ = MH_deriv_Z(meanZ) # Derivative of MH_func(Z) with respect to Z
            dZdα = (max_age - age) # Derivative of metal mass fraction with respect to α
            dLdβ = dAdμ * dμdZ * dZdβ
            dLdα = dLdβ * dZdα
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
    result = fit_templates_logamr(models::AbstractVector{<:AbstractMatrix{S}},
                                  data::AbstractMatrix{<:Number},
                                  logAge::AbstractVector{<:Number},
                                  metallicities::AbstractVector{<:Number} [, σ::Number];
                                  x0 = vcat(construct_x0_mdf(logAge, convert(S,log10(13.7e9))),
                                            [1e-4, 5e-5, 0.2]),
                                  MH_func = StarFormationHistories.MH_from_Z,
                                  MH_deriv_Z = StarFormationHistories.dMH_dZ,
                                  max_logAge = maximum(logAge),
                                  kws...) where {S <: Number}
    result = fit_templates_logamr(models::AbstractMatrix{S},
                                  data::AbstractVector{<:Number},
                                  logAge::AbstractVector{<:Number},
                                  metallicities::AbstractVector{<:Number} [, σ::Number];
                                  x0 = vcat(construct_x0_mdf(logAge, convert(S,log10(13.7e9))),
                                            [1e-4, 5e-5, 0.2]),
                                  MH_func = StarFormationHistories.MH_from_Z,
                                  MH_deriv_Z = StarFormationHistories.dMH_dZ,
                                  max_logAge = maximum(logAge),
                                  kws...) where {S <: Number}

Method that fits a linear combination of the provided Hess diagrams `models` to the observed Hess diagram `data`, constrained to have a logarithmic age-metallicity relation with the mean metal mass fraction `μ_Z` of element `i` of `unique(logAge)` being `μ_Z[i] = α * (exp10(max_logAge) - exp10(unique(logAge)[i])) / 1e9 + β`. This is converted to a mean metallicity in [M/H] via the provided callable keyword argument `MH_func` which defaults to [`MH_from_Z`](@ref StarFormationHistories.MH_from_Z). `α` is therefore a slope in the units of inverse Gyr, and `β` is the mean metal mass fraction of stars born at the earliest valid lookback time, determined by keyword argument `max_logAge`. Individual weights for each isochrone template are then determined via Gaussian weighting with the above mean [M/H] and the standard deviation `σ` in dex, which can either be fixed or fit.

This function is designed to work best with a "grid" of stellar models, defined by the outer product of `N` unique entries in `logAge` and `M` unique entries in `metallicities`. See the examples for more information on usage.

The second call signature supports the flattened formats for `models` and `data`. See the notes for the flattened call signature of [`StarFormationHistories.composite!`](@ref) for more details.

# Arguments
 - `models` are the template Hess diagrams for the simple stellar populations that compose the observed Hess diagram. 
 - `data` is the Hess diagram for the observed data. 
 - `logAge::AbstractVector{<:Number}` is the vector containing the effective ages of the stellar populations used to create the templates in `models`, in units of `log10(age [yr])`. For example, if a population has an age of 1 Myr, its entry in `logAge` should be `log10(10^6) = 6.0`.
 - `metallicities::AbstractVector{<:Number}` is the vector containing the effective metallicities of the stellar populations used to create the templates in `models`. This should be a logarithmic abundance like [M/H] or [Fe/H].

# Optional Arguments
 - If provided, `σ::Number` is the fixed width of the Gaussian the defines the metallicity distribution function (MDF) at fixed `logAge`. If this argument is omitted, `σ` will be a free parameter in the fit. 

# Keyword Arguments
 - `x0` is the vector of initial guesses for the stellar mass coefficients per *unique* entry in `logAge`, plus the variables that define the metallicity evolution model. You should basically always be calculating and passing this keyword argument. We provide [`construct_x0_mdf`](@ref StarFormationHistories.construct_x0_mdf) to prepare the first part of `x0` assuming constant star formation rate, which is typically a good initial guess. You then have to concatenate that result with an initial guess for the metallicity evolution parameters. For example, `x0=vcat(construct_x0_mdf(logAge, 10.13; normalize_value=1e4), [1e-4, 5e-5, 0.2])`, where `logAge` is a valid argument for this function (see above), and the initial guesses on the parameters are `[α, β, σ] = [1e-4, 5e-5, 0.2]`. If you provide `σ` as an optional argument, then you should not include an entry for it in `x0`.
 - `MH_func` is a callable that takes a metal mass fraction `Z` and returns the logarithmic abundance [M/H]; by default uses [`MH_from_Z`](@ref StarFormationHistories.MH_from_Z).
 - `MH_deriv_Z` is a callable that takes a metal mass fraction `Zj` and returns the derivative of `MH_func` with respect to the metal mass fraction `Z` evaluated at `Zj`. For the default value of `MH_func`, [`dMH_dZ`](@ref StarFormationHistories.dMH_dZ) provides the correct derivative. You only need to change this if you use an alternate `MH_func`.
 - `max_logAge` is the maximum `log10(age [Gyr])` for which the age-metallicity relation is to be valid. In most cases this should just be the maximum of the `logAge` vector argument: `maximum(logAge)`. By definition, this is the `logAge` at which `μ_Z = β`. 
 - Other `kws...` are passed to `Optim.options` to set things like convergence criteria for the optimization.
"""
function fit_templates_logamr(models::AbstractMatrix{S},
                              data::AbstractVector{<:Number},
                              logAge::AbstractVector{<:Number},
                              metallicities::AbstractVector{<:Number};
                              x0 = vcat(construct_x0_mdf(logAge, convert(S,log10(13.7e9))),
                                        [1e-4, 5e-5, 0.2]),
                              MH_func = MH_from_Z,
                              MH_deriv_Z = dMH_dZ,
                              max_logAge = maximum(logAge),
                              kws...) where {S <: Number}
    composite = Vector{S}(undef,length(data)) # Scratch matrix for storing complex Hess model
    unique_logage = unique(logAge)
    max_age = exp10(max_logAge) / 1e9 # Lookback time at which to normalize β in Gyr
    @assert length(x0) == length(unique_logage) + 3
    @assert size(models,1) == length(data)
    @assert size(models,2) == length(logAge) == length(metallicities)
    # All variables must be positive  since Z = α * (max_age - age) + β with age being
    # positive lookback time, α must always be positive for an increasing AMR.
    # Perform logarithmic transformation on the provided x0 for all variables.
    x0 = log.(x0)
    # Make scratch array for assessing transformations
    x = similar(x0)
    # Define wrapper function to pass to Optim.only_fg!
    # It looks like you don't need the Jacobian correction to arrive at the maximum likelihood
    # result, and if you remove the Jacobian corrections it actually converges to the
    # non-log-transformed case. However, the uncertainty estimates from the inverse Hessian
    # don't seem reliable without the Jacobian corrections.
    function fg_logamr!_map(F, G, xvec)
        # All variables are log-transformed
        x .= exp.(xvec) 
        logL = fg_logamr!(F, G, x, models, data, composite,
                          logAge, metallicities, max_age, MH_func, MH_deriv_Z)
        # Appy Jacobian correction to logL; α should be the same as
        # D[Log[x],x] = D[Log[-x],x] = 1/x, Log[1/x] = -Log[x]
        logL -= sum( xvec ) 
        # Add the Jacobian correction for every element of G
        G .= G .* x .- 1
        return logL
    end
    function fg_logamr!_mle(F, G, xvec)
        # All variables are log-transformed
        x .= exp.(xvec) 
        logL = fg_logamr!(F, G, x, models, data, composite,
                          logAge, metallicities, max_age, MH_func, MH_deriv_Z)
        G .= G .* x 
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
    result_map = Optim.optimize(Optim.only_fg!( fg_logamr!_map ), x0, bfgs_struct, bfgs_options)
    result_mle = Optim.optimize(Optim.only_fg!( fg_logamr!_mle ), Optim.minimizer(result_map), bfgs_struct, bfgs_options)
    # Transform the resulting variables
    μ_map = exp.(deepcopy( Optim.minimizer(result_map) ))
    μ_mle = exp.(deepcopy( Optim.minimizer(result_mle) ))

    # Estimate parameter uncertainties from the inverse Hessian approximation
    σ_map = sqrt.(diag(Optim.trace(result_map)[end].metadata["~inv(H)"]))
    σ_mle = sqrt.(diag(Optim.trace(result_mle)[end].metadata["~inv(H)"]))
    # Need to account for the logarithmic transformation
    σ_map .*= μ_map
    σ_mle .*= μ_mle
    # return (map = LogTransformMDFResult(μ_map, σ_map, Optim.trace(result_map)[end].metadata["~inv(H)"], result_map),
    #         mle = LogTransformMDFResult(μ_mle, σ_mle, Optim.trace(result_mle)[end].metadata["~inv(H)"], result_mle))
    return (map = (μ = μ_map, σ = σ_map, invH = Optim.trace(result_map)[end].metadata["~inv(H)"], result = result_map),
            mle = (μ = μ_mle, σ = σ_mle, invH = Optim.trace(result_map)[end].metadata["~inv(H)"], result = result_mle))
end
fit_templates_logamr(models::AbstractVector{<:AbstractMatrix{<:Number}}, data::AbstractMatrix{<:Number}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}; kws...) = fit_templates_logamr(stack_models(models), vec(data), logAge, metallicities; kws...)


# Calculate loglikelihood and gradient with respect to SFR parameters and logarithmic AMR parameters
@inline function fg_logamr_fixedσ!(F, G, variables::AbstractVector{<:Number}, models::Union{AbstractMatrix{<:Number}, AbstractVector{<:AbstractMatrix{<:Number}}}, data::Union{AbstractVector{<:Number}, AbstractMatrix{<:Number}}, composite::Union{AbstractVector{<:Number}, AbstractMatrix{<:Number}}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}, max_age::Number, MH_func, MH_deriv_Z, σ::Number)
    # `variables` should have length `length(unique(logAge)) + 2`; coeffs for each unique
    # entry in logAge, plus α and β to define the MDF at fixed logAge; σ is provided as a separate fixed argument
    @assert axes(data) == axes(composite)
    S = promote_type(eltype(variables), eltype(eltype(models)), eltype(eltype(data)), eltype(composite), eltype(logAge), eltype(metallicities), typeof(max_age))
    # Compute the coefficients on each model template given the `variables` andthe MDF
    α, β = variables[end-1], variables[end]
    unique_logAge = unique(logAge)
    max_logAge = log10(max_age) + 9
    # Calculate the per-template coefficents and normalization values
    coeffs = calculate_coeffs_logamr(view(variables,firstindex(variables):lastindex(variables)-2), logAge, metallicities, α, β, σ; MH_func=MH_func, max_logAge=max_logAge)

    # Fill the composite array with the equivalent of sum( coeffs .* models )
    # composite = sum( coeffs .* models )
    # return -loglikelihood(composite, data)
    composite!(composite, coeffs, models)
    logL = loglikelihood(composite, data) # Need to do this before ∇loglikelihood! because it will overwrite composite
    if (F != nothing) & (G != nothing) # Optim.optimize wants objective and gradient
        @assert axes(G) == axes(variables)
        # Calculate the ∇loglikelihood with respect to model coefficients; we will need all of these
        fullG = Vector{eltype(G)}(undef,length(coeffs)) # length(models))
        ∇loglikelihood!(fullG, composite, models, data)
        # Now need to do the transformation to the `variables` rather than model coefficients
        G[end-1] = zero(eltype(G))
        G[end] = zero(eltype(G))
        dZdβ = 1 # Derivative of metal mass fraction with respect to β
        for i in axes(G,1)[begin:end-2] 
            la = unique_logAge[i]
            age = exp10(la) / 1e9 # the / 1e9 makes α the slope in MH/Gyr, improves convergence
            # meanZ = α * age + β # Mean metal mass fraction
            meanZ = α * (max_age - age) + β # Mean metal mass fraction
            μ = MH_func(meanZ) # Find the mean metallicity [M/H] of this age bin
            idxs = findall( ==(la), logAge) # Find all entries that match this logAge
            tmp_coeffs = [_gausspdf(metallicities[j], μ, σ) for j in idxs] # Calculate relative weights
            A = sum(tmp_coeffs)
            # This should be correct for any MDF model at fixed logAge
            @inbounds G[i] = -sum( fullG[j] * coeffs[j] / variables[i] for j in idxs )
            βsum = sum( ((metallicities[idxs[j]]-μ) * tmp_coeffs[j]) for j in eachindex(idxs))
            dAdμ = -sum( fullG[idxs[j]] * variables[i] *
                ( ((metallicities[idxs[j]]-μ) * tmp_coeffs[j]) - tmp_coeffs[j] / A * βsum )
                         for j in eachindex(idxs)) / A / σ^2
            dμdZ = MH_deriv_Z(meanZ) # Derivative of MH_func(Z) with respect to Z
            dZdα = (max_age - age) # Derivative of metal mass fraction with respect to α
            dLdβ = dAdμ * dμdZ * dZdβ
            dLdα = dLdβ * dZdα
            G[end-1] += dLdα
            G[end] += dLdβ
        end
        return -logL
    elseif F != nothing # Optim.optimize wants only objective
        return -logL
    end
end

# for fixed σ
function fit_templates_logamr(models::AbstractMatrix{S},
                              data::AbstractVector{<:Number},
                              logAge::AbstractVector{<:Number},
                              metallicities::AbstractVector{<:Number},
                              σ::Number;
                              x0 = vcat(construct_x0_mdf(logAge, convert(S,log10(13.7e9))),
                                        [1e-4, 5e-5]),
                              MH_func = MH_from_Z,
                              MH_deriv_Z = dMH_dZ,
                              max_logAge = maximum(logAge),
                              kws...) where {S <: Number}
    composite = Vector{S}(undef,length(data)) # Scratch matrix for storing complex Hess model
    unique_logage = unique(logAge)
    max_age = exp10(max_logAge) / 1e9 # Lookback time at which to normalize β in Gyr
    @assert length(x0) == length(unique_logage) + 2
    @assert size(models,1) == length(data)
    @assert size(models,2) == length(logAge) == length(metallicities)
    # All variables must be positive  since Z = α * (max_age - age) + β with age being
    # positive lookback time, α must always be positive for an increasing AMR.
    # Perform logarithmic transformation on the provided x0 for all variables.
    x0 = log.(x0)
    # Make scratch array for assessing transformations
    x = similar(x0)
    # Define wrapper function to pass to Optim.only_fg!
    # It looks like you don't need the Jacobian correction to arrive at the maximum likelihood
    # result, and if you remove the Jacobian corrections it actually converges to the
    # non-log-transformed case. However, the uncertainty estimates from the inverse Hessian
    # don't seem reliable without the Jacobian corrections.
    function fg_logamr!_map(F, G, xvec)
        # All variables are log-transformed
        x .= exp.(xvec) 
        logL = fg_logamr_fixedσ!(F, G, x, models, data, composite,
                          logAge, metallicities, max_age, MH_func, MH_deriv_Z, σ)
        # Appy Jacobian correction to logL; α should be the same as
        # D[Log[x],x] = D[Log[-x],x] = 1/x, Log[1/x] = -Log[x]
        logL -= sum( xvec ) 
        # Add the Jacobian correction for every element of G
        G .= G .* x .- 1
        return logL
    end
    function fg_logamr!_mle(F, G, xvec)
        # All variables are log-transformed
        x .= exp.(xvec) 
        logL = fg_logamr_fixedσ!(F, G, x, models, data, composite,
                          logAge, metallicities, max_age, MH_func, MH_deriv_Z, σ)
        G .= G .* x 
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
    result_map = Optim.optimize(Optim.only_fg!( fg_logamr!_map ), x0, bfgs_struct, bfgs_options)
    result_mle = Optim.optimize(Optim.only_fg!( fg_logamr!_mle ), Optim.minimizer(result_map), bfgs_struct, bfgs_options)
    # Transform the resulting variables
    μ_map = exp.(deepcopy( Optim.minimizer(result_map) ))
    μ_mle = exp.(deepcopy( Optim.minimizer(result_mle) ))

    # Estimate parameter uncertainties from the inverse Hessian approximation
    σ_map = sqrt.(diag(Optim.trace(result_map)[end].metadata["~inv(H)"]))
    σ_mle = sqrt.(diag(Optim.trace(result_mle)[end].metadata["~inv(H)"]))
    # Need to account for the logarithmic transformation
    σ_map .*= μ_map
    σ_mle .*= μ_mle
    # return (map = LogTransformMDFResult(μ_map, σ_map, Optim.trace(result_map)[end].metadata["~inv(H)"], result_map),
    #         mle = LogTransformMDFResult(μ_mle, σ_mle, Optim.trace(result_mle)[end].metadata["~inv(H)"], result_mle))
    return (map = (μ = μ_map, σ = σ_map, invH = Optim.trace(result_map)[end].metadata["~inv(H)"], result = result_map),
            mle = (μ = μ_mle, σ = σ_mle, invH = Optim.trace(result_map)[end].metadata["~inv(H)"], result = result_mle))
end
fit_templates_logamr(models::AbstractVector{<:AbstractMatrix{<:Number}}, data::AbstractMatrix{<:Number}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}, σ::Number; kws...) = fit_templates_logamr(stack_models(models), vec(data), logAge, metallicities, σ; kws...)

" Not yet implemented "
function hmc_sample_logamr()
end

include("fixed_log_amr.jl")
