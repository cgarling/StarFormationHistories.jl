# This file contains types and methods implementing mass-metallicity relations
# for use with the SFH fitting routines in generic_fitting.jl

# AbstractAMR API

""" `AbstractAMR{T <: Real} <: AbstractMetallicityModel{T}`: abstract supertype for all metallicity models that are age-metallicity relations. Concrete subtypes `T <: AbstractAMR` should implement the following API: 
 - `(model::T)(logAge::Real)` should be defined so that the struct is callable with a logarithmic age (`log10(age [yr])`), returning the mean metallicity given the AMR model. This is ``\\mu_j \\left( t_j \\right)`` in the derivations presented in the documentation.
 - `nparams(model::T)` should return the number of fittable parameters in the model.
 - `fittable_params(model::T)` should return the values of the fittable parameters in the model.
 - `gradient(model::T, logAge::Real)` should return a tuple that contains the partial derivative of the mean metallicity ``\\mu_j`` with respect to each fittable model parameter evaluated at logarithmic age `logAge`.
 - `update_params(model::T, newparams)` should return a new instance of `T` with the fittable parameters contained in `newparams` (which is typically a vector or tuple) and non-fittable parameters inherited from the provided `model`.
 - `transforms(model::T)` should return a tuple of length `nparams(model)` which indicates how the fittable variables should be transformed for optimization, if at all. Elements should be `1` for parameters that are constrained to always be positive, `0` for parameters that can be positive or negative, and `-1` for parameters that are constrained to always be negative.
 - `free_params(model::T)` should return an `NTuple{nparams(model), Bool}` that is `true` for fittable parameters that you want to optimize and `false` for fittable parameters that you want to stay fixed during optimization. """
abstract type AbstractAMR{T <: Real} <: AbstractMetallicityModel{T} end

"""
    nparams(model::AbstractAMR)::Int
Returns the number of fittable parameters in the model. 
"""
nparams(model::AbstractAMR)
"""
    fittable_params(model::AbstractAMR{T})::NTuple{nparams(model), T}
Returns the values of the fittable parameters in the provided AMR `model`.
"""
fittable_params(model::AbstractAMR)
"""
    gradient(model::AbstractAMR{T}, logAge::Real)::NTuple{nparams(model), T}
Returns a tuple containing the partial derivative of the mean metallicity with respect to all fittable parameters evaluated at logarithmic age `logAge`.
"""
gradient(model::AbstractAMR, logAge::Real)
"""
    update_params(model::T, newparams)::T where {T <: AbstractAMR}
Returns a new instance of the model type `T` with the fittable parameters contained in `newparams` (which is typically a vector or tuple), with non-fittable parameters inherited from the provided `model`. 
"""
update_params(model::AbstractAMR, newparams::Any)
"""
    transforms(model::AbstractAMR)::NTuple{nparams(model), Int}
Returns a tuple of length `nparams(model)` which indicates how the fittable variables should be transformed for optimization, if at all. Elements should be `1` for parameters that are constrained to always be positive, `0` for parameters that can be positive or negative, and `-1` for parameters that are constrained to always be negative.
"""
transforms(model::AbstractAMR)
"""
    free_params(model::AbstractAMR)::NTuple{nparams(model), Bool}
Returns an tuple of length `nparams(model)` that is `true` for fittable parameters that you want to optimize and `false` for fittable parameters that you want to stay fixed during optimization.
 """
free_params(model::AbstractAMR)

########################################################
# Calculating per-SSP weights (r_{j,k}) from AbstractAMR

function calculate_coeffs(amr_model::AbstractAMR{T}, disp_model::AbstractDispersionModel{U},
                          variables::AbstractVector{<:Number}, 
                          logAge::AbstractVector{<:Number},
                          metallicities::AbstractVector{<:Number}) where {T, U}
    unique_logAge = unique(logAge)
    @assert(length(variables) == length(unique_logAge),
            "Length of `variables` must be the same as `unique_logAge`.")
    @assert length(logAge) == length(metallicities)
    S = promote_type(eltype(variables), eltype(logAge), eltype(metallicities), T, U)

    coeffs = Vector{S}(undef, length(logAge))
    for i in eachindex(unique_logAge)
        la = unique_logAge[i]
        # Find the mean metallicity of this age bin based on the cumulative stellar mass
        μ = amr_model(la)
        idxs = findall(==(la), logAge)
        # Calculate relative weights
        tmp_coeffs = [disp_model(metallicities[ii], μ) for ii in idxs]
        A = sum(tmp_coeffs)
        # Make sure sum over tmp_coeffs equals 1 and write to coeffs
        coeffs[idxs] .= tmp_coeffs .* variables[i] ./ A
    end
    return coeffs
end

###############################################
# Compute objective and gradient for AMR models

function fg!(F, G, MHmodel0::AbstractAMR{T}, dispmodel0::AbstractDispersionModel{U},
             variables::AbstractVector{<:Number},
             models::Union{AbstractMatrix{<:Number},
                           AbstractVector{<:AbstractMatrix{<:Number}}},
             data::Union{AbstractVector{<:Number}, AbstractMatrix{<:Number}},
             composite::Union{AbstractVector{<:Number}, AbstractMatrix{<:Number}},
             logAge::AbstractVector{<:Number},
             metallicities::AbstractVector{<:Number}) where {T, U}

    @assert axes(data) == axes(composite)
    S = promote_type(eltype(variables), eltype(eltype(models)), eltype(eltype(data)),
                     eltype(composite), eltype(logAge), eltype(metallicities),
                     T, U)
    # Number of fittable parameters in MHmodel;
    # in G, these come after the stellar mass coefficients R_j
    Zpar = nparams(MHmodel0)
    # Number of fittable parameters in metallicity dispersion model;
    # in G, these come after the MHmodel parameters
    disppar = nparams(dispmodel0)

    # Construct new instance of MHmodel0 with updated parameters
    MHmodel = update_params(MHmodel0, @view(variables[end-(Zpar+disppar)+1:(end-disppar)]))
    # Get indices of free parameters in MZR model
    Zfree = BitVector(free_params(MHmodel))
    # Construct new instance of dispmodel0 with updated parameters
    dispmodel = update_params(dispmodel0, @view(variables[end-(disppar)+1:end]))
    # Get indices of free parameters in metallicity dispersion model
    dispfree = BitVector(free_params(dispmodel))
    # Calculate all coefficients r_{j,k} for each template
    coeffs = calculate_coeffs(MHmodel, dispmodel,
                              @view(variables[begin:end-(Zpar+disppar)]),
                              logAge, metallicities)

    # Fill the composite array with the equivalent of sum( coeffs .* models )
    # composite = sum( coeffs .* models )
    # return -loglikelihood(composite, data)
    composite!(composite, coeffs, models)
    # Need to do compute logL before ∇loglikelihood! because it will overwrite composite
    logL = loglikelihood(composite, data)

    if !isnothing(G) # Optim.optimize wants gradient -- update G in place
        Base.require_one_based_indexing(G)
        @assert axes(G) == axes(variables)
        unique_logAge = unique(logAge)
        # Calculate the ∇loglikelihood with respect to model coefficients
        fullG = Vector{eltype(G)}(undef, length(coeffs))
        ∇loglikelihood!(fullG, composite, models, data)
        # Zero-out gradient vector in preparation to accumulate sums
        G .= zero(eltype(G))
        # Loop over j
        for i in eachindex(unique_logAge)
            la = unique_logAge[i]
            μ = MHmodel(la)
            idxs = findall( ==(la), logAge) # Find all entries that match this logAge
            tmp_mh = metallicities[idxs]
            tmp_coeffs = dispmodel.(tmp_mh, μ)
            A = sum(tmp_coeffs)
            
            # Partial derivatives with respect to stellar mass coefficients
            # \begin{aligned}
            #     \frac{\partial F}{\partial R_j} &= \sum_k \frac{\partial F}{\partial r_{j,k}} \frac{\partial r_{j,k}}{\partial R_j} \\
            #     &= \sum_k \frac{\partial F}{\partial r_{j,k}} \frac{r_{j,k}}{R_j} \\
            # \end{aligned}
            G[i] = -sum(fullG[j] * coeffs[j] / variables[i] for j in idxs)

            # Gradient of MHmodel with respect to parameters
            gradμ = values(gradient(MHmodel, la))
            # Gradient of the dispersion model with respect to parameters
            # and mean metallicity μ_j
            grad_disp = tups_to_mat(values.(gradient.(dispmodel, tmp_mh, μ)))
            # \frac{\partial A_{j,k}}{\partial \mu_j}
            dAjk_dμj = grad_disp[end,:]
            
            # Partial derivatives with respect to MHmodel parameters
            ksum_dAjk_dμj = sum(dAjk_dμj)
            psum = -sum( fullG[idxs[j]] * variables[i] / A *
                (dAjk_dμj[j] - tmp_coeffs[j] / A * ksum_dAjk_dμj) for j in eachindex(idxs))
            for par in (1:Zpar)[Zfree]
                G[end-(Zpar+disppar-par)] += psum * gradμ[par]
            end

            # Partial derivatives with respect to dispmodel parameters
            for par in (1:disppar)[dispfree]
                # Extract the partial derivative of A_jk
                # with respect to parameter P
                dAjk_dP = grad_disp[par,:]
                ksum_dAjk_dP = sum(dAjk_dP)
                G[end-(disppar-par)] -= sum( fullG[idxs[j]] * variables[i] / A *
                    (dAjk_dP[j] - tmp_coeffs[j] / A * ksum_dAjk_dP) for j in eachindex(idxs))
            end
        end
    end
    if !isnothing(F) # Optim.optimize wants objective returned
        return -logL
    end
end

###################
# Concrete subtypes

############
# Linear AMR

struct LinearAMR{T <: Real} <: AbstractAMR{T}
    α::T     # Slope
    β::T     # Normalization / intercept
    T_max::T # Earliest valid lookback time in Gyr, at which <[M/H]> = β; e.g. 
    free::NTuple{2, Bool}
    function LinearAMR(α::T, β::T, T_max::T, free::NTuple{2, Bool}) where T <: Real
        if α < zero(T)
            throw(ArgumentError("α must be ≥ 0"))
        elseif T_max ≤ zero(T)
            throw(ArgumentError("T_max must be > 0"))
        end
        return new{T}(α, β, T_max, free)
    end
end
"""
    LinearAMR(α::Real,
              β::Real,
              T_max::Real = 137//10,
              free::NTuple{2, Bool} = (true, true))
Subtype of [`AbstractAMR`](@ref StarFormationHistories.AbstractAMR) implementing the linear age-metallicity relation where the mean metallicity at a lookback time ``t_j`` (in Gyr) is `μ_j = α * (T_max - t_j) + β`. `α` is therefore a slope describing the rate of change in the metallicity per Gyr, and `β` is the mean metallicity value of stars being born at a lookback time of `T_max`, which has units of Gyr. `free` controls whether `α` and `β` should be freely fit or fixed when passed into [`fit_sfh`](@ref); if `free[1] == true` then `α` will be freely fit, whereas it will fixed if `free[1] == false`. `free[2]` has the same effect but for `β`. 
"""
LinearAMR(α::Real, β::Real, T_max::Real=137//10, free::NTuple{2, Bool}=(true, true)) =
    LinearAMR(promote(α, β, T_max)..., free)
nparams(::LinearAMR) = 2
fittable_params(model::LinearAMR) = (α = model.α, β = model.β)
(model::LinearAMR)(logAge::Real) = model.β + model.α * (model.T_max - exp10(logAge - 9))
gradient(model::LinearAMR{T}, logAge::S) where {T, S <: Real} =
    (α = model.T_max - exp10(logAge - 9),
     β = one(promote_type(T, S)))
update_params(model::LinearAMR, newparams) =
    LinearAMR(newparams..., model.T_max, model.free)
transforms(::LinearAMR) = (1, 0)
free_params(model::LinearAMR) = model.free

"""
    LinearAMR(constraint1, constraint2, T_max::Real=137//10,
              free::NTuple{2, Bool}=(true, true))
Construct an instance of `LinearAMR` from MH constraints at two different lookback times. Each of `constraint1` and `constraint2` should be length-2 indexables (e.g., tuples) whose first element is a metallicity [M/H] and second element is a lookback time in Gyr. The order of the constraints does not matter. 

# Examples
```jldoctest; setup = :(import StarFormationHistories: LinearAMR)
julia> LinearAMR((-2.5, 13.7), (-1.0, 0.0), 13.7) isa LinearAMR{Float64}
true

julia> LinearAMR((-2.5, 13.7), (-1.0, 0.0), 13.7) == LinearAMR((-1.0, 0.0), (-2.5, 13.7), 13.7)
true
```
"""
function LinearAMR(constraint1, constraint2, T_max::Real=137//10,
                   free::NTuple{2, Bool}=(true, true))
    @assert length(constraint1) == length(constraint2) == 2
    times = (last(constraint1), last(constraint2))
    MH_constraints = (first(constraint1), first(constraint2))
    δt = times[2] - times[1]
    if δt == 0
        throw(ArgumentError("Constraints are given at identical times."))
    end
    δMH = MH_constraints[2] - MH_constraints[1]
    if !(sign(δt) != sign(δMH))
        throw(ArgumentError("The constraints passed to `LinearAMR` indicate metallicity is decreasing towards present-day. This is not allowed under the `LinearAMR` model. Note that the time variables in the constraints should be lookback times in units of Gyr."))
    end
    α = δMH / -δt
    β = minimum(MH_constraints) - α * (T_max - maximum(times))
    return LinearAMR(α, β, T_max, free)
end

#################
# Logarithmic AMR

struct LogarithmicAMR{T <: Real, S, V} <: AbstractAMR{T}
    α::T     # Slope
    β::T     # Normalization / intercept
    T_max::T # Earliest valid lookback time in Gyr, at which <[M/H]> = β; e.g.
    MH_func::S # Function to convert Z to MH
    dMH_dZ::V  # Function to return the derivative of MH_func with respect to Z
    free::NTuple{2, Bool}
    function LogarithmicAMR(α::T, β::T, T_max::T,
                            MH_func::S, dMH_dZ::V,
                            free::NTuple{2, Bool}) where {T <: Real, S, V}
        if α < zero(T)
            throw(ArgumentError("α must be ≥ 0"))
        elseif β < zero(T)
            throw(ArgumentError("β must be ≥ 0"))
        elseif T_max ≤ zero(T)
            throw(ArgumentError("T_max must be > 0"))
        end
        return new{T,S,V}(α, β, T_max, MH_func, dMH_dZ, free)
    end
end
"""
    LogarithmicAMR(α::Real,
                   β::Real,
                   T_max::Real = 137//10,
                   MH_func = MH_from_Z,
                   dMH_dZ = dMH_dZ,
                   free::NTuple{2, Bool} = (true, true))
Subtype of [`AbstractAMR`](@ref StarFormationHistories.AbstractAMR) implementing the logarithmic age-metallicity relation where the mean metal mass fraction `Z`  at a lookback time ``t_j`` (in Gyr) is `Z = α * (T_max - t_j) + β`. `α` is therefore a slope describing the rate of change in the metal mass fraction per Gyr, and `β` is the mean metal mass fraction of stars being born at a lookback time of `T_max`, which has units of Gyr. `MH_func` must be a callable (e.g., a function) that takes a single argument `Z` and converts it to [M/H]; the default function is appropriate for PARSEC stellar models. `dMH_dZ` must be a callable that takes a single argument `Z` and returns the derivative of `MH_func` with respect to `Z`. `free` controls whether `α` and `β` should be freely fit or fixed when passed into [`fit_sfh`](@ref); if `free[1] == true` then `α` will be freely fit, whereas it will fixed if `free[1] == false`. `free[2]` has the same effect but for `β`. 
"""
LogarithmicAMR(α::Real, β::Real, T_max::Real=137//10,
               MH_func = MH_from_Z, dMH_dZ = dMH_dZ, free::NTuple{2, Bool}=(true, true)) =
                   LogarithmicAMR(promote(α, β, T_max)..., MH_func, dMH_dZ, free)
nparams(::LogarithmicAMR) = 2
fittable_params(model::LogarithmicAMR) = (α = model.α, β = model.β)
(model::LogarithmicAMR)(logAge::Real) =
    model.MH_func(model.β + model.α * (model.T_max - exp10(logAge - 9)))
function gradient(model::LogarithmicAMR{T}, logAge::S) where {T, S <: Real}
    age = exp10(logAge - 9) # Convert logAge to age in Gyr
    μ_Z = model.β + model.α * (model.T_max - age)
    dMHdZ = model.dMH_dZ(μ_Z)
    dZdα = model.T_max - age
    dZdβ = one(promote_type(T, S))
    return (α = dMHdZ * dZdα,
            β = dMHdZ * dZdβ)
end
update_params(model::LogarithmicAMR, newparams) =
    LogarithmicAMR(newparams..., model.T_max, model.MH_func, model.dMH_dZ, model.free)
transforms(::LogarithmicAMR) = (1, 1)
free_params(model::LogarithmicAMR) = model.free


"""
    LogarithmicAMR(constraint1, constraint2, T_max::Real=137//10,
                   MH_func = MH_from_Z, dMH_dZ = dMH_dZ, Z_func = Z_from_MH,
                   free::NTuple{2, Bool}=(true, true))
Construct an instance of `LogarithmicAMR` from MH constraints at two different lookback times. Each of `constraint1` and `constraint2` should be length-2 indexables (e.g., tuples) whose first element is a metallicity [M/H] and second element is a lookback time in Gyr. The order of the constraints does not matter.

# Examples
```jldoctest; setup = :(import StarFormationHistories: LogarithmicAMR)
julia> LogarithmicAMR((-2.5, 13.7), (-1.0, 0.0), 13.7) isa LogarithmicAMR{Float64}
true

julia> LogarithmicAMR((-2.5, 13.7), (-1.0, 0.0), 13.7) ==
           LogarithmicAMR((-1.0, 0.0), (-2.5, 13.7), 13.7)
true
```
"""
function LogarithmicAMR(constraint1, constraint2, T_max::Real=137//10,
                        MH_func = MH_from_Z, dMH_dZ = dMH_dZ, Z_func = Z_from_MH,
                        free::NTuple{2, Bool}=(true, true))
    @assert length(constraint1) == length(constraint2) == 2
    times = (last(constraint1), last(constraint2))
    Z_constraints = (Z_func(first(constraint1)), Z_func(first(constraint2)))
    δt = times[2] - times[1]
    if δt == 0
        throw(ArgumentError("Constraints are given at identical times."))
    end
    δZ = Z_constraints[2] - Z_constraints[1]
    if !(sign(δt) != sign(δZ))
        throw(ArgumentError("The constraints passed to `LinearAMR` indicate metallicity is decreasing towards present-day. This is not allowed under the `LinearAMR` model. Note that the time variables in the constraints should be lookback times in units of Gyr."))
    end
    α = δZ / -δt
    β = minimum(Z_constraints) - α * (T_max - maximum(times))
    if β < 0
        throw(ArgumentError("Given constraints and Z_func result in a metal mass fraction Z < 0 at T_max. Please revise arguments."))
    end
    return LogarithmicAMR(α, β, T_max, MH_func, dMH_dZ, free)
end
