# Gradient-based optimization for SFH given a fixed input linear age-Z relation,
# which becomes a logarithmic age-[M/H] relation.
# Uses a fixed Gaussian spread σ in [M/H]. 

"""
    fixed_log_amr(models,
                  data,
                  logAge::AbstractVector{<:Number},
                  metallicities::AbstractVector{<:Number},
                  α::Number,
                  β::Number,
                  σ::Number;
                  MH_func = StarFormationHistories.MH_from_Z,
                  max_logAge = maximum(logAge),
                  kws...)

Given a fully specified logarithmic age-metallicity relation with parameters (α, β, σ), fits maximum likelihood and maximum a posteriori star formation parameters. `MH_func` is a callable that returns a logarithmic metallicity [M/H] for a metal mass fraction argument and defaults to [`MH_from_Z`](@ref StarFormationHistories.MH_from_Z). `max_logAge` is the earliest time for which the age-metallicity relation is defined to hold; by default it simply takes the maximum of the provided `logAge` vector argument. See [`fixed_amr`](@ref StarFormationHistories.fixed_amr) for info on format of returned result.
"""
function fixed_log_amr(models, #::AbstractVector{<:AbstractMatrix{<:Number}},
                       data,   #::AbstractMatrix{<:Number},
                       logAge::AbstractVector{<:Number},
                       metallicities::AbstractVector{<:Number},
                       α::Number,
                       β::Number,
                       σ::Number;
                       MH_func=MH_from_Z,
                       max_logAge=maximum(logAge),
                       kws...) #where {S <: Number, T <: AbstractMatrix{S}}
    
    # Calculate relative per-model weights since AMR is fixed
    relweights = calculate_coeffs_logamr( ones(length(unique(logAge))), logAge, metallicities, α, β, σ; MH_func = MH_func, max_logAge = max_logAge)
    return fixed_amr(models, data, logAge, metallicities, relweights; kws...)
end

"""
    (α, β) = calculate_αβ_logamr(low_constraint,
                                 high_constraint,
                                 Z_func=Z_from_MH;
                                 max_age = max(last(low_constrant), last(high_constraint)))

Calculates linear Z (log [M/H]) age-metallicity relation slope α and intercept β from two points on the line with form `([M/H], age [Gyr])` given by the first two arguments. By default it is assumed that one of these points is the earliest time you are interested in, at which by definition `Z = β`. You can provide a larger lookback time to normalize β by providing a larger `max_age` (units of Gyr). More info given in [`fixed_log_amr`](@ref StarFormationHistories.fixed_log_amr). 
"""
function calculate_αβ_logamr(low_constraint,
                             high_constraint,
                             Z_func = Z_from_MH;
                             max_age = max(last(low_constraint), last(high_constraint)))
    # Written so that order of the constraints doesn't actually matter...
    times = (last(low_constraint), last(high_constraint))
    t_max = maximum(times)
    @assert max_age >= t_max
    δt = t_max - minimum(times)
    MH_constraints = (first(low_constraint), first(high_constraint))
    Z_min = Z_func(minimum(MH_constraints))
    Z_max = Z_func(maximum(MH_constraints))
    # Z = α * (max_age - age)[Gyr] + β = α * δt + β
    α = (Z_max - Z_min) / δt
    # β = Z_func(minimum(MH_constraints)) # By definition, Z at earliest time
    β = Z_min - α * (max_age - t_max)
    @assert (α > 0) & (β > 0)
    return α, β
end

"""
    fixed_log_amr(models,
                  data,
                  logAge::AbstractVector{<:Number},
                  metallicities::AbstractVector{<:Number},
                  constraint1,
                  constraint2,
                  σ::Number;
                  Z_func = StarFormationHistories.Z_from_MH,
                  max_age = maximum(logAge),
                  kws...)

Call signature that takes two fixed points `low_constraint` and `high_constraint` that define points that must lie on the logarithmic age-metallicity relation and calculates the slope paramters α and β for you. Format is ([M/H], age [Gyr]), i.e. `constraint1 = (-2.5, 13.7)` for the first point at [M/H] = -2.5 at 13.7 Gyr lookback time and `constraint2 = (-0.8, 0.0)` for the second point at [M/H] = -0.8 at present-day (0.0 Gyr lookback time). By default it is assumed that one of these points is the earliest time you are interested in, at which by definition `Z = β`. You can provide a larger lookback time to normalize β by providing a larger `max_age` (units of Gyr). Metallicities in [M/H] are converted to metal mass fractions Z via the provided callable keyword argument `Z_func` which defaults to [`Z_from_MH`](@ref StarFormationHistories.Z_from_MH). See [`fixed_amr`](@ref StarFormationHistories.fixed_amr) for info on format of returned result.
"""
function fixed_log_amr(models, data, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}, low_constraint, high_constraint, σ::Number; Z_func=Z_from_MH, max_age=max(last(low_constraint),last(high_constraint)), kws...)
    α, β = calculate_αβ_logamr(low_constraint, high_constraint, Z_func; max_age = max_age)
    return fixed_log_amr(models, data, logAge, metallicities, α, β, σ; kws...)
end
