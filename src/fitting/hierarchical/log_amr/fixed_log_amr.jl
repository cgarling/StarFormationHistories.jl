# Gradient-based optimization for SFH given a fixed input linear age-Z relation,
# which becomes a logarithmic age-[M/H] relation.
# Uses a fixed Gaussian spread σ in [M/H]. 

"""
    fixed_log_amr(models,
                  data,
                  logAge::AbstractVector{<:Number},
                  metallicities::AbstractVector{<:Number},
                  T_max::Number,
                  α::Number,
                  β::Number,
                  σ::Number;
                  MH_func = StarFormationHistories.MH_from_Z,
                  kws...)

Given a fully specified logarithmic age-metallicity relation with parameters (α, β, σ), fits maximum likelihood and maximum a posteriori star formation parameters. `MH_func` is a callable that returns a logarithmic metallicity [M/H] for a metal mass fraction argument and defaults to [`MH_from_Z`](@ref StarFormationHistories.MH_from_Z). `T_max` is the lookback time in Gyr at which the mean metal mass fraction is `\beta`. See [`fixed_amr`](@ref StarFormationHistories.fixed_amr) for info on format of returned result.
"""
function fixed_log_amr(models, #::AbstractVector{<:AbstractMatrix{<:Number}},
                       data,   #::AbstractMatrix{<:Number},
                       logAge::AbstractVector{<:Number},
                       metallicities::AbstractVector{<:Number},
                       T_max::Number,
                       α::Number,
                       β::Number,
                       σ::Number;
                       MH_func=MH_from_Z,
                       kws...) #where {S <: Number, T <: AbstractMatrix{S}}
    
    # Calculate relative per-model weights since AMR is fixed
    relweights = calculate_coeffs_logamr( ones(length(unique(logAge))), logAge, metallicities, T_max, α, β, σ; MH_func = MH_func)
    return fixed_amr(models, data, logAge, metallicities, relweights; kws...)
end

"""
    (α, β) = calculate_αβ_logamr(low_constraint,
                                 high_constraint,
                                 T_max,
                                 Z_func=Z_from_MH)

Calculates linear Z (log [M/H]) age-metallicity relation (AMR) slope α and intercept β from two points on the line with form `([M/H], age [Gyr])` given by the first two arguments. The AMR is normalized so that the mean metal mass fraction at a lookback time in Gyr of `T_max` is `Z = β`. More info given in [`fixed_log_amr`](@ref StarFormationHistories.fixed_log_amr). 
"""
function calculate_αβ_logamr(low_constraint,
                             high_constraint,
                             T_max,
                             Z_func = Z_from_MH)
    # Written so that order of the constraints doesn't actually matter...
    times = (last(low_constraint), last(high_constraint))
    times_max = maximum(times)
    @assert T_max >= times_max
    δt = times_max - minimum(times)
    MH_constraints = (first(low_constraint), first(high_constraint))
    Z_min = Z_func(minimum(MH_constraints))
    Z_max = Z_func(maximum(MH_constraints))
    # Z = α * (T_max - age)[Gyr] + β = α * δt + β
    α = (Z_max - Z_min) / δt
    # β = Z_func(minimum(MH_constraints)) # By definition, Z at earliest time
    β = Z_min - α * (T_max - times_max)
    @assert (α > 0) & (β > 0)
    return α, β
end

"""
    fixed_log_amr(models,
                  data,
                  logAge::AbstractVector{<:Number},
                  metallicities::AbstractVector{<:Number},
                  T_max::Number,
                  constraint1,
                  constraint2,
                  σ::Number;
                  Z_func = StarFormationHistories.Z_from_MH,
                  kws...)

Call signature that takes two fixed points `low_constraint` and `high_constraint` that define points that must lie on the logarithmic age-metallicity relation and calculates the slope paramters α and β for you. Format is ([M/H], age [Gyr]), i.e. `constraint1 = (-2.5, 13.7)` for the first point at [M/H] = -2.5 at 13.7 Gyr lookback time and `constraint2 = (-0.8, 0.0)` for the second point at [M/H] = -0.8 at present-day (0.0 Gyr lookback time). The AMR is normalized so that the mean metal mass fraction at a lookback time in Gyr of `T_max` is `Z = β`. Metallicities in [M/H] are converted to metal mass fractions Z via the provided callable keyword argument `Z_func` which defaults to [`Z_from_MH`](@ref StarFormationHistories.Z_from_MH). See [`fixed_amr`](@ref StarFormationHistories.fixed_amr) for info on format of returned result.
"""
function fixed_log_amr(models, data, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}, T_max::Number, low_constraint, high_constraint, σ::Number; Z_func=Z_from_MH, max_age=max(last(low_constraint),last(high_constraint)), kws...)
    α, β = calculate_αβ_logamr(low_constraint, high_constraint, T_max, Z_func)
    return fixed_log_amr(models, data, logAge, metallicities, T_max, α, β, σ; kws...)
end
