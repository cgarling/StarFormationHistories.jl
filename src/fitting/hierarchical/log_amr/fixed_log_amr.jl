# Gradient-based optimization for SFH given a fixed input linear age-Z relation,
# which becomes a logarithmic age-[M/H] relation.
# Uses a fixed Gaussian spread σ in [M/H]. 

function fixed_log_amr(models, #::AbstractVector{<:AbstractMatrix{<:Number}},
                       data,   #::AbstractMatrix{<:Number},
                       logAge::AbstractVector{<:Number},
                       metallicities::AbstractVector{<:Number},
                       α::Number,
                       β::Number,
                       σ::Number;
                       MH_func=MH_from_Z,
                       kws...) #where {S <: Number, T <: AbstractMatrix{S}}
    
    # Calculate relative per-model weights since AMR is fixed
    relweights = calculate_coeffs_logamr( ones(length(unique(logAge))), logAge, metallicities, α, β, σ; MH_func = MH_func)
    return fixed_amr(models, data, logAge, metallicities, relweights; kws...)
end

"
    (α, β) = calculate_αβ_logamr(low_constraint, high_constraint, Z_func=Z_from_MH)
Calculates linear Z (log [M/H]) AMR slope α and intercept β from two points on the line given by the first two arguments; more info given in [`StarFormationHistories.fixed_log_amr`](@ref). 
"
function calculate_αβ_logamr(low_constraint, high_constraint, Z_func=Z_from_MH)
    # Written so that order of the constraints doesn't actually matter...
    times = (last(low_constraint), last(high_constraint))
    δt = maximum(times) - minimum(times)
    MH_constraints = (first(low_constraint), first(high_constraint))
    Z_min = Z_func(minimum(MH_constraints))
    Z_max = Z_func(maximum(MH_constraints))
    α = -(Z_max - Z_min) / δt
    β = Z_func(maximum(MH_constraints)) - α * minimum(times)
    return α, β
end

"""
    fixed_log_amr(models, data, logAge, metallicities, constraint1, constraint2, σ; Z_func=Z_from_MH, kws...)
Call signature that takes two fixed points `low_constraint` and `high_constraint` that define points that must lie on the logarithmic age-metallicity relation and calculates the paramters α and β for you. Format is ([M/H], age [Gyr]), i.e. `constraint1 = (-2.5,13.7)` for the first point at [M/H] = -2.5 at 13.7 Gyr lookback time and `constraint2 = (-0.8,0.0)` for the second point at [M/H] = -0.8 at present-day (0.0 Gyr lookback time). Metallicities in [M/H] iare converted to metal mass fractions Z via the provided callable keyword argument `Z_func` which defaults to [`StarFormationHistories.Z_from_MH`](@ref).
"""
function fixed_log_amr(models, data, logAge, metallicities, low_constraint, high_constraint, σ; Z_func=Z_from_MH, kws...)
    α, β = calculate_αβ_logamr(low_constraint, high_constraint, Z_func)
    return fixed_log_amr(models, data, logAge, metallicities, α, β, σ; kws...)
end
