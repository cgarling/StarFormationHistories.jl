# This file contains types and methods implementing mass-metallicity relations
# for use with the SFH fitting routines in mzr_fitting.jl

""" `AbstractMZR{T <: Real} <: AbstractMetallicityModel{T}`: abstract supertype for all metallicity models that are mass-metallicity relations. Concrete subtypes `T <: AbstractMZR` should implement the following API: 
 - `(model::T)(Mstar::Real)` should be defined so that the struct is callable with a stellar mass `Mstar` in solar masses, returning the mean metallicity given the MZR model. This is ``\\mu_j \\left( \\text{M}_* \\right)`` in the derivations presented in the documentation.
 - `nparams(model::T)` should return the number of fittable parameters in the model.
 - `fittable_params(model::T)` should return the values of the fittable parameters in the model.
 - `gradient(model::T, Mstar::Real)` should return a tuple that contains the partial derivative of the mean metallicity ``\\mu_j`` with respect to each fittable model parameter, plus the partial derivative with respect to the stellar mass `Mstar` as the final element.
 - `update_params(model::T, newparams)` should return a new instance of `T` with the fittable parameters contained in `newparams` (which is typically a vector or tuple) and non-fittable parameters inherited from the provided `model`.
 - `transforms(model::T)` should return a tuple of length `nparams(model)` which indicates how the fittable variables should be transformed for optimization, if at all. Elements should be `1` for parameters that are constrained to always be positive, `0` for parameters that can be positive or negative, and `-1` for parameters that are constrained to always be negative.
 - `free_params(model::T)` should return an `NTuple{nparams(model), Bool}` that is `true` for fittable parameters that you want to optimize and `false` for fittable parameters that you want to stay fixed during optimization. """
abstract type AbstractMZR{T <: Real} <: AbstractMetallicityModel{T} end
Base.Broadcast.broadcastable(m::AbstractMZR) = Ref(m)

"""
    nparams(model::AbstractMZR)::Int
Returns the number of fittable parameters in the model. 
"""
nparams(model::AbstractMZR)
"""
    fittable_params(model::AbstractMZR{T})::NTuple{nparams(model), T}
Returns the values of the fittable parameters in the provided MZR `model`.
"""
fittable_params(model::AbstractMZR)
"""
    gradient(model::AbstractMZR{T}, Mstar::Real)::NTuple{nparams(model)+1, T}
 Returns a tuple containing the partial derivative of the mean metallicity with respect to all fittable parameters, plus the partial derivative with respect to the stellar mass `Mstar` as the final element. These partial derivatives are evaluated at stellar mass `Mstar`.
"""
gradient(model::AbstractMZR, Mstar::Real)
"""
    update_params(model::T, newparams)::T where {T <: AbstractMZR}
Returns a new instance of the model type `T` with the fittable parameters contained in `newparams` (which is typically a vector or tuple), with non-fittable parameters inherited from the provided `model`. 
"""
update_params(model::AbstractMZR, newparams::Any)
"""
    transforms(model::AbstractMZR)::NTuple{nparams(model), Int}
Returns a tuple of length `nparams(model)` which indicates how the fittable variables should be transformed for optimization, if at all. Elements should be `1` for parameters that are constrained to always be positive, `0` for parameters that can be positive or negative, and `-1` for parameters that are constrained to always be negative.
"""
transforms(model::AbstractMZR)
"""
    free_params(model::AbstractMZR)::NTuple{nparams(model), Bool}
Returns an tuple of length `nparams(model)` that is `true` for fittable parameters that you want to optimize and `false` for fittable parameters that you want to stay fixed during optimization.
 """
free_params(model::AbstractMZR)

###################
# Concrete subtypes

"""
    PowerLawMZR(α::Real, MH0::Real, logMstar0::Real=6,
                free::NTuple{2, Bool}=(true, true)) <: AbstractMZR
Mass-metallicity model described by a single power law index `α > 0`, a metallicity normalization `MH0`, and the logarithm of the stellar mass `logMstar0 = log10(Mstar0 [M⊙])` at which the mean metallicity is `MH0`. Because `logMstar0` and `MH0` are degenerate, we treat `MH0` as a fittable parameter and `logMstar0` as a fixed parameter that will not be changed during optimizations. Such a power law MZR is often used when extrapolating literature results to low masses, e.g., ``\\text{M}_* < 10^8 \\; \\text{M}_\\odot.`` `α` will be fit freely during optimizations if `free[1] == true` and `MH0` will be fit freely if `free[2] == true`. The MZR is defined by

```math
\\begin{aligned}
[\\text{M} / \\text{H}] \\left( \\text{M}_* \\right) &= [\\text{M} / \\text{H}]_0 + \\text{log} \\left( \\left( \\frac{\\text{M}_*}{\\text{M}_{*,0}} \\right)^\\alpha \\right) \\\\
&= [\\text{M} / \\text{H}]_0 + \\alpha \\, \\left( \\text{log} \\left( \\text{M}_* \\right) - \\text{log} \\left( \\text{M}_{*,0} \\right) \\right) \\\\
\\end{aligned}
```

# Examples
```jldoctest; setup=:(using StarFormationHistories: nparams, gradient, update_params, transforms, free_params)
julia> PowerLawMZR(1.0, -1) isa PowerLawMZR{Float64}
true

julia> import Test

julia> Test.@test_throws(ArgumentError, PowerLawMZR(-1.0, -1)) isa Test.Pass
true

julia> nparams(PowerLawMZR(1.0, -1)) == 2
true

julia> PowerLawMZR(1.0, -1, 6)(1e7) ≈ 0
true

julia> all(values(gradient(PowerLawMZR(1.0, -1, 6), 1e8)) .≈
                (2.0, 1.0, 1 / 1e8 / log(10)))
true

julia> update_params(PowerLawMZR(1.0, -1, 7, (true, false)), (2.0, -2)) ==
         PowerLawMZR(2.0, -2, 7, (true, false))
true

julia> transforms(PowerLawMZR(1.0, -1)) == (1,0)
true

julia> free_params(PowerLawMZR(1.0, -1, 7, (true, false))) == (true, false)
true
```
"""
struct PowerLawMZR{T <: Real} <: AbstractMZR{T}
    α::T   # Power-law slope
    MH0::T # Normalization / intercept
    logMstar0::T # log10(Mstar) at which [M/H] = MH0
    free::NTuple{2, Bool}
    PowerLawMZR(α::T, MH0::T, logMstar0::T, free::NTuple{2, Bool}) where T <: Real =
        α ≤ zero(T) ? throw(ArgumentError("α must be > 0")) : new{T}(α, MH0, logMstar0, free)
end
PowerLawMZR(α::Real, MH0::Real, logMstar0::Real=6, free::NTuple{2, Bool}=(true, true)) =
    PowerLawMZR(promote(α, MH0, logMstar0)..., free)
nparams(d::PowerLawMZR) = 2
fittable_params(d::PowerLawMZR) = (α = d.α, β = d.MH0)
(mzr::PowerLawMZR)(Mstar::Real) = mzr.MH0 + mzr.α * (log10(Mstar) - mzr.logMstar0)
gradient(model::PowerLawMZR{T}, Mstar::S) where {T, S <: Real} =
    (α = log10(Mstar) - model.logMstar0,
     β = one(promote_type(T, S)),
     # \frac{\partial \mu_j}{\partial M_*} = \frac{\partial \mu_j}{\partial R_j}
     Mstar = model.α / Mstar / logten)
update_params(model::PowerLawMZR, newparams) =
    PowerLawMZR(newparams..., model.logMstar0, model.free)
transforms(::PowerLawMZR) = (1, 0)
free_params(model::PowerLawMZR) = model.free

