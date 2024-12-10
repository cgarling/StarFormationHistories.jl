""" Abstract type for all models of metallicity dispersion at fixed time ``t_j``, for which the mean metallicity is ``\\mu_j``. Concrete subtypes `T <: AbstractDispersionModel` should implement the following API: 
 - `(model::T)(x::Real, μ::Real)` should be defined so that the struct is callable with a metallicity `x` and a mean metallicity `μ`, returning the relative weight for the metallicity `x` given the dispersion model. This is ``A_{j,k}`` for ``\\mu = \\mu_j`` in the derivations presented in the documentation.
 - `nparams(model::T)` should return the number of fittable parameters in the model.
 - `fittable_params(model::T)` should return the values of the fittable parameters in the model.
 - `gradient(model::T, x::Real, μ::Real)` should return a tuple that contains the partial derivative of the ``A_{j,k}`` with respect to each fittable model parameter, plus the partial derivative with respect to `μ` as the final element.
 - `update_params(model::T, newparams)` should return a new instance of `T` with the fittable parameters contained in `newparams` (which is typically a vector or tuple) and non-fittable parameters inherited from the provided `model`.
 - `transforms(model::T)` should return a tuple of length `nparams(model)` which indicates how the fittable variables should be transformed for optimization, if at all. Elements should be `1` for parameters that are constrained to always be positive, `0` for parameters that can be positive or negative, and `-1` for parameters that are constrained to always be negative.
 - `free_params(model::T)` should return an `NTuple{nparams(model), Bool}` that is `true` for fittable parameters that you want to optimize and `false` for fittable parameters that you want to stay fixed during optimization. """
abstract type AbstractDispersionModel{T <: Real} end
Base.Broadcast.broadcastable(m::AbstractDispersionModel) = Ref(m)

"""
    nparams(model::AbstractDispersionModel)::Int
Returns the number of fittable parameters in the model. 
"""
nparams(model::AbstractDispersionModel)
"""
    fittable_params(model::AbstractDispersionModel{T})::NTuple{nparams(model), T}
Returns the values of the fittable parameters in the provided dispersion model `model`.
"""
fittable_params(model::AbstractDispersionModel)
"""
    gradient(model::AbstractDispersionModel{T}, x::Real, μ::Real)::NTuple{nparams(model)+1, T}
 Returns a tuple containing the partial derivative of the `model` with respect to all fittable parameters, plus the partial derivative with respect to the mean metallicity `μ` as the final element. These partial derivatives are evaluated at metallicity `x` where the model has expectation value `μ`.
"""
gradient(model::AbstractDispersionModel, x::Real, μ::Real)
"""
    update_params(model::T, newparams)::T where {T <: AbstractDispersionModel}
Returns a new instance of the model type `T` with the fittable parameters contained in `newparams` (which is typically a vector or tuple), with non-fittable parameters inherited from the provided `model`. 
"""
update_params(model::AbstractDispersionModel, newparams::Any)
"""
    transforms(model::AbstractDispersionModel)::NTuple{nparams(model), Int}
Returns a tuple of length `nparams(model)` which indicates how the fittable variables should be transformed for optimization, if at all. Elements should be `1` for parameters that are constrained to always be positive, `0` for parameters that can be positive or negative, and `-1` for parameters that are constrained to always be negative.
"""
transforms(model::AbstractDispersionModel)
"""
    free_params(model::AbstractDispersionModel)::NTuple{nparams(model), Bool}
Returns an tuple of length `nparams(model)` that is `true` for fittable parameters that you want to optimize and `false` for fittable parameters that you want to stay fixed during optimization.
 """
free_params(model::AbstractDispersionModel)


#### Concrete subtypes

"""
    GaussianDispersion(σ::Real, free::NTuple{1, Bool} = (true,)) <: AbstractDispersionModel
Dispersion model for a Gaussian (i.e., Normal) spread in metallicities with standard deviation `σ` (which must be greater than 0) at fixed age. The relative weights for this model are given by ``A_{j,k} = \\exp(-((x_k - μ_j)/σ)^2/2).`` The `σ` can be fit during optimizations if `free == (true,)` or fixed if `free == (false,)`.

# Examples
```jldoctest; setup=:(using StarFormationHistories: nparams, gradient, update_params, transforms, free_params)
julia> GaussianDispersion(0.2) isa GaussianDispersion{Float64}
true

julia> import Test

julia> Test.@test_throws(ArgumentError, GaussianDispersion(-0.2)) isa Test.Pass
true

julia> nparams(GaussianDispersion(0.2)) == 1
true

julia> GaussianDispersion(0.2)(1.0, 1.2) ≈ exp(-0.5)
true

julia> all(values(gradient(GaussianDispersion(0.2), 1.0, 1.2)) .≈
                (3.0326532985631656, -3.0326532985631656))
true

julia> update_params(GaussianDispersion(0.2), 0.3) == GaussianDispersion(0.3)
true

julia> transforms(GaussianDispersion(0.2)) == (1,)
true

julia> free_params(GaussianDispersion(0.2, (false,))) == (false,)
true
```
"""
struct GaussianDispersion{T <: Real} <: AbstractDispersionModel{T}
    σ::T
    free::NTuple{1, Bool}
    GaussianDispersion(σ::T, free::NTuple{1, Bool}) where T <: Real =
        σ ≤ zero(T) ? throw(ArgumentError("σ must be > 0")) : new{T}(σ, free)
end
# If constructed without `free`, assume all variables are free
GaussianDispersion(σ::Real) = GaussianDispersion(σ, (true,))
# Number of fittable parameters
nparams(model::GaussianDispersion) = 1
fittable_params(model::GaussianDispersion) = (σ = model.σ,)
# Make struct callable to compute relative weights
(model::GaussianDispersion)(x::Real, μ::Real) = exp(-((x-μ)/model.σ)^2/2)
# `gradient` for `AbstractDispersionModel` subtypes is partial derivative with respect
# to each fittable model parameter, and at the end, partial derivative with respect to μ
function gradient(model::GaussianDispersion, x::Real, μ::Real)
    A_jk = model(x, μ)
    σ = model.σ
    return (σ = A_jk * (x - μ)^2 / σ^3,
            μ = A_jk * (x - μ) / σ^2)
end
# Make a new instance of `GaussianDispersion` with the new fitting parameters `newparams`,
# inheriting any fixed parameters from old instance `model`
update_params(model::GaussianDispersion, newparams) =
    GaussianDispersion(first(newparams), model.free)
# Whether each fittable parameter is constrained to always be negative (-1), positive (1),
# or unconstrained (0); used to calculate variable transformations
transforms(::GaussianDispersion) = (1,)
# Determine which fittable parameters are free to be fit,
# and which are to be fixed during optimization
free_params(model::GaussianDispersion) = model.free
