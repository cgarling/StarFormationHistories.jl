# Background models for fit_sfh: a fixed Hess diagram shape scaled by the expected number of background stars,
# which follows the same parameter API as the metallicity and dispersion models (nparams, fittable_params,
# update_params, transforms, free_params) so the background amplitude is fit alongside their parameters

""" `AbstractBackground` is the abstract supertype of the background models accepted by [`fit_sfh`](@ref), [`NoBackground`](@ref) and [`HessBackground`](@ref). """
abstract type AbstractBackground end

"""
    NoBackground()
Background model for fits without a background, so that every observed star must be explained by the stellar population templates. This is the default for [`fit_sfh`](@ref).
"""
struct NoBackground <: AbstractBackground end

"""
    HessBackground(shape::AbstractArray{<:Real}, b::Union{Real, Nothing}=nothing; floor::Real=0.05, free::Bool=true)
Background model for [`fit_sfh`](@ref) in which the expected number of stars in Hess diagram bin ``i`` is

```math
m_i = \\sum_{j,k} r_{j,k} \\, c_{i,j,k} + b \\, u_i
```

where ``u_i`` is the background shape and ``b`` is the expected number of background stars in the fitted bins. Background stars are contaminants such as unresolved background galaxies or foreground stars, or stars the templates do not model. Without a background, observed stars in bins where the templates are (nearly) zero distort the fit, biasing the stellar mass high and the metallicity dispersion broad.

`shape` is the background Hess diagram over the same bins as the `data` passed to [`fit_sfh`](@ref), e.g., the Hess diagram of a nearby control field, or `ones(length(data))` for a uniform background. It is normalized to sum to 1, after which the fraction `floor` of the background is spread uniformly over the bins, so that stars in bins where both `shape` and the templates are zero can be explained by the background. `b` is the initial value of the expected number of background stars (a number of stars, not a fraction), which is fit if `free` is `true` and fixed otherwise. If `b` is `nothing`, [`fit_sfh`](@ref) starts from 1% of the observed stars. Like the stellar mass coefficients ``R_j``, ``b`` is fit in logarithmic coordinates, so it must be positive to be fit, and starting far from its best-fit value (e.g., at a single star when there are thousands of contaminants) slows or destabilizes the fit. A fixed `b` must be given, e.g., the number of stars in a control field scaled by the ratio of the areas.

A uniform background removes most of the bias from contamination spread over the Hess diagram, and costs nothing when there is no contamination, since ``b`` then goes to zero. Contamination concentrated in part of the Hess diagram, such as faint background galaxies, should be modeled with its own `shape`; a uniform background explains only part of it.

After fitting, the best-fit background model is stored in the `bg_model` field of the [`BFGSResult`](@ref StarFormationHistories.BFGSResult), and its expected Hess diagram is `bg_model.b .* bg_model.shape`. ``b`` is the last element of the fitted parameters `μ`.
"""
struct HessBackground{T <: Real} <: AbstractBackground
    shape::Vector{T}
    b::Union{T, Nothing} # `nothing` until fit_sfh sets the initial value from the data
    free::NTuple{1, Bool}
end
function HessBackground(shape::AbstractArray{<:Real}, b::Union{Real, Nothing}=nothing; floor::Real=0.05, free::Bool=true)
    if isnothing(b)
        free || throw(ArgumentError("A fixed background needs its expected number of stars `b`."))
    else
        @argcheck (free ? b > 0 : b >= 0) "The initial expected number of background stars `b` must be positive to be fit."
    end
    u = background_shape(vec(shape), floor)
    T = isnothing(b) ? eltype(u) : promote_type(eltype(u), typeof(b))
    return HessBackground{T}(u, b, (free,))
end

# Background shape normalized to sum to 1 over the fitted bins `keep`, with the fraction `floor` spread uniformly
# over them; also used by SSPModel
function background_shape(shape::AbstractVector, floor::Real, keep=trues(length(shape)))
    @argcheck 0 <= floor <= 1 "The background `floor` must be between 0 and 1."
    @argcheck all(>=(0), shape) "`background` must not be negative."
    total = sum(view(shape, keep))
    total > 0 || throw(ArgumentError("`background` is zero in every fitted bin."))
    return (1 - floor) .* shape ./ total .+ floor / count(keep)
end

nparams(::NoBackground) = 0
fittable_params(::NoBackground) = NamedTuple()
update_params(model::NoBackground, newparams) = model
transforms(::NoBackground) = ()
free_params(::NoBackground) = ()

nparams(::HessBackground) = 1
fittable_params(model::HessBackground) = (b = model.b,)
update_params(model::HessBackground{T}, newparams) where T = HessBackground{T}(model.shape, first(newparams), model.free)
transforms(::HessBackground) = (1,)
free_params(model::HessBackground) = model.free

# Expected number of background stars
background_stars(::NoBackground) = 0
background_stars(model::HessBackground) = model.b

# Sets the initial expected number of background stars to 1% of the observed stars if it was not given
initialize_background(model::AbstractBackground, data) = model
initialize_background(model::HessBackground, data) = isnothing(model.b) ? update_params(model, (sum(data) / 100,)) : model

# Adds the expected background to the `composite` model Hess diagram
add_background!(composite, ::NoBackground) = composite
function add_background!(composite, model::HessBackground)
    @argcheck length(model.shape) == length(composite) "The background shape must have one entry per Hess diagram bin of the data."
    vec(composite) .+= model.b .* model.shape
    return composite
end

# Writes the partial derivative of the negative log likelihood with respect to b into the last element of `G`,
# given `composite` as overwritten with `1 .- data ./ composite` by `∇loglikelihood!`
background_gradient!(G, ::NoBackground, composite) = G
function background_gradient!(G, model::HessBackground, composite)
    G[end] = dot(model.shape, vec(composite))
    return G
end
