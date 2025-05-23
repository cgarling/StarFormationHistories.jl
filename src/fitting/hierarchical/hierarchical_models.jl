# Code that is generic between AMRs and MZRs will be placed here

""" `AbstractMetallicityModel{T <: Real}` is the abstract supertype for all hierarchical metallicity models. Abstract subtypes are [`AbstractAMR`](@ref StarFormationHistories.AbstractAMR) for age-metallicity relations and [`AbstractMZR`](@ref StarFormationHistories.AbstractMZR) for mass-metallicity relations. """
abstract type AbstractMetallicityModel{T <: Real} end
Base.Broadcast.broadcastable(m::AbstractMetallicityModel) = Ref(m)

"""
    nparams(models...)
Returns the sum of the number of fittable parameters for each `model` in `models` via `mapreduce(nparams, +, models)`.

# Examples
```jldoctest; setup = :(import StarFormationHistories: nparams, LinearAMR, GaussianDispersion)
julia> nparams(LinearAMR(1.0, 1.0), GaussianDispersion(0.2))
3
```
"""
nparams(models...) = mapreduce(nparams, +, models)

include("construct_x0_mdf.jl")  # utility function to set initial guess x0
include("transformations.jl")   # Variable transformations
include("dispersion_models.jl") # AbstractDispersionModel and subtypes
include("bfgs_result.jl")       # BFGSResult and CompositeBFGSResult types
include("fixed_amr.jl")         # Fit under a fixed AMR -- deprecate?
include("amr.jl")               # Age-metallicity relations
include("mzr.jl")               # Mass-metallicity relations
include("generic_fitting.jl")   # Fitting and sampling functions for both AMR and MZR
