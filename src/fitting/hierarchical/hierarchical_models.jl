# Code that is generic between AMRs and MZRs will be placed here

""" `AbstractMetallicityModel{T <: Real}` is the abstract supertype for all hierarchical metallicity models. Abstract subtypes are [`AbstractAMR`](@ref StarFormationHistories.AbstractAMR) for age-metallicity relations and [`AbstractMZR`](@ref StarFormationHistories.AbstractMZR) for mass-metallicity relations. """
abstract type AbstractMetallicityModel{T <: Real} end
Base.Broadcast.broadcastable(m::AbstractMetallicityModel) = Ref(m)

include("transformations.jl")   # Variable transformations
include("dispersion_models.jl") # AbstractDispersionModel and subtypes
include("bfgs_result.jl")       # BFGSResult and CompositeBFGSResult types
include("fixed_amr.jl")         # Fit under a fixed AMR -- deprecate?
include("linear_amr/linear_amr.jl")
include("log_amr/log_amr.jl")
include("amr.jl")               # Age-metallicity relations
include("mzr.jl")               # Mass-metallicity relations
include("generic_fitting.jl")   # Fitting and sampling functions for both AMR and MZR
