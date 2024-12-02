""" `AbstractMetallicityModel{T <: Real}` is the abstract supertype for all hierarchical metallicity models. Abstract subtypes are `AbstractAMR` for age-metallicity relations and [`AbstractMZR`](@ref StarFormationHistories.AbstractMZR) for mass-metallicity relations. """
abstract type AbstractMetallicityModel{T <: Real} end
Base.Broadcast.broadcastable(m::AbstractMetallicityModel) = Ref(m)

include("dispersion_models.jl")
include("fixed_amr.jl")
include("linear_amr/linear_amr.jl")
include("log_amr/log_amr.jl")
include("MZR/mzr_models.jl")
include("MZR/mzr_fitting.jl")
