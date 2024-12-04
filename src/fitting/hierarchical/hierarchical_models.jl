""" `AbstractMetallicityModel{T <: Real}` is the abstract supertype for all hierarchical metallicity models. Abstract subtypes are `AbstractAMR` for age-metallicity relations and [`AbstractMZR`](@ref StarFormationHistories.AbstractMZR) for mass-metallicity relations. """
abstract type AbstractMetallicityModel{T <: Real} end
Base.Broadcast.broadcastable(m::AbstractMetallicityModel) = Ref(m)

"""
    logtransform(params, transformations)
Applies logarithmic transform to parameters `params`; applied transformations are as follows:
 - ``x^\\prime = \\text{log}(x)`` for each element in `transformations` that `==1`.
 - ``x^\\prime = x`` for each element in `transformations` that `==0`.
 - ``x^\\prime = log(-x)`` for each element in `transformations` that `==0`.

See also `exptransform` which performs the inverse transformation.

# Examples
```jldoctest; setup=:(using StarFormationHistories: logtransform)
julia> logtransform((0.5, -1.0, 1.0), (1, 0, 1)) ≈ [log(0.5), -1.0, 0.0]
true
```
"""
logtransform(params, transformations) =
    [ begin
         tfi = transformations[i]
         if tfi == 1
             log(params[i])
         elseif tfi == 0
             params[i]
         elseif tfi == -1
             log(-params[i])
         end
      end
      for i in eachindex(params, transformations) ]
"""
    exptransform(params, transformations)
Applies exponential transform to parameters `params`, effectively inverting `logtransform`. Applied transformations are as follows:
 - ``x^\\prime = \\text{exp}(x)`` for each element in `transformations` that `==1`.
 - ``x^\\prime = x`` for each element in `transformations` that `==0`.
 - ``x^\\prime = -\\text{exp}(x)`` for each element in `transformations` that `==0`.

# Examples
```jldoctest; setup=:(using StarFormationHistories: exptransform)
julia> logtransform((log(0.5), -1.0, 0.0), (1, 0, 1)) ≈ [0.5, -1.0, 1.0]
true
```
"""
exptransform(params, transformations) =
    [ begin
         tfi = transformations[i]
         if tfi == 1
             exp(params[i])
         elseif tfi == 0
             params[i]
         elseif tfi == -1
             -exp(params[i])
         end
      end
      for i in eachindex(params, transformations) ]


include("dispersion_models.jl")
include("fixed_amr.jl")
include("linear_amr/linear_amr.jl")
include("log_amr/log_amr.jl")
include("MZR/mzr_models.jl")
include("MZR/mzr_fitting.jl")
