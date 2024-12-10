# Code that is generic between AMRs and MZRs will be placed here

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
julia> exptransform((log(0.5), -1.0, 0.0), (1, 0, 1)) ≈ [0.5, -1.0, 1.0]
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

function exptransform_samples!(samples::AbstractVecOrMat{<:Number},
                               μ::AbstractVector{<:Number}, 
                               transformations,
                               free)
    
    Base.require_one_based_indexing(samples)
    # length of transformations and free are equal to the number of
    # metallicity model parameters plus dispersion model parameters
    @assert length(transformations) == length(free)
    # size(samples) = (nvariables, nsamples)
    Nsamples = size(samples, 2)
    Nbins = size(samples, 1) - length(free) # Number of stellar mass coefficients / SFRs
    # Perform variable transformations, first for SFR parameters
    for i in axes(samples,1)[begin:Nbins]
        for j in axes(samples,2)
            samples[i,j] = exp(samples[i,j])
        end
    end
    for i in axes(samples,1)[Nbins+1:end]
        tfi = transformations[i - Nbins]
        freei = free[i - Nbins] # true if variable is free, false if fixed
        if freei # Variable is free, -- transform samples if necessary
            if tfi == 1
                for j in axes(samples,2)
                    samples[i,j] = exp(samples[i,j])
                end
            elseif tfi == -1
                for j in axes(samples,2)
                    samples[i,j] = -exp(samples[i,j])
                end
                # elseif tfi == 0
                #     continue
            end
        else # Variable is fixed -- overwrite samples with μi
            μi = μ[i]
            for j in axes(samples,2)
                samples[i,j] = μ[i]
            end
        end
    end
end

"""
    fit_sfh(Zmodel0::AbstractMetallicityModel,
            disp0::AbstractDispersionModel,
            models::AbstractMatrix{<:Number},
            data::AbstractVector{<:Number},
            logAge::AbstractVector{<:Number},
            metallicities::AbstractVector{<:Number};
            x0::AbstractVector{<:Number} = <...>
            kws...)

Returns a [`CompositeBFGSResult`](@ref StarFormationHistories.CompositeBFGSResult) instance that contains the maximum a posteriori (MAP) and maximum likelihood estimates (MLE) obtained from fitting the provided simple stellar population (SSP) templates `models` (with logarithmic ages `logAge = log10(age [yr])` and metallicities `metallicities`) to the provided `data`. The metallicity evolution is modelled using the provided `Zmodel0`, whose parameters can be free or fixed, with metallicity dispersion at fixed time modelled by `disp0`, whose parameters can be free or fixed.

This method is designed to work best with a "grid" of stellar models, defined by the outer product of `N` unique entries in `logAge` and `M` unique entries in `metallicities`. See the examples for more information on usage.

We provide several options for age-metallicity relations and mass-metallicity relations that can be used for `Zmodel0` and define APIs for users to create new models that will integrate with this function. Similar flexibility is allowed for the metallicity dispersion model `disp0`.

The primary method signature uses flattened formats for `models` and `data`. See the notes for the flattened call signature of [`StarFormationHistories.composite!`](@ref) for more details, as well as [`stack_models`](@ref StarFormationHistories.stack_models) that facilitates rearranging the `models` into this flattened format.

# Arguments
 - `Zmodel0` is an instance of [`AbstractMetallicityModel`](@ref StarFormationHistories.AbstractMetallicityModel) that defines how the average metallicity stars being formed in the population changes over time. The fittable parameters contained in this instance are used as the initial values to start the optimization. 
 - `disp0` is an instance of [`AbstractDispersionModel`](@ref StarFormationHistories.AbstractDispersionModel) that defines the distribution of metallicities of stars forming in a fixed time bin (i.e., the dispersion in metallicity around the mean at fixed time). The fittable parameters contained in this instance are used as the initial values to start the optimization. 
 - `models` are the template Hess diagrams for the SSPs that compose the observed Hess diagram.
 - `data` is the Hess diagram for the observed data.
 - `logAge::AbstractVector{<:Number}` is the vector containing the effective ages of the stellar populations used to create the templates in `models`, in units of `log10(age [yr])`. For example, if a population has an age of 1 Myr, its entry in `logAge` should be `log10(10^6) = 6.0`.
 - `metallicities::AbstractVector{<:Number}` is the vector containing the effective metallicities of the stellar populations used to create the templates in `models`. This is most commonly a logarithmic abundance like [M/H] or [Fe/H], but you could use a linear abundance like the metal mass fraction Z if you wanted to. There are some notes on the [Wikipedia](https://en.wikipedia.org/wiki/Metallicity) that might be useful.

# Keyword Arguments
 - `x0` is the vector of initial guesses for the stellar mass coefficients per *unique* entry in `logAge`. We try to set reasonable defaults, but in most cases users should be calculating and passing this keyword argument. We provide [`StarFormationHistories.construct_x0_mdf`](@ref) to prepare `x0` assuming a constant star formation rate and total stellar mass, which is typically a good initial guess.

# Returns
 - This function returns a [`CompositeBFGSResult`](@ref StarFormationHistories.CompositeBFGSResult) that contains the output from both MLE and MAP optimizations, accessible via `result.mle` and `result.map`. These are each instances of [`BFGSResult`](@ref StarFormationHistories.BFGSResult). See the docs for these structs for more information.
"""
function fit_sfh end

include("dispersion_models.jl")
include("bfgs_result.jl")
include("fixed_amr.jl")
include("linear_amr/linear_amr.jl")
include("log_amr/log_amr.jl")
include("MZR/mzr_models.jl")
include("MZR/mzr_fitting.jl")
