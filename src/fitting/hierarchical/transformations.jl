# Methods for performing variable transformations

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
    
    Base.require_one_based_indexing(samples, μ, transformations, free)
    # length of transformations and free are equal to the number of
    # metallicity model parameters plus dispersion model parameters
    @argcheck length(transformations) == length(free)
    # size(samples) = (nvariables, nsamples)
    Nsamples = size(samples, 2)
    Nbins = size(samples, 1) - length(free) # Number of stellar mass coefficients / SFRs
    # Perform variable transformations, first for SFR parameters
    for i=1:Nbins, j=1:size(samples,2)
        samples[i,j] = exp(samples[i,j])
    end
    for i in Nbins+1:size(samples,1)
        tfi = transformations[i - Nbins]
        freei = free[i - Nbins] # true if variable is free, false if fixed
        if freei # Variable is free, -- transform samples if necessary
            if tfi == 1
                for j in 1:size(samples,2)
                    samples[i,j] = exp(samples[i,j])
                end
            elseif tfi == -1
                for j in 1:size(samples,2)
                    samples[i,j] = -exp(samples[i,j])
                end
                # elseif tfi == 0
                #     continue
            end
        else # Variable is fixed -- overwrite samples with μi
            μi = μ[i]
            for j in 1:size(samples,2)
                samples[i,j] = μ[i]
            end
        end
    end
end
