# Methods for performing variable transformations
# We use a squared transformation (x = y²) to enforce positivity constraints.
# This maps the real line to [0, ∞) without the singularity at 0 that log/exp has.
# When the true parameter x → 0, the optimization variable y → 0 (finite),
# unlike log(x) → -∞ which can contaminate BFGS Hessian approximations.
# This approach follows Dolphin (2013).

"""
    forward_transform(params, transformations)
Applies forward (parameter → optimization space) transform to parameters `params`:
 - ``y = \\text{sign}(x) \\sqrt{|x|}`` for each element in `transformations` that `==1` (positive constraint, so `y = \\sqrt{x}`).
 - ``y = x`` for each element in `transformations` that `==0` (unconstrained).
 - ``y = \\sqrt{-x}`` for each element in `transformations` that `==-1` (negative constraint).

See also [`inverse_transform`](@ref) which performs the inverse transformation.

# Examples
```jldoctest; setup=:(using StarFormationHistories: forward_transform)
julia> forward_transform((0.25, -1.0, 1.0), (1, 0, 1)) ≈ [0.5, -1.0, 1.0]
true
```
"""
forward_transform(params, transformations) =
    [ begin
         tfi = transformations[i]
         if tfi == 1
             sqrt(params[i])
         elseif tfi == 0
             params[i]
         elseif tfi == -1
             sqrt(-params[i])
         end
      end
      for i in eachindex(params, transformations) ]

"""
    inverse_transform(params, transformations)
Applies inverse (optimization space → parameter) transform to parameters `params`, effectively inverting [`forward_transform`](@ref):
 - ``x = y^2`` for each element in `transformations` that `==1` (positive constraint).
 - ``x = y`` for each element in `transformations` that `==0` (unconstrained).
 - ``x = -y^2`` for each element in `transformations` that `==-1` (negative constraint).

# Examples
```jldoctest; setup=:(using StarFormationHistories: inverse_transform)
julia> inverse_transform((sqrt(0.5), -1.0, 1.0), (1, 0, 1)) ≈ [0.5, -1.0, 1.0]
true
```
"""
inverse_transform(params, transformations) =
    [ begin
         tfi = transformations[i]
         if tfi == 1
             params[i]^2
         elseif tfi == 0
             params[i]
         elseif tfi == -1
             -(params[i]^2)
         end
      end
      for i in eachindex(params, transformations) ]

# Keep old names as aliases for backward compatibility in case any user code references them
const logtransform = forward_transform
const exptransform = inverse_transform

"""
    inverse_transform_samples!(samples, μ, transformations, free)

Applies inverse transformation to columns of `samples` in-place. Stellar mass coefficient
rows (first `Nbins` rows) are squared, and metallicity/dispersion model parameter rows are
transformed according to `transformations`. Fixed parameters are overwritten with their
values from `μ`.
"""
function inverse_transform_samples!(samples::AbstractVecOrMat{<:Number},
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
    # Perform variable transformations, first for SFR parameters (y² transform)
    for i=1:Nbins, j=1:size(samples,2)
        samples[i,j] = samples[i,j]^2
    end
    for i in Nbins+1:size(samples,1)
        tfi = transformations[i - Nbins]
        freei = free[i - Nbins] # true if variable is free, false if fixed
        if freei # Variable is free -- transform samples if necessary
            if tfi == 1
                for j in 1:size(samples,2)
                    samples[i,j] = samples[i,j]^2
                end
            elseif tfi == -1
                for j in 1:size(samples,2)
                    samples[i,j] = -(samples[i,j]^2)
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

# Keep old name as alias
const exptransform_samples! = inverse_transform_samples!
