"""
    BFGSResult(μ::AbstractVector{<:Number},
               σ::AbstractVector{<:Number},
               invH::AbstractMatrix{<:Number},
               result,
               MH_model::AbstractMetallicityModel,
               disp_model::AbstractDispersionModel)

Type for containing the maximum likelihood estimate (MLE) *or* maximum a posteriori (MAP) results from BFGS optimizations that use Optim.jl. Fields are as follows:

 - `μ` contains the final values of the fitting parameters. The `mode` and `median` methods will both return `μ`, but the mean of samples is not always equal to `μ` due to the variable transformations we perform.
 - `σ` contains the standard errors estimated for the parameters and is returned by the `std` method.
 - `invH` is the BFGS approximation to the inverse Hessian, which is an estimator for the covariance matrix of the parameters if the objective function is approximately Gaussian near the best-fit `μ`.
 - `result` is the full result object returned by Optim.jl.
 - `MH_model` is the best-fit metallicity model.
 - `disp_model` is the best-fit metallicity dispersion model.

This type is implemented as a subtype of `Distributions.Sampleable{Multivariate, Continuous}` to enable sampling from an estimate of the likelihood / posterior distribution constructed from the `invH`. You can obtain `N::Integer` samples from the distribution with `rand(R, N)` where `R` is an instance of this type. This will return a size `length(μ) x N` matrix.

You can also directly obtain the per-SSP template coefficients (``r_{j,k}`` in the [derivation](@ref mzr_derivation)) using the optimization results stored in a `BFGSResult` with [`calculate_coeffs`](@ref).

# See also
 - [`CompositeBFGSResult`](@ref StarFormationHistories.CompositeBFGSResult) is a type that contains two instances of `BFGSResult`, one for the MAP and one for the MLE.
"""
struct BFGSResult{A <: AbstractVector{<:Number},
                  B <: AbstractVector{<:Number},
                  C <: AbstractMatrix{<:Number},
                  D,
                  E <: AbstractMetallicityModel,
                  F <: AbstractDispersionModel} <: Sampleable{Multivariate, Continuous}
    μ::A
    σ::B
    invH::C
    result::D
    MH_model::E
    disp_model::F
end
Base.length(result::BFGSResult) = length(result.μ)
mode(result::BFGSResult) = result.μ
median(result::BFGSResult) = result.μ
std(result::BFGSResult) = result.σ
function _rand!(rng::AbstractRNG, result::BFGSResult,
                samples::Union{AbstractVector{S}, DenseMatrix{S}}) where {S <: Real}
    
    MH_model, disp_model = result.MH_model, result.disp_model
    μ = result.μ
    Nbins = length(μ) - nparams(MH_model) - nparams(disp_model) # Number of SFR parameters
    dist = MvNormal(Optim.minimizer(result.result), result.invH)
    # Construct view into samples to get rows corresponding to free parameters
    tf = (transforms(MH_model)..., transforms(disp_model)...)
    free = SVector(free_params(MH_model)..., free_params(disp_model)...)
    # Now sure why but this vcat causes _rand! to make an allocation for each sample
    # Still somehow faster than using a LazyArrays.Vcat ...
    row_idxs = vcat(1:Nbins, (Nbins+1:Nbins+length(free))[free])
    # row_idxs = LazyArrays.Vcat(1:Nbins, (Nbins+1:Nbins+length(free))[free])
    fittable_view = view(samples, row_idxs, :)
    # _rand!(rng, dist, fittable_view) # See issue #59
    tmpsample = rand(rng, dist, size(samples, 2))
    samples[row_idxs, :] .= tmpsample
    # Now perform variable transformations for free metallicity and dispersion parameters
    exptransform_samples!(fittable_view, μ, tf[free], free[free])
    # Now write in fixed parameters
    par = (values(fittable_params(MH_model))..., values(fittable_params(disp_model))...)
    for i in 1:length(free)
        if ~free[i] # if parameter is fixed,
            samples[Nbins+i, :] .= par[i]
        end
    end
    return samples
end

"""
    calculate_coeffs(result::Union{BFGSResult, CompositeBFGSResult}
                     logAge::AbstractVector{<:Number},
                     metallicities::AbstractVector{<:Number})

Returns per-SSP stellar mass coefficients (``r_{j,k}`` in the [derivation](@ref mzr_derivation)) using the optimized metallicity model, metallicity dispersion model, and stellar mass coefficients from the result of a BFGS optimization. In the case that the provided `result` is a `CompositeBFGSResult` containing both the maximum a posteriori and maximum likelihood estimate (MLE), the MLE is used to construct the coefficients.
"""
function calculate_coeffs(result::BFGSResult,
                          logAge::AbstractVector{<:Number},
                          metallicities::AbstractVector{<:Number})
    MH_model, disp_model, μ = result.MH_model, result.disp_model, result.μ
    # Get number of stellar mass coefficients
    Nbins = length(μ) - nparams(MH_model) - nparams(disp_model)
    return calculate_coeffs(result.MH_model, result.disp_model,
                            @view(μ[begin:Nbins]), logAge,
                            metallicities)
end

# Put MAP and MLE result together so we can use MLE for best-fit values
# and MAP for its better conditioned inverse Hessian approximation
"""
    CompositeBFGSResult(map::BFGSResult, mle::BFGSResult)

Type for containing the maximum a posteriori (MAP) *AND* maximum likelihood estimate (MLE) results from BFGS optimizations that use Optim.jl, which are individually accessible via the `:mle` and `:map` properties (i.e., for an instance of this type `t`, `t.mle` or `getproperty(t, :mle)` and `t.map` or `getproperty(t, :map)`).

Random samples can be drawn from an instance `t` as `rand(t, N::Integer)`. This will return a size `length(μ) x N` matrix. This will use the MLE result for the best-fit values and the inverse Hessian approximation to the covariance matrix from the MAP result, which is more robust when best-fit values that are constrained to be positive approach 0.

Per-SSP coefficients can be calculated with `calculate_coeffs(result::CompositeBFGSResult, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number})`, which uses the MLE result (see [these docs](@ref StarFormationHistories.calculate_coeffs(::BFGSResult, ::AbstractVector{<:Number}, ::AbstractVector{<:Number}))).
"""
struct CompositeBFGSResult{A <: BFGSResult,
                           B <: BFGSResult} <: Sampleable{Multivariate, Continuous}
    map::A
    mle::B
end
Base.length(result::CompositeBFGSResult) = length(result.map)
mode(result::CompositeBFGSResult) = result.mle.μ
median(result::CompositeBFGSResult) = result.mle.μ
std(result::CompositeBFGSResult) = result.map.σ
function _rand!(rng::AbstractRNG,
                result::CompositeBFGSResult,
                samples::Union{AbstractVector{S}, DenseMatrix{S}}) where {S <: Real}

    MLE, MAP = result.mle, result.map
    MH_model, disp_model = MLE.MH_model, MLE.disp_model
    μ = MLE.μ
    Nbins = length(μ) - nparams(MH_model) - nparams(disp_model) # Number of SFR parameters
    dist = MvNormal(Optim.minimizer(MLE.result), MAP.invH)
    # Construct view into samples to get rows corresponding to free parameters
    tf = (transforms(MH_model)..., transforms(disp_model)...)
    free = SVector(free_params(MH_model)..., free_params(disp_model)...)
    # Now sure why but this vcat causes _rand! to make an allocation for each sample
    # Still somehow faster than using a LazyArrays.Vcat ...
    row_idxs = vcat(1:Nbins, (Nbins+1:Nbins+length(free))[free])
    # row_idxs = LazyArrays.Vcat(1:Nbins, (Nbins+1:Nbins+length(free))[free])
    fittable_view = view(samples, row_idxs, :)
    # _rand!(rng, dist, fittable_view) # See issue #59
    tmpsample = rand(rng, dist, size(samples, 2))
    samples[row_idxs, :] .= tmpsample
    # Now perform variable transformations for free metallicity and dispersion parameters
    exptransform_samples!(fittable_view, μ, tf[free], free[free])
    # Now write in fixed parameters
    par = (values(fittable_params(MH_model))..., values(fittable_params(disp_model))...)
    for i in 1:length(free)
        if ~free[i] # if parameter is fixed,
            samples[Nbins+i, :] .= par[i]
        end
    end
    return samples
end

calculate_coeffs(result::CompositeBFGSResult, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}) = calculate_coeffs(result.mle, logAge, metallicities)
