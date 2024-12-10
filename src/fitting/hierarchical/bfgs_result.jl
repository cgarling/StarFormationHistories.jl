"""
    BFGSResult(μ::AbstractVector{<:Number},
               σ::AbstractVector{<:Number},
               invH::AbstractMatrix{<:Number},
               result,
               Zmodel::AbstractMetallicityModel,
               dispmodel::AbstractDispersionModel)

Type for containing the maximum likelihood estimate (MLE) *or* maximum a posteriori (MAP) results from BFGS optimizations that use Optim.jl. Fields are as follows:

 - `μ` contains the final values of the fitting parameters. The `mode` and `median` methods will both return `μ`, but the mean of samples is not always equal to `μ` due to the variable transformations we perform.
 - `σ` contains the standard errors estimated for the parameters and is returned by the `std` method.
 - `invH` is the BFGS approximation to the inverse Hessian, which is an estimator for the covariance matrix of the parameters if the objective function is approximately Gaussian near the best-fit `μ`.
 - `result` is the full result object returned by Optim.jl.
 - `Zmodel` is the best-fit metallicity model.
 - `dispmodel` is the best-fit metallicity dispersion model.

This type is implemented as a subtype of `Distributions.Sampleable{Multivariate, Continuous}` to enable sampling from an estimate of the likelihood / posterior distribution constructed from the `invH`. You can obtain `N::Integer` samples from the distribution with `rand(R, N)` where `R` is an instance of this type. This will return a size `(length(μ)+2) x N` matrix.

# See also
 - [`CompositeBFGSResult`](@ref StarFormationHistories.CompositeBFGSResult) is a type that contains two instances of `BFGSResult`; one for the MAP and one for the MLE.
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
    Zmodel::E
    dispmodel::F
end
Base.length(result::BFGSResult) = length(result.μ)
mode(result::BFGSResult) = result.μ
median(result::BFGSResult) = result.μ
std(result::BFGSResult) = result.σ
function _rand!(rng::AbstractRNG, result::BFGSResult,
                samples::Union{AbstractVector{S}, DenseMatrix{S}}) where {S <: Real}
    
    Zmodel, dispmodel = result.Zmodel, result.dispmodel
    μ = result.μ
    Nbins = length(μ) - nparams(Zmodel) - nparams(dispmodel) # Number of SFR parameters
    dist = MvNormal(Optim.minimizer(result.result), result.invH)
    _rand!(rng, dist, samples)
    # Now perform variable transformations for metallicity and dispersion models
    tf = (transforms(Zmodel)..., transforms(dispmodel)...)
    free = (free_params(Zmodel)..., free_params(dispmodel)...)
    exptransform_samples!(samples, μ, tf, free)
    return samples
end

# Put MAP and MLE result together so we can use MLE for best-fit values
# and MAP for its better conditioned inverse Hessian approximation
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
    Zmodel, dispmodel = MLE.Zmodel, MLE.dispmodel
    μ = MLE.μ
    Nbins = length(μ) - nparams(Zmodel) - nparams(dispmodel) # Number of SFR parameters
    dist = MvNormal(Optim.minimizer(MLE.result), MAP.invH)
    _rand!(rng, dist, samples)
    
    # Now perform variable transformations for metallicity and dispersion models
    tf = (transforms(Zmodel)..., transforms(dispmodel)...)
    free = (free_params(Zmodel)..., free_params(dispmodel)...)
    exptransform_samples!(samples, μ, tf, free)
    return samples
end
