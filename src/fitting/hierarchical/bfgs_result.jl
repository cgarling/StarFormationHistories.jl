"""
    BFGSResult(μ::AbstractVector{<:Number},
               σ::AbstractVector{<:Number},
               invH::AbstractMatrix{<:Number},
               result,
               Zmodel::AbstractMetallicityModel,
               dispmodel::AbstractDispersionModel)

Type for containing the maximum likelihood estimate (MLE) and maximum a posteriori (MAP) results from BFGS optimizations that use Optim.jl. Fields are as follows:

 - `μ` contains the final values of the fitting parameters. The `mode` and `median` methods will both return `μ`, but the mean of samples is not always equal to `μ` due to the variable transformations we perform.
 - `σ` contains the standard errors estimated for the parameters and is returned by the `std` method.
 - `invH` is the BFGS approximation to the inverse Hessian, which is an estimator for the covariance matrix of the parameters if the objective function is approximately Gaussian near the best-fit `μ`.
 - `result` is the full result object returned by Optim.jl.
 - `Zmodel` is the best-fit metallicity model.
 - `dispmodel` is the best-fit metallicity dispersion model.

This type is implemented as a subtype of `Distributions.Sampleable{Multivariate, Continuous}` to enable sampling from an estimate of the likelihood / posterior distribution constructed from the `invH`. You can obtain `N::Integer` samples from the distribution with `rand(R, N)` where `R` is an instance of this type. This will return a size `(length(μ)+2) x N` matrix.
"""
# Result type to return out of Optim.jl BFGS fits
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
                x::Union{AbstractVector{T}, DenseMatrix{T}}) where T <: Real
    
    Base.require_one_based_indexing(x) 
    Zmodel, dispmodel = result.Zmodel, result.dispmodel
    μ = result.μ
    Nbins = length(μ) - nparams(Zmodel) - nparams(dispmodel) # Number of SFR parameters
    dist = MvNormal(Optim.minimizer(result.result), result.invH)
    _rand!(rng, dist, x)
    # Perform variable transformations, first for SFR parameters
    for i in axes(x,1)[begin:Nbins]
        for j in axes(x,2)
            x[i,j] = exp(x[i,j])
        end
    end
    # Now perform variable transformations for metallicity and dispersion models
    tf = (transforms(Zmodel)..., transforms(dispmodel)...)
    free = (free_params(Zmodel)..., free_params(dispmodel)...)
    for i in axes(x,1)[Nbins+1:end]
        tfi = tf[i - Nbins]
        freei = free[i - Nbins] # true if variable is free, false if fixed
        if freei # Variable is free, -- transform samples if necessary
            if tfi == 1
                for j in axes(x,2)
                    x[i,j] = exp(x[i,j])
                end
            elseif tfi == -1
                for j in axes(x,2)
                    x[i,j] = -exp(x[i,j])
                end
                # elseif tfi == 0
                #     continue
            end
        else # Variable is fixed -- overwrite samples with μi
            μi = μ[i]
            for j in axes(x,2)
                x[i,j] = μ[i]
            end
        end
    end
    return x
end
