# MCMC sampling of SFH and distance for fixed input linear age-metallicity relation
# and Gaussian spread σ

# struct MCMCFixedLAMRDistance{A <: AbstractVector{<:AbstractMatrix{<:Number}},
#                              B <: AbstractVector{<:AbstractMatrix{<:Number}},
#                              C <: AbstractVector{<:Number},
#                              D <: AbstractVector{<:Number},
#                              E <: Distribution{Univariate, Continuous},
#                              F,
#                              G <: AbstractVector{<:Number},
#                              H <: AbstractVector{<:Number},
#                              I <: Number}
#     models::A
#     composites::B # Vector of matrices, one per thread
#     xcolors::C
#     ymags::D
#     distance_prior::E
#     edges::F
#     logAge::G
#     metallicities::H
#     α::I
#     β::I
#     σ::I
# end
# # edges is a 2-tuple; first element is x-bins, second element is y-bins where
# # y-bins are the absolute magnitude bins used to construct the models from isochrones.

# function MCMCFixedLAMRDistance(models::A,
#                                xcolors::B,
#                                ymags::C,
#                                distance_prior::D,
#                                edges,
#                                logAge::AbstractVector{<:Number},
#                                metallicities::AbstractVector{<:Number},
#                                α::Number,
#                                β::Number,
#                                σ::Number) where {A <: AbstractVector{<:AbstractMatrix{<:Number}},
#                                                  B <: AbstractVector{<:Number},
#                                                  C <: AbstractVector{<:Number},
#                                                  D <: Distribution{Univariate, Continuous}}
#     V = promote_type(eltype(eltype(A)), eltype(B), eltype(C), typeof(α), typeof(β), typeof(σ))
#     @assert length(xcolors) == length(ymags)
#     @assert length(models) == length(logAge) == length(metallicities)
#     @assert σ > 0
#     return MCMCFixedLAMRDistance(models, [Matrix{V}(undef, size(first(models))) for i in 1:Threads.nthreads()], xcolors, ymags, distance_prior, edges, logAge, metallicities, convert.(V,(α, β, σ))...)
# end

struct MCMCFixedLAMRDistance{A <: AbstractVector{<:AbstractMatrix{<:Number}},
                             B <: AbstractVector{<:AbstractMatrix{<:Number}},
                             C <: AbstractVector{<:Number},
                             D <: AbstractVector{<:Number},
                             E <: Distribution{Univariate, Continuous},
                             F,
                             G <: AbstractVector{<:Number},
                             H,
                             I <: AbstractVector{<:AbstractVector{<:Number}}}
    models::A
    composites::B # Vector of matrices, one per thread
    xcolors::C
    ymags::D
    distance_prior::E
    edges::F
    relweights::G # Pre-calculate relative weights since the LAMR is fixed
    idxlogAge::H
    coeffs::I
end
# edges is a 2-tuple; first element is x-bins, second element is y-bins where
# y-bins are the absolute magnitude bins used to construct the models from isochrones.

function MCMCFixedLAMRDistance(models::A,
                               xcolors::B,
                               ymags::C,
                               distance_prior::D,
                               edges,
                               logAge::AbstractVector{<:Number},
                               metallicities::AbstractVector{<:Number},
                               T_max::Number,
                               α::Number,
                               β::Number,
                               σ::Number) where {A <: AbstractVector{<:AbstractMatrix{<:Number}},
                                                 B <: AbstractVector{<:Number},
                                                 C <: AbstractVector{<:Number},
                                                 D <: Distribution{Univariate, Continuous}}
    V = promote_type(eltype(eltype(A)), eltype(B), eltype(C))
    @assert length(xcolors) == length(ymags)
    @assert length(logAge) == length(metallicities)
    @assert σ > 0
    # Pre-calculate relative per-model weights since LAMR is fixed
    relweights = calculate_coeffs_mdf( ones(length(unique(logAge))), logAge, metallicities, T_max, α, β, σ)
    # Save out the index masks for each unique entry in logAge so we can
    # construct the full `coeffs` vector when evaluating the likelihood
    idxlogAge = [logAge .== i for i in unique(logAge)]
    return MCMCFixedLAMRDistance(models, [Matrix{V}(undef, size(first(models))) for i in 1:Threads.nthreads()], xcolors, ymags, distance_prior, edges, relweights, idxlogAge, [Vector{V}(undef, length(models)) for i in 1:Threads.nthreads()])
end

# This model will return loglikelihood
LogDensityProblems.capabilities(::Type{<:MCMCFixedLAMRDistance}) = LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.dimension(problem::MCMCFixedLAMRDistance) = length(problem.models) + 1

# Make the type callable for the loglikelihood
function (problem::MCMCFixedLAMRDistance{A,B,C,D,E,F,G,H})(θ) where {A,B,C,D,E,F,G,H}
    T = promote_type( eltype(eltype(A)), eltype(eltype(B)), eltype(C), eltype(D), eltype(G))
    new_distance = first(θ)
    for coeff in θ # If any coefficients are negative, return -Inf
        if coeff < zero(T)
            return typemin(T)
        end
    end
    # Construct new Hess diagram for the data given the new distance (in kpc)
    data = bin_cmd(problem.xcolors, problem.ymags, edges=(problem.edges[1], problem.edges[2] .+ distance_modulus(new_distance * 1000)))

    # This should be valid for vectors with non 1-based indexing
    tid = first(axes(problem.composites))[Threads.threadid()] 

    # Calculate per-model coefficients based on the relative coefficient vector
    # and the provided θ
    coeffs = problem.coeffs[tid] # coeffs = Vector{T}(undef, length(problem.models))
    for (i, idxs) in enumerate(problem.idxlogAge)
        @. coeffs[idxs] = problem.relweights[idxs] * θ[i+1]
    end

    composite!( problem.composites[tid], coeffs, problem.models )
    return loglikelihood(problem.composites[tid], data.weights) + convert(T, logpdf(problem.distance_prior,new_distance))
end
LogDensityProblems.logdensity(problem::MCMCFixedLAMRDistance, θ) = problem(θ)

## Convenience functions


# import Distributions: Distribution, Univariate, Continuous, Normal, logpdf, Poisson
# import StarFormationHistories: calculate_coeffs_mdf, distance_modulus, bin_cmd, composite!, loglikelihood

# logAge = repeat(range(6.6, 10.2, 10); inner=10)
# metallicities = repeat(range(-2.0,0.0,10),10)
# relative_weights = calculate_coeffs_mdf(ones(10), logAge, metallicities, -0.08, -0.5, 0.2)
# # Test the the sum of relative weights for each unique logAge are 1
# all( [sum(relative_weights[logAge .== i]) for i in logAge] .≈ 1 )
# idxlogAge = [logAge .== i for i in unique(logAge)]

# inst = MCMCFixedLAMRDistance([rand(20,20) for i in 1:100], rand(100), rand(100), Normal(0.0,1.0), (0:0.05:1, 0:0.05:1), repeat(range(6.6, 10.2, 10); inner=10), repeat(range(-2.0,0.0,10),10), -0.08, -0.5, 0.2)
# inst(vcat(0,ones(10).+1))
