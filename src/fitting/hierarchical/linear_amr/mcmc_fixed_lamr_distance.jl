struct MCMCFixedLAMRDistance{A <: AbstractVector{<:AbstractMatrix{<:Number}},
                             B <: AbstractVector{<:AbstractMatrix{<:Number}},
                             C <: AbstractVector{<:Number},
                             D <: AbstractVector{<:Number},
                             E <: Distribution{Univariate, Continuous},
                             F,
                             G <: AbstractVector{<:Number},
                             H <: AbstractVector{<:Number},
                             I <: Number}
    models::A
    composites::B # Vector of matrices, one per thread
    xcolors::C
    ymags::D
    distance_prior::E
    edges::F
    logAge::G
    metallicities::H
    α::I
    β::I
    σ::I
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
                               α::Number,
                               β::Number,
                               σ::Number) where {A <: AbstractVector{<:AbstractMatrix{<:Number}},
                                                 B <: AbstractVector{<:Number},
                                                 C <: AbstractVector{<:Number},
                                                 D <: Distribution{Univariate, Continuous}}
    V = promote_type(eltype(eltype(A)), eltype(B), eltype(C), typeof(α), typeof(β), typeof(σ))
    @assert length(xcolors) == length(ymags)
    @assert length(models) == length(logAge) == length(metallicities)
    @assert σ > 0
    return MCMCFixedLAMRDistance(models, [Matrix{V}(undef, size(first(models))) for i in 1:Threads.nthreads()], xcolors, ymags, distance_prior, edges, logAge, metallicities, convert.(V,(α, β, σ))...)
end

# This model will return loglikelihood
LogDensityProblems.capabilities(::Type{<:MCMCFixedLAMRDistance}) = LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.dimension(problem::MCMCFixedLAMRDistance) = length(problem.models) + 1




# Make the type callable for the loglikelihood
function (problem::MCMCFixedLAMRDistance{A,B,C,D,E,F,G,H,I})(θ)
    T = I
    new_distance = first(θ)
    for coeff in θ # If any coefficients are negative, return -Inf
        if coeff < zero(T)
            return typemin(T)
        end
    end
    # Construct new Hess diagram for the data given the new distance (in kpc)
    data = bin_cmd(problem.xcolors, problem.ymags, edges=(problem.edges[1], problem.edges[2] .+ distance_modulus(new_distance * 1000)))

    # Calculate per-model coefficients based on the relative coefficient vector
    # and the provided θ

    # This should be valid for vectors with non 1-based indexing
    idx = first(axes(problem.composites))[Threads.threadid()] 
    composite!( problem.composites[idx], view(θ,2:lastindex(θ)), problem.models )
    return loglikelihood(problem.composites[idx], data.weights) + convert(T, logpdf(problem.distance_prior,new_distance))
end
LogDensityProblems.logdensity(problem::MCMCModelDistance, θ) = problem(θ)
