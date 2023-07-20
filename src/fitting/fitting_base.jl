# Contains basic likelihood and gradient functions for SFH analysis

"""
     composite!(composite::AbstractMatrix{<:Number}, coeffs::AbstractVector{<:Number}, models::AbstractVector{T}) where T <: AbstractMatrix{<:Number}

Updates the `composite` matrix in place with the linear combination of `sum( coeffs .* models )`; this is equation 1 in Dolphin 2002, ``m_i = \\sum_j \\, r_j \\, c_{i,j}``.

# Examples
```julia
julia> C = zeros(5,5);
julia> models = [rand(size(C)...) for i in 1:5];
julia> coeffs = rand(length(models));
julia> composite!(C, coeffs, models);
julia> C ≈ sum( coeffs .* models)
true
```
"""
@inline function composite!(composite::AbstractMatrix{<:Number}, coeffs::AbstractVector{<:Number}, models::AbstractVector{T}) where T <: AbstractMatrix{<:Number}
    @assert axes(coeffs) == axes(models)
    fill!(composite, zero(eltype(composite))) # Zero-out array
    for k in axes(coeffs,1) # @turbo doesn't help with this loop 
        @inbounds ck = coeffs[k]
        @inbounds model = models[k]
        @simd for idx in eachindex(composite, model)
            # @inbounds composite[idx] += model[idx] * ck 
            @inbounds composite[idx] = muladd(model[idx], ck, composite[idx])
        end
    end
end


"""
    loglikelihood(composite::AbstractMatrix{<:Number}, data::AbstractMatrix{<:Number})

Returns the logarithm of the Poisson likelihood ratio given by equation 10 in Dolphin 2002,

```math
\\text{ln} \\, \\mathscr{L} = \\sum_i -m_i + n_i \\times \\left( 1 - \\text{ln} \\, \\left( \\frac{n_i}{m_i} \\right) \\right)
```

with `composite` being the complex Hess model diagram ``m_i`` (see [`StarFormationHistories.composite!`](@ref)) and `data` being the observed Hess diagram ``n_i``.

# Performance Notes
 - ~18.57 μs for `composite=Matrix{Float64}(undef,99,99)` and `data=similar(composite)`.
 - ~20 μs for `composite=Matrix{Float64}(undef,99,99)` and `data=Matrix{Int64}(undef,99,99)`.
 - ~9.3 μs for `composite=Matrix{Float32}(undef,99,99)` and `data=similar(composite)`.
 - ~9.6 μs for `composite=Matrix{Float32}(undef,99,99)` and `data=Matrix{Int64}(undef,99,99)`.
"""
@inline function loglikelihood(composite::AbstractMatrix{<:Number}, data::AbstractMatrix{<:Number})
    T = promote_type(eltype(composite), eltype(data))
    @assert axes(composite) == axes(data) 
    @assert ndims(composite) == 2
    result = zero(T) 
    @turbo thread=false for idx in eachindex(composite, data) # LoopVectorization.@turbo gives 2x speedup here
        # Setting eps() as minimum of composite greatly improves stability of convergence
        @inbounds ci = max( composite[idx], eps(T) ) 
        @inbounds ni = data[idx]
        # result += (ci > zero(T)) & (ni > zero(T)) ? ni - ci - ni * log(ni / ci) : zero(T)
        # result += ni > zero(T) ? ni - ci - ni * log(ni / ci) : zero(T)
        result += ifelse( ni > zero(T), ni - ci - ni * log(ni / ci), zero(T) )
    end
    # Penalizing result==0 here improves stability of convergence
    result != zero(T) ? (return result) : (return -typemax(T))
end
"""
    loglikelihood(coeffs::AbstractVector{<:Number}, models::AbstractVector{T}, data::AbstractMatrix{<:Number}) where T <: AbstractMatrix{<:Number}

Returns the logarithm of the Poisson likelihood ratio, but constructs the complex Hess diagram model as `sum(coeffs .* models)` rather than taking `composite` directly as an argument. 
"""
function loglikelihood(coeffs::AbstractVector{<:Number}, models::AbstractVector{T}, data::AbstractMatrix{<:Number}) where T <: AbstractMatrix{<:Number}
    @assert axes(coeffs) == axes(models)
    S = promote_type(eltype(coeffs), eltype(eltype(models)), eltype(data))
    composite = Matrix{S}(undef,size(data)) # composite = sum( coeffs .* models )
    composite!(composite, coeffs, models) # Fill the composite array
    return loglikelihood(composite, data)
end

"""
    ∇loglikelihood(model::AbstractMatrix{<:Number}, composite::AbstractMatrix{<:Number}, data::AbstractMatrix{<:Number})

Returns the partial derivative of the logarithm of the Poisson likelihood ratio ([`StarFormationHistories.loglikelihood`](@ref)) with respect to the coefficient ``r_j`` on the provided `model`. If the complex Hess diagram model is ``m_i = \\sum_j \\, r_j \\, c_{i,j}``, then `model` is ``c_{i,j}``, and this function computes the partial derivative of ``\\text{log} \\, \\mathscr{L}`` with respect to the coefficient ``r_j``. This is given by equation 21 in Dolphin 2002,

```math
\\frac{\\partial \\, \\text{log} \\, \\mathscr{L}}{\\partial \\, r_j} = \\sum_i c_{i,j} \\left( \\frac{n_i}{m_i} - 1 \\right)
```

where ``n_i`` is bin ``i`` of the observed Hess diagram `data`. 

# Performance Notes
 - ~4.1 μs for model, composite, data all being `Matrix{Float64}(undef,99,99)`.
 - ~1.3 μs for model, composite, data all being `Matrix{Float32}(undef,99,99)`. 
"""
@inline function ∇loglikelihood(model::AbstractMatrix{<:Number}, composite::AbstractMatrix{<:Number}, data::AbstractMatrix{<:Number})
    T = promote_type(eltype(model), eltype(composite), eltype(data))
    @assert axes(model) == axes(composite) == axes(data)
    @assert ndims(model) == 2
    result = zero(T)
    # ~4x speedup from LoopVectorization.@turbo here
    @turbo thread=false for idx in eachindex(model, composite, data) 
        # Setting eps() as minimum of composite greatly improves stability of convergence
        # and prevents divide by zero errors.
        @inbounds ci = max( composite[idx], eps(T) )
        @inbounds mi = model[idx]
        @inbounds ni = data[idx]
        result += ifelse( ni > zero(T), -mi * (one(T) - ni/ci), zero(T) )
    end
    return result
end
"""
    ∇loglikelihood(models::AbstractVector{T}, composite::AbstractMatrix{<:Number}, data::AbstractMatrix{<:Number}) where T <: AbstractMatrix{<:Number}

Computes the gradient of the logarithm of the Poisson likelihood ratio with respect to the coefficients by calling the single-model `∇loglikelihood` for every model in `models`. 
"""
function ∇loglikelihood(models::AbstractVector{T}, composite::AbstractMatrix{<:Number}, data::AbstractMatrix{<:Number}) where T <: AbstractMatrix{<:Number}
    @assert axes(composite) == axes(data)
    return [ ∇loglikelihood(i, composite, data) for i in models ]
end
function ∇loglikelihood(coeffs::AbstractVector{<:Number}, models::AbstractVector{T}, data::AbstractMatrix{<:Number}) where T <: AbstractMatrix{<:Number}
    @assert axes(coeffs) == axes(models)
    S = promote_type(eltype(coeffs), eltype(eltype(models)), eltype(data))
    composite = Matrix{S}(undef,size(data)) # composite = sum( coeffs .* models )
    composite!(composite, coeffs, models) # Fill the composite array
    return ∇loglikelihood(models, composite, data) # Call to above function.
end

"""
     ∇loglikelihood!(G::AbstractVector, composite::AbstractMatrix{<:Number}, models::AbstractVector{S}, data::AbstractMatrix{<:Number}) where S <: AbstractMatrix{<:Number}

Efficiently computes the gradient of [`StarFormationHistories.loglikelihood`](@ref) with respect to all coefficients by updating `G` with the gradient. This will overwrite `composite` with the result of `1 .- (data ./ composite)` so it shouldn't be reused after being passed to this function. 

# Arguments
 - `G::AbstractVector` is the vector that  will be mutated in-place with the computed gradient values.
 - `models::AbstractVector{<:AbstractMatrix{<:Number}}` is the vector of matrices giving the model Hess diagrams.
 - `composite::AbstractMatrix{<:Number}` is a matrix that contains the composite model `sum(coeffs .* models)`.
 - `data::AbstractMatrix{<:Number}` contains the observed Hess diagram that is being fit.
"""
function ∇loglikelihood!(G::AbstractVector, composite::AbstractMatrix{<:Number}, models::AbstractVector{S}, data::AbstractMatrix{<:Number}) where S <: AbstractMatrix{<:Number}
    T = eltype(G) 
    @assert axes(composite) == axes(data) 
    @assert axes(G,1) == axes(models,1)
    # Build the (1 .- data ./ composite) matrix which is all we need for this method
    @turbo for idx in eachindex(composite, data)
        # Setting eps() as minimum of composite greatly improves stability of convergence
        # and prevents divide by zero errors.
        @inbounds ci = max( composite[idx], eps(T) )
        @inbounds ni = data[idx]
        @inbounds composite[idx] = one(T) - ni/ci
    end
    for k in eachindex(G, models)
        @inbounds model = models[k]
        @assert axes(model) == axes(data) == axes(composite)
        result = zero(T)
        @turbo thread=false for idx in eachindex(model, data, composite)
            @inbounds mi = model[idx]
            @inbounds ni = data[idx]
            @inbounds nici = composite[idx]
            result += ifelse( ni > zero(T), -mi * nici, zero(T) )
        end
        @inbounds G[k] = result
    end
end
