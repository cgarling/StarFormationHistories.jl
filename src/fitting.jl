# Methods and utilities for fitting star formation histories

"""

Log(likelihood) given by Equation 10 in Dolphin 2002. 
"""
# function loglikelihood(coeff::Number, model::AbstractMatrix{<:Number}, data::AbstractMatrix{<:Number})
#     T = promote_type(typeof(coeff), eltype(model), eltype(data))
#     @assert axes(model) == axes(data) 
#     @assert ndims(model) == 2
#     result = zero(T) 
#     for j in axes(model, 2)  # LoopVectorization.@turbo gives 4x speedup here
#         @simd for i in axes(model, 1) 
#             @inbounds mi = coeff * model[i,j]
#             @inbounds ni = data[i,j]
#             result += ifelse( (mi != zero(T)) & (ni != zero(T)), ni - mi - ni * log(ni / mi), zero(T))
#             # result += ifelse( (mi == z0) & (ni != z0), ni - mi, z0)
#             # result += ifelse( (mi != z0) & (ni == z0), ni - mi - ni * log(ni / one(T)), z0)
#         end
#     end
#     return result
# end
@inline function loglikelihood(composite::AbstractMatrix{<:Number}, data::AbstractMatrix{<:Number})
    T = promote_type(eltype(composite), eltype(data))
    @assert axes(composite) == axes(data) 
    @assert ndims(composite) == 2
    result = zero(T) 
    @turbo for j in axes(composite, 2)  # LoopVectorization.@turbo gives 4x speedup here
        for i in axes(composite, 1) 
            @inbounds ci = composite[i,j]
            @inbounds ni = data[i,j]
            result += ifelse( (ci != zero(T)) & (ni != zero(T)), ni - ci - ni * log(ni / ci), zero(T))
            # result += ifelse( (mi == z0) & (ni != z0), ni - mi, z0)
            # result += ifelse( (mi != z0) & (ni == z0), ni - mi - ni * log(ni / one(T)), z0)
        end
    end
    return result
end

# function loglikelihood(coeffs::AbstractVector{<:Number}, models::Vector{T}, data::AbstractMatrix{<:Number}) where T <: AbstractMatrix{<:Number}
#     @assert axes(coeffs) == axes(models)
#     # return [loglikelihood(coeffs[i], models[i], data) for i in axes(coeffs,1)]
#     return sum( loglikelihood(coeffs[i], models[i], data) for i in axes(coeffs,1) )
# end
function loglikelihood(coeffs::AbstractVector{<:Number}, models::Vector{T}, data::AbstractMatrix{<:Number}) where T <: AbstractMatrix{<:Number}
    @assert axes(coeffs) == axes(models)
    S = promote_type(eltype(coeffs), eltype(eltype(models)), eltype(data))
    composite = sum( coeffs .* models )
    return loglikelihood(composite, data)
end

"""

Gradient of [`SFH.loglikelihood`](@ref) with respect to the coefficient; Equation 21 in Dolphin 2002. 
"""
# function ∇loglikelihood(coeff::Number, model::AbstractMatrix{<:Number}, data::AbstractMatrix{<:Number})
#     T = promote_type(typeof(coeff), eltype(model), eltype(data))
#     @assert axes(model) == axes(data) 
#     @assert ndims(model) == 2
#     result = zero(T) 
#     for j in axes(model, 2)  # LoopVectorization.@turbo gives 4x speedup here
#         @simd for i in axes(model, 1) 
#             @inbounds mi = coeff * model[i,j]
#             @inbounds ni = data[i,j]
#             result += ifelse( mi != zero(T), -mi * (one(T) - ni/mi), zero(T))
#         end
#     end
#     return result
# end
@inline function ∇loglikelihood(model::AbstractMatrix{<:Number}, composite::AbstractMatrix{<:Number}, data::AbstractMatrix{<:Number})
    T = promote_type(eltype(model), eltype(composite), eltype(data))
    @assert axes(model) == axes(data) == axes(composite)
    @assert ndims(model) == 2
    result = zero(T) 
    @turbo for j in axes(model, 2)  # LoopVectorization.@turbo gives 4x speedup here
        for i in axes(model, 1)
            @inbounds ci = composite[i,j]
            @inbounds mi = model[i,j]
            @inbounds ni = data[i,j]
            result += ifelse( ci != zero(T), -mi * (one(T) - ni/ci), zero(T))
        end
    end
    return result
end

# function ∇loglikelihood(coeffs::AbstractVector{<:Number}, models::Vector{T}, data::AbstractMatrix{<:Number}) where T <: AbstractMatrix{<:Number}
#     @assert axes(coeffs) == axes(models)
#     # return [∇loglikelihood(coeffs[i], models[i], data) for i in axes(coeffs,1)]
#     return sum( ∇loglikelihood(coeffs[i], models[i], data) for i in axes(coeffs,1) )
# end
function ∇loglikelihood(models::AbstractVector{T}, composite::AbstractMatrix{<:Number}, data::AbstractMatrix{<:Number}) where T <: AbstractMatrix{<:Number}
    @assert axes(composite) == axes(data)
    return [ ∇loglikelihood(i, composite, data) for i in models ]
end
function ∇loglikelihood(coeffs::AbstractVector{<:Number}, models::Vector{T}, data::AbstractMatrix{<:Number}) where T <: AbstractMatrix{<:Number}
    @assert axes(coeffs) == axes(models)
    composite = sum( coeffs .* models )
    return ∇loglikelihood(models, composite, data) # Call to above function.
end

"""

Function to simultaneously compute the loglikelihood and its gradient for Optim.jl; see fg! below.
"""
# function fg(coeff::Number, model::AbstractMatrix{<:Number}, data::AbstractMatrix{<:Number})
#     T = promote_type(typeof(coeff), eltype(model), eltype(data))
#     @assert axes(model) == axes(data) 
#     @assert ndims(model) == 2
#     logL = zero(T) 
#     ∇logL = zero(T) 
#     @turbo for j in axes(model, 2)  # LoopVectorization.@turbo gives 4x speedup here
#         for i in axes(model, 1) 
#             @inbounds mi = coeff * model[i,j]
#             @inbounds ni = data[i,j]
#             logL += ifelse( (mi != zero(T)) & (ni != zero(T)), ni - mi - ni * log(ni / mi), zero(T))
#             ∇logL += ifelse( mi != zero(T), -mi * (one(T) - ni/mi), zero(T))
#         end
#     end
#     return logL, ∇logL
# end
function fg(model::AbstractMatrix{<:Number}, composite::AbstractMatrix{<:Number}, data::AbstractMatrix{<:Number})
    T = promote_type(eltype(model), eltype(composite), eltype(data))
    @assert axes(model) == axes(data) == axes(composite)
    @assert ndims(model) == 2
    logL = zero(T) 
    ∇logL = zero(T) 
    for j in axes(model, 2)  # LoopVectorization.@turbo gives 4x speedup here
        @simd for i in axes(model, 1)
            @inbounds ci = composite[i,j]
            @inbounds mi = model[i,j]
            @inbounds ni = data[i,j]
            logL += ifelse( (ci != zero(T)) & (ni != zero(T)), ni - ci - ni * log(ni / ci), zero(T))
            ∇logL += ifelse( ci != zero(T), -mi * (one(T) - ni/ci), zero(T))
        end
    end
    return logL, ∇logL
end
"""

Light wrapper for `SFH.fg` that computes loglikelihood and gradient simultaneously; this version is set up for use with Optim.jl. See documentation [here](https://julianlsolvers.github.io/Optim.jl/stable/#user/tipsandtricks/). 
"""
# function fg!(F, G, coeffs::AbstractVector{<:Number}, models::Vector{T}, data::AbstractMatrix{<:Number}) where T <: AbstractMatrix{<:Number}
#     @assert axes(G) == axes(coeffs) == axes(models)
#     S = promote_type(eltype(coeffs), eltype(eltype(models)), eltype(eltype(data)))
#     if (F != nothing) & (G != nothing)
#         Fsum = zero(S)
#         for i in axes(coeffs,1)
#             logL, ∇logL = fg(coeffs[i], models[i], data)
#             Fsum += logL
#             G[i] = ∇logL
#         end
#         return Fsum
#     elseif G != nothing
#         for i in axes(coeffs,1)
#             G[i] = ∇loglikelihood(coeffs[i], models[i], data)
#         end
#     elseif F != nothing
#         return loglikelihood(coeffs, models, data)
#     end
# end
# function fg!(F, G, coeffs::AbstractVector{<:Number}, models::Vector{T}, data::AbstractMatrix{<:Number}) where T <: AbstractMatrix{<:Number}
#     @assert axes(G) == axes(coeffs) == axes(models)
#     S = promote_type(eltype(coeffs), eltype(eltype(models)), eltype(eltype(data)))
#     if (F != nothing) & (G != nothing)
#         Fsum = zero(S)
#         for i in axes(coeffs,1)
#             logL, ∇logL = fg(coeffs[i], models[i], data)
#             Fsum -= logL
#             G[i] = -∇logL/coeffs[i]
#         end
#         return Fsum
#     elseif G != nothing
#         for i in axes(coeffs,1)
#             G[i] = -∇loglikelihood(coeffs[i], models[i], data)/coeffs[i]
#         end
#     elseif F != nothing
#         return -loglikelihood(coeffs, models, data)
#     end
# end
@inline function fg!(F, G, coeffs::AbstractVector{<:Number}, models::Vector{T}, data::AbstractMatrix{<:Number}, composite::AbstractMatrix{<:Number}) where T <: AbstractMatrix{<:Number}
    @assert axes(coeffs) == axes(models)
    @assert axes(data) == axes(composite) 
    S = promote_type(eltype(coeffs), eltype(eltype(models)), eltype(eltype(data)), eltype(composite))
    # Fill the composite array with the equivalent of sum( coeffs .* models )
    composite .= zero(eltype(composite))
    for k in axes(coeffs,1) # @turbo doesn't help with this loop 
        @inbounds ck = coeffs[k]
        @inbounds model = models[k]
        for j in axes(composite,2)
            @simd for i in axes(composite,1) # Putting @turbo here doesn't really help.
                @inbounds composite[i,j] += model[i,j] * ck
            end
        end
    end
    if (F != nothing) & (G != nothing) # Optim.optimize wants objective and gradient
        logL = loglikelihood(composite, data)
        for i in axes(models,1)
            @inbounds G[i] = -∇loglikelihood(models[i], composite, data)
        end
        return -logL
    elseif G != nothing # Optim.optimize wants only gradient (Does this ever happen?)
        for i in axes(models,1)
            @inbounds G[i] = -∇loglikelihood(models[i], composite, data)
        end
    elseif F != nothing # Optim.optimize wants only objective
        return -loglikelihood(composite, data)
    end
end

function fit_templates(models::Vector{T}, data::AbstractMatrix{<:Number}; composite=similar(first(models)), x0=ones(S,length(models))) where {S <: Number, T <: AbstractMatrix{S}}
    # return Optim.optimize(Optim.only_fg!( (F,G,x)->fg!(F,G,x,models,data,composite) ), x0, Optim.LBFGS())
    return Optim.optimize(Optim.only_fg!( (F,G,x)->fg!(F,G,x,models,data,composite) ),
                          zeros(S,length(models)), fill(convert(S,Inf),length(models)), # Bounds constraints
                          x0, Optim.Fminbox(Optim.LBFGS()), # ; alphaguess=LineSearches.InitialStatic(1.0, false),
                                                         # linesearch=LineSearches.MoreThuente())), # ; alphamin=0.01,
                                                         # alphamax=Inf))))#,
                                                         # ; linesearch=LineSearches.HagerZhang())),
                          Optim.Options(f_tol=1e-5))
end

# M1 = rand(120,100)
# M2 = rand(120, 100)
# N1 = rand.( Poisson.( (250.0 .* M1))) .+ rand.(Poisson.((500.0 .* M2)))
# Optim.optimize(x->-loglikelihood(x,[M1,M2],N1),[1.0,1.0],Optim.LBFGS())
# C1 = similar(M1)
# Optim.optimize(Optim.only_fg!( (F,G,x)->fg!(F,G,x,[M1,M2],N1,C1) ),[1.0,1.0],Optim.LBFGS())
# G=[1.0, 1.0]; coe=[5.0,5.0]; MM=[M1,M2]
# fg!(true,G,coe,MM,N1,C1)
# @benchmark fg!($true,$G,$coe,$MM,$N1,$C1)
