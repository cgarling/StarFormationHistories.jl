# Methods and utilities for fitting star formation histories
# Currently not sure if I should require data and composite model > 0 or != 0 for loglikelihood and ∇loglikelihood; something to look at. 
"""

Log(likelihood) given by Equation 10 in Dolphin 2002. 
"""
@inline function loglikelihood(composite::AbstractMatrix{<:Number}, data::AbstractMatrix{<:Number})
    T = promote_type(eltype(composite), eltype(data))
    @assert axes(composite) == axes(data) 
    @assert ndims(composite) == 2
    result = zero(T) 
    @turbo for j in axes(composite, 2)  # LoopVectorization.@turbo gives 4x speedup here
        for i in axes(composite, 1)       
            @inbounds ci = composite[i,j]
            @inbounds ni = data[i,j]
            # log_arg = ni / ci
            # (log_arg == Inf) | (log_arg < zero(T)) ? (return -Inf) : (result += ni - ci - ni * log(log_arg))
            # result += ifelse( (ci != zero(T)) & (ni != zero(T)), ni - ci - ni * log(ni / ci), zero(T))
            # result += ifelse( (ci > zero(T)) & (ni > zero(T)), ni - ci - ni * log(ni / ci), zero(T))
            result += (ci > zero(T)) & (ni > zero(T)) ? ni - ci - ni * log(ni / ci) : zero(T)
            # result += ifelse( (mi == z0) & (ni != z0), ni - mi, z0)
            # result += ifelse( (mi != z0) & (ni == z0), ni - mi - ni * log(ni / one(T)), z0)
        end
    end
    return result
end
function loglikelihood(coeffs::AbstractVector{<:Number}, models::Vector{T}, data::AbstractMatrix{<:Number}) where T <: AbstractMatrix{<:Number}
    @assert axes(coeffs) == axes(models)
    S = promote_type(eltype(coeffs), eltype(eltype(models)), eltype(data))
    composite = sum( coeffs .* models )
    return loglikelihood(composite, data)
end

"""

Gradient of [`SFH.loglikelihood`](@ref) with respect to the coefficient; Equation 21 in Dolphin 2002. 
"""
@inline function ∇loglikelihood(model::AbstractMatrix{<:Number}, composite::AbstractMatrix{<:Number}, data::AbstractMatrix{<:Number})
    T = promote_type(eltype(model), eltype(composite), eltype(data))
    @assert axes(model) == axes(composite) == axes(data)
    @assert ndims(model) == 2
    result = zero(T)
    @turbo for j in axes(model, 2)  # ~4x speedup from LoopVectorization.@turbo here
        for i in axes(model, 1)
            @inbounds ci = composite[i,j]
            @inbounds mi = model[i,j]
            @inbounds ni = data[i,j]
            # result += ifelse( ci > zero(T), -mi * (one(T) - ni/ci), zero(T))
            # Returning NaN is required for Optim.jl but not for SPGBox.jl
            result += ifelse( ci > zero(T), -mi * (one(T) - ni/ci), zero(T)) # NaN) 
        end
    end
    return result
end
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

Function to simultaneously compute the loglikelihood and its gradient for one input `model`; see also fg! below, which calculates gradients with respect to multiple models.
"""
function fg(model::AbstractMatrix{<:Number}, composite::AbstractMatrix{<:Number}, data::AbstractMatrix{<:Number})
    T = promote_type(eltype(model), eltype(composite), eltype(data))
    @assert axes(model) == axes(data) == axes(composite)
    @assert ndims(model) == 2
    logL = zero(T) 
    ∇logL = zero(T) 
    @turbo for j in axes(model, 2)   # ~3x speedup from LoopVectorization.@turbo
        for i in axes(model, 1)
            @inbounds ci = composite[i,j]
            @inbounds mi = model[i,j]
            @inbounds ni = data[i,j]
            cond1 = ci > zero(T)
            logL += (cond1 & (ni > zero(T))) ? ni - ci - ni * log(ni / ci) : zero(T)
            ∇logL += cond1 ? -mi * (one(T) - ni/ci) : zero(T)
        end
    end
    return logL, ∇logL
end
"""

Light wrapper for `SFH.fg` that computes loglikelihood and gradient simultaneously; this version is set up for use with Optim.jl. See documentation [here](https://julianlsolvers.github.io/Optim.jl/stable/#user/tipsandtricks/). 
"""
@inline function fg!(F, G, coeffs::AbstractVector{<:Number}, models::AbstractVector{T}, data::AbstractMatrix{<:Number}, composite::AbstractMatrix{<:Number}) where T <: AbstractMatrix{<:Number}
    @assert axes(coeffs) == axes(models)
    @assert axes(data) == axes(composite)
    S = promote_type(eltype(coeffs), eltype(eltype(models)), eltype(eltype(data)), eltype(composite))
    # Fill the composite array with the equivalent of sum( coeffs .* models )
    fill!(composite, zero(eltype(composite))) # composite .= zero(eltype(composite))
    for k in axes(coeffs,1) # @turbo doesn't help with this loop 
        @inbounds ck = coeffs[k]
        @inbounds model = models[k]
        for j in axes(composite,2)
            for i in axes(composite,1) # Putting @turbo here doesn't really help.
                @inbounds composite[i,j] += model[i,j] * ck
            end
        end
    end
    if (F != nothing) & (G != nothing) # Optim.optimize wants objective and gradient
        @assert axes(G) == axes(models)
        # Fill the gradient array
        for i in axes(models,1) # Threads.@threads only helps here if single computation is > 10 ms
            @inbounds G[i] = -∇loglikelihood(models[i], composite, data)
        end
        return -loglikelihood(composite, data) # Return the negative loglikelihood
    elseif G != nothing # Optim.optimize wants only gradient (Does this ever happen?)
        @assert axes(G) == axes(models)
        # Fill the gradient array
        for i in axes(models,1)
            @inbounds G[i] = -∇loglikelihood(models[i], composite, data)
        end
    elseif F != nothing # Optim.optimize wants only objective
        return -loglikelihood(composite, data) # Return the negative loglikelihood
    end
end

"""

# Notes
 - It can be helpful to normalize your `models` to contain realistic total stellar masses; then the fit coefficients can be low and have a tighter dynamic range which can help with the optimization.
 - We recommend that the initial coefficients vector `x0` be set for constant star formation rate. 
"""
# function fit_templates(models::Vector{T}, data::AbstractMatrix{<:Number}; composite=similar(first(models)), x0=ones(S,length(models))) where {S <: Number, T <: AbstractMatrix{S}}
#     # return Optim.optimize(Optim.only_fg!( (F,G,x)->fg!(F,G,x,models,data,composite) ), x0, Optim.LBFGS())
#     return Optim.optimize(Optim.only_fg!( (F,G,x)->fg!(F,G,x,models,data,composite) ),
#                           zeros(S,length(models)), fill(convert(S,Inf),length(models)), # Bounds constraints
#                           x0, Optim.Fminbox(Optim.LBFGS()), # ; alphaguess=LineSearches.InitialStatic(1.0, false),
#                                                          # linesearch=LineSearches.MoreThuente())), # ; alphamin=0.01,
#                                                          # alphamax=Inf))))#,
#                                                          # ; linesearch=LineSearches.HagerZhang())),
#                           Optim.Options(f_tol=1e-5))
# end
# function fit_templates(models::Vector{T}, data::AbstractMatrix{<:Number}; composite=similar(first(models)), x0=ones(S,length(models))) where {S <: Number, T <: AbstractMatrix{S}}
#     return Optim.optimize(Optim.only_fg!( (F,G,x)->SFH.fg!(F,G,x,models,data,composite) ), x0, Optim.LBFGS())
# end
function fit_templates(models::AbstractVector{T}, data::AbstractMatrix{<:Number}; composite=Matrix{Float64}(undef,size(data)), x0=ones(length(models)), eps=1e-5) where {S <: Number, T <: AbstractMatrix{S}}
    return SPGBox.spgbox((g,x)->SFH.fg!(true,g,x,models,data,composite), x0; lower=zeros(length(models)), upper=fill(Inf,length(models)), eps=eps)
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
