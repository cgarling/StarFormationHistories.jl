# Methods and utilities for fitting star formation histories
# Currently not sure if I should require data and composite model > 0 or != 0 for loglikelihood and ∇loglikelihood; something to look at.
"""
     composite!(composite::AbstractMatrix{<:Number}, coeffs::AbstractVector{<:Number}, models::AbstractVector{T}) where T <: AbstractMatrix{<:Number}

Updates the `composite` matrix in place with the linear combination of `sum( coeffs .* models )`.

# Examples
```julia
julia> C = zeros(5,5);
julia> models = [rand(size(C)...) for i in 1:5];
julia> coeffs = rand(length(models));
julia> SFH.composite!(C, coeffs, models);
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
        @assert axes(model) == axes(composite)
        for j in axes(composite,2)
            @simd for i in axes(composite,1) # Putting @turbo here doesn't really help.
                @inbounds composite[i,j] += model[i,j] * ck 
                # @inbounds composite[i,j] = muladd(model[i,j], ck, composite[i,j])
            end
        end
    end
end


"""

Log(likelihood) given by Equation 10 in Dolphin 2002.

# Performance Notes
 - ~18.57 μs for `composite=Matrix{Float64}(undef,99,99)' and `data=similar(composite)`.
 - ~20 μs for `composite=Matrix{Float64}(undef,99,99)' and `data=Matrix{Int64}(undef,99,99)`.
 - ~9.3 μs for `composite=Matrix{Float32}(undef,99,99)' and `data=similar(composite)`.
 - ~9.6 μs for `composite=Matrix{Float32}(undef,99,99)' and `data=Matrix{Int64}(undef,99,99)`.
"""
@inline function loglikelihood(composite::AbstractMatrix{<:Number}, data::AbstractMatrix{<:Number})
    T = promote_type(eltype(composite), eltype(data))
    @assert axes(composite) == axes(data) 
    @assert ndims(composite) == 2
    result = zero(T) 
    @turbo for j in axes(composite, 2)  # LoopVectorization.@turbo gives 4x speedup here
        for i in axes(composite, 1)       
            # Setting eps() as minimum of composite greatly improves stability of convergence
            @inbounds ci = max( composite[i,j], eps(T) ) 
            @inbounds ni = data[i,j]
            # result += (ci > zero(T)) & (ni > zero(T)) ? ni - ci - ni * log(ni / ci) : zero(T)
            # result += ni > zero(T) ? ni - ci - ni * log(ni / ci) : zero(T)
            result += ifelse( ni > zero(T), ni - ci - ni * log(ni / ci), zero(T) )
        end
    end
    # Penalizing result==0 here improves stability of convergence
    result != zero(T) ? (return result) : (return -typemax(T))
end
function loglikelihood(coeffs::AbstractVector{<:Number}, models::AbstractVector{T}, data::AbstractMatrix{<:Number}) where T <: AbstractMatrix{<:Number}
    @assert axes(coeffs) == axes(models)
    S = promote_type(eltype(coeffs), eltype(eltype(models)), eltype(data))
    composite = Matrix{S}(undef,size(data)) # composite = sum( coeffs .* models )
    composite!(composite, coeffs, models) # Fill the composite array
    return loglikelihood(composite, data)
end

"""

Gradient of [`SFH.loglikelihood`](@ref) with respect to the coefficient; Equation 21 in Dolphin 2002.

# Performance Notes
 - ~4.1 μs for model, composite, data all being Matrix{Float64}(undef,99,99).
 - ~1.3 μs for model, composite, data all being Matrix{Float32}(undef,99,99). 
"""
@inline function ∇loglikelihood(model::AbstractMatrix{<:Number}, composite::AbstractMatrix{<:Number}, data::AbstractMatrix{<:Number})
    T = promote_type(eltype(model), eltype(composite), eltype(data))
    @assert axes(model) == axes(composite) == axes(data)
    @assert ndims(model) == 2
    result = zero(T)
    for j in axes(model, 2)  # ~4x speedup from LoopVectorization.@turbo here
        @turbo for i in axes(model, 1)
            # Setting eps() as minimum of composite greatly improves stability of convergence.
            @inbounds ci = max( composite[i,j], eps(T) )
            @inbounds mi = model[i,j]
            @inbounds ni = data[i,j]
            # Returning NaN is required for Optim.jl but not for SPGBox.jl
            # result += ifelse( (ci > zero(T)) & (ni > zero(T)), -mi * (one(T) - ni/ci), zero(T)) # NaN)
            result += ifelse( ni > zero(T), -mi * (one(T) - ni/ci), zero(T) )
        end
    end
    return result
end
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
    composite!(composite, coeffs, models) 
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

# Write a function here to take in logage array and return x0 (normalized total mass) such that the SFR is constant. Afterward, might want to write another method call that also takes metallicity and metallicity model and tweaks x0 according to some prior metallicity model.
function construct_x0(logage::AbstractVector{T}; normalize_value::Number=one(T)) where T<:Number
    minlog, maxlog = extrema(logage)
    sfr = normalize_value / (exp10(maxlog) - exp10(minlog)) # Average SFR / yr
    unique_logage = sort!(unique(logage))
    num_ages = [count(logage .== la) for la in unique_logage] # number of entries per unique
    dt = [exp10(unique_logage[i+1]) - exp10(unique_logage[i]) for i in 1:length(unique_logage)-1]
    result = similar(logage)
    for i in eachindex(logage, result)
        la = logage[i]
        idx = findfirst( x -> x==la, unique_logage )
        idx = min( length(dt), idx )
        result[i] = sfr * dt[idx] / num_ages[idx]
    end
    return result
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
# function fit_templates(models::AbstractVector{T}, data::AbstractMatrix{<:Number}; composite=Matrix{Float64}(undef,size(data)), x0=ones(length(models)), eps=1e-5, nfevalmax::Integer=1000, nitmax::Integer=100) where {S <: Number, T <: AbstractMatrix{S}}
#     return SPGBox.spgbox((g,x)->fg!(true,g,x,models,data,composite), x0; lower=zeros(length(models)), upper=fill(Inf,length(models)), eps=eps, nfevalmax=nfevalmax, nitmax=nitmax, m=100)
# end
# function fit_templates2(models::AbstractVector{T}, data::AbstractMatrix{<:Number}; composite=Matrix{Float64}(undef,size(data)), x0=ones(length(models))) where {S <: Number, T <: AbstractMatrix{S}}
#     G = similar(x0)
#     fg(x) = (R = SFH.fg!(true,G,x,models,data,composite); return R,G)
#     scipy_opt.fmin_l_bfgs_b(fg, x0; factr=1e-12, bounds=[(0.0,Inf) for i in 1:length(x0)])
#     # scipy_opt.fmin_l_bfgs_b(x->SFH.fg!(true,G,x,models,data,composite), x0; approx_grad=true, factr=1, bounds=[(0.0,Inf) for i in 1:length(x0)], maxfun=20000)
# end
function fit_templates(models::AbstractVector{T}, data::AbstractMatrix{<:Number}; composite=Matrix{S}(undef,size(data)), x0=ones(S,length(models)), factr::Number=1e-12, pgtol::Number=1e-5, iprint::Integer=0, kws...) where {S <: Number, T <: AbstractMatrix{S}}
    G = similar(x0)
    fg(x) = (R = SFH.fg!(true,G,x,models,data,composite); return R,G)
    LBFGSB.lbfgsb(fg, x0; lb=zeros(length(models)), ub=fill(Inf,length(models)), factr=factr, pgtol=pgtol, iprint=iprint, kws...)
end
# LBFGSB.lbfgsb(f, g!, x0; m, lb, ub, kwargs...)
# LBFGSB.lbfgsb(f, x0; m, lb, ub, kwargs...) # f returns objective, gradient
# LBFGSB.lbfgsb is considerably more efficient than SPGBox, but doesn't work very nicely with Float32 and other numeric types. Majority of time is spent in calls to function evaluation (fg!), which is good. 
# Efficiency scales pretty strongly with `m` parameter that sets the memory size for the hessian approximation

# Not very efficient but don't care
# Returns cumulative SFH, 
function calculate_cum_sfr(coeffs::AbstractVector, logAge::AbstractVector, MH::AbstractVector; normalize_value=1, sorted::Bool=false)
    @assert axes(coeffs) == axes(logAge) == axes(MH)
    coeffs = coeffs .* normalize_value # Transform the coefficients to proper stellar masses
    mstar_total = sum(coeffs) # Calculate the total stellar mass of the model
    # Calculate the stellar mass per time bin by summing over the different MH at each logAge
    if ~sorted # If we aren't sure that logAge is sorted, we sort. 
        idx = sortperm(logAge)
        logAge = logAge[idx]
        coeffs = coeffs[idx]
        MH = MH[idx] 
    end
    unique_logAge = unique(logAge)
    dt = diff( vcat(0, exp10.(unique_logAge)) )
    mstar_arr = similar(coeffs)
    mean_mh_arr = zeros(eltype(MH), length(logAge))
    for i in eachindex(unique_logAge)
        Mstar_tmp = zero(eltype(mstar_arr))
        mh_tmp = Vector{eltype(MH)}(undef,0)
        for j in eachindex(logAge)
            if unique_logAge[i] == logAge[j]
                Mstar_tmp += coeffs[j]
                push!(mh_tmp, MH[j])
            end
        end
        mstar_arr[i] = Mstar_tmp
        mean_mh_arr[i] = mean(mh_tmp)
    end
    cum_sfr_arr = cumsum(reverse(mstar_arr)) ./ mstar_total
    reverse!(cum_sfr_arr)
    return unique_logAge, cum_sfr_arr, mstar_arr ./ dt, mean_mh_arr
end

# function calculate_sfr(coeffs::AbstractVector, logAge::AbstractVector; normalize_value=1, sorted::Bool=false)
#     coeffs = coeffs .* normalize_value # Transform the coefficients to proper stellar masses
#     if ~sorted # If we aren't sure that logAge is sorted, we sort. 
#         idx = sortperm(logAge)
#         logAge = logAge[idx]
#         coeffs = coeffs[idx]
#     end
#     unique_logAge = unique(logAge)
#     dt = diff( vcat(0, exp10.(unique_logAge)) )
#     # Figure out total stellar mass as a function of logAge if there are duplicates in logAge
#     mstar_arr = similar(coeffs)
#     for i in eachindex(unique_logAge)
#         Mstar_tmp = zero(eltype(mstar_arr))
#         for j in eachindex(logAge)
#             if unique_logAge[i] == logAge[j]
#                 Mstar_tmp += coeffs[j]
#             end
#         end
#         mstar_arr[i] = Mstar_tmp
#     end
#     return mstar_arr ./ dt
# end

# M1 = rand(120,100)
# M2 = rand(120, 100)
# N1 = rand.( Poisson.( (250.0 .* M1))) .+ rand.(Poisson.((500.0 .* M2)))
# Optim.optimize(x->-loglikelihood(x,[M1,M2],N1),[1.0,1.0],Optim.LBFGS())
# C1 = similar(M1)
# Optim.optimize(Optim.only_fg!( (F,G,x)->fg!(F,G,x,[M1,M2],N1,C1) ),[1.0,1.0],Optim.LBFGS())
# G=[1.0, 1.0]; coe=[5.0,5.0]; MM=[M1,M2]
# fg!(true,G,coe,MM,N1,C1)
# @benchmark fg!($true,$G,$coe,$MM,$N1,$C1)


###############################################################################
# HMC utilities

struct HMCModel{T,S,V}
    models::T
    composite::S
    data::V
end

# This model will return loglikelihood and gradient
LogDensityProblems.capabilities(::Type{<:HMCModel}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(problem::HMCModel) = length(problem.models)

function LogDensityProblems.logdensity_and_gradient(problem::HMCModel, logx)
    composite = problem.composite
    models = problem.models
    data = problem.data
    dims = length(models)
    # Transform the provided x
    x = SVector{dims}(exp(i) for i in logx)
    # Update the composite model matrix
    composite!( composite, x, models )
    logL = loglikelihood(composite, data) + sum(logx) # + sum(logx) is the Jacobian correction
    # ∇logL = SVector{dims}( ∇loglikelihood(models[i], composite, data) * x[i] for i in eachindex(models,x) ) # The `* x[i]` is the Jacobian correction
    ∇logL = [ ∇loglikelihood(models[i], composite, data) * x[i] + 1 for i in eachindex(models,x) ] # The `* x[i] + 1` is the Jacobian correction
    return logL, ∇logL
end

function hmc_sample(models::AbstractVector{T}, data::AbstractMatrix{<:Number}, nsteps::Integer=100; composite=Matrix{S}(undef,size(data)), rng::AbstractRNG=default_rng(), kws...) where {S <: Number, T <: AbstractMatrix{S}}
    instance = HMCModel( models, composite, data )
    return DynamicHMC.mcmc_with_warmup(rng, instance, nsteps; kws...)
end
