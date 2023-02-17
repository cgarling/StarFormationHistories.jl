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
"""
    x0::typeof(logage) = construct_x0(logage::AbstractVector{T}; normalize_value::Number=one(T)) where T <: Number

Generates a vector of initial stellar mass normalizations for input to `fit_templates` or `hmc_sample` with a total stellar mass of `normalize_value` such that the implied star formation rate is constant across the provided `logage` vector that contains the `log10(age [yr])` of each isochrone that you are going to input as models.

# Examples
```julia
julia> x0 = SFH.construct_x0(repeat([7.0,8.0,9.0],3); normalize_value=5.0)
9-element Vector{Float64}: ...

julia> sum(x0) = 5.05... # Close to `normalize_value`. 
"""
function construct_x0(logage::AbstractVector{T}; normalize_value::Number=one(T)) where T <: Number
    minlog, maxlog = extrema(logage)
    sfr = normalize_value / (exp10(maxlog) - exp10(minlog)) # Average SFR / yr
    unique_logage = sort!(unique(logage))
    num_ages = [count(logage .== la) for la in unique_logage] # number of entries per unique
    # unique_logage is sorted, but we want the first element to be zero to properly calculate
    # the dt from present-day to the most recent logage in the vector, so vcat it on
    unique_logage = vcat( [zero(T)], unique_logage )
    dt = [exp10(unique_logage[i+1]) - exp10(unique_logage[i]) for i in 1:length(unique_logage)-1]
    result = similar(logage)
    for i in eachindex(logage, result)
        la = logage[i]
        idx = findfirst( x -> x==la, unique_logage )
        result[i] = sfr * dt[idx-1] / num_ages[idx-1]
    end
    return result
end

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
    mstar_arr = similar(unique_logAge) # similar(coeffs)
    mean_mh_arr = zeros(eltype(MH), length(unique_logAge))
    for i in eachindex(unique_logAge)
        Mstar_tmp = zero(eltype(mstar_arr))
        mh_tmp = Vector{eltype(MH)}(undef,0)
        coeff_tmp = Vector{eltype(coeffs)}(undef,0)
        for j in eachindex(logAge)
            if unique_logAge[i] == logAge[j]
                Mstar_tmp += coeffs[j]
                push!(mh_tmp, MH[j])
                push!(coeff_tmp, coeffs[j])
            end
        end
        mstar_arr[i] = Mstar_tmp
        coeff_sum = sum(coeff_tmp)
        if coeff_sum == 0
            if i == 1
                mean_mh_arr[i] = mean(mh_tmp)
            else
                mean_mh_arr[i] = mean_mh_arr[i-1]
            end
        else
            mean_mh_arr[i] = sum( mh_tmp .* coeff_tmp ) / sum(coeff_tmp) # mean(mh_tmp)
        end
    end
    cum_sfr_arr = cumsum(reverse(mstar_arr)) ./ mstar_total
    reverse!(cum_sfr_arr)
    return unique_logAge, cum_sfr_arr, mstar_arr ./ dt, mean_mh_arr
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
# LBFGSB.lbfgsb is considerably more efficient than SPGBox, but doesn't work very nicely with Float32 and other numeric types. Majority of time is spent in calls to function evaluation (fg!), which is good. 
# Efficiency scales pretty strongly with `m` parameter that sets the memory size for the hessian approximation.

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
# Fitting with a metallicity distribution function rather than totally free per logage
_gausspdf(x,μ,σ) = inv(σ) * exp( -((x-μ)/σ)^2 / 2 )  # Unnormalized, 1-D Gaussian PDF
# _gausstest(x,age,α,β,σ) = inv(σ) * exp( -((x-(α*age+β))/σ)^2 / 2 )
# ForwardDiff.derivative(x -> _gausstest(-1.0, 1e9, -1e-10, x, 0.2), -0.4) = -2.74
# _dgaussdβ(x,age,α,β,σ) = (μ = α*age+β; (x-μ) * exp( -((x-μ)/σ)^2 / 2 ) * inv(σ)^3)
# _dgaussdβ(-1.0,1e9,-1e-10,-0.4,0.2) = -2.74
@inline function fg2!(F, G, variables::AbstractVector{<:Number}, models::AbstractVector{T}, data::AbstractMatrix{<:Number}, composite::AbstractMatrix{<:Number}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}, σ::Number) where T <: AbstractMatrix{<:Number}
    # `variables` should have length `length(unique(logAge)) + 2`; coeffs for each unique
    # entry in logAge, plus α and β to define the MDF at fixed logAge
    @assert axes(data) == axes(composite)
    S = promote_type(eltype(variables), eltype(eltype(models)), eltype(eltype(data)), eltype(composite), eltype(logAge), eltype(metallicities))
    # Compute the coefficients on each model template given the `variables` and the MDF
    coeffs = Vector{S}(undef,length(models))
    coeff_variables = variables[begin:end-2]
    α, β = variables[end-1], variables[end]
    unique_logAge = unique(logAge)
    @assert length(variables) == length(unique_logAge)+2
    norm_vals = Vector{S}(undef,length(unique_logAge))
    for i in eachindex(unique_logAge)
        la = unique_logAge[i]
        μ = α * exp10(la) + β # Find the mean metallicity of this age bin
        idxs = findall( ==(la), logAge) # Find all entries that match this logAge
        tmp_coeffs = [_gausspdf(metallicities[j], μ, σ) for j in idxs] # Calculate relative weights
        A = sum(tmp_coeffs)
        norm_vals[i] = A
        # Make sure sum over tmp_coeffs equals 1 and write to coeffs
        coeffs[idxs] .= tmp_coeffs .* coeff_variables[i] ./ A 
    end
    # Fill the composite array with the equivalent of sum( coeffs .* models )
    composite!(composite, coeffs, models)
    if (F != nothing) & (G != nothing) # Optim.optimize wants objective and gradient
        @assert axes(G) == axes(variables)
        # Calculate the ∇loglikelihood with respect to model coefficients; we will need all of these
        fullG = [ ∇loglikelihood(models[i], composite, data) for i in axes(models,1) ]
        # Now need to do the transformation to the `variables` rather than model coefficients
        G[end-1] = zero(eltype(G))
        G[end] = zero(eltype(G))
        for i in axes(G,1)[begin:end-2] # 1:length(variables)-2
            la = unique_logAge[i]
            age = exp10(la)
            μ = α * age + β # Find the mean metallicity of this age bin
            idxs = findall( ==(la), logAge) # Find all entries that match this logAge
            tmp_coeffs = [_gausspdf(metallicities[j], μ, σ) for j in idxs] # Calculate relative weights
            A = sum(tmp_coeffs)
            # This should be correct for any MDF model at fixed logAge
            # @inbounds G[i] = -sum( fullG[j] * coeffs[j] / variables[i] for j in idxs )
            @inbounds G[i] = -sum( fullG[idxs[j]] * tmp_coeffs[j] / A for j in eachindex(idxs) )
            
            # G[end] += -sum( fullG[idxs[j]] * variables[i] *
            #     ( ((metallicities[idxs[j]]-μ) * exp( -((metallicities[idxs[j]]-μ)/σ)^2 / 2 ) * inv(σ)^3) / A -
            #     tmp_coeffs[j] / A^2 * sum( ((metallicities[idxs[k]]-μ) * exp( -((metallicities[idxs[k]]-μ)/σ)^2 / 2 ) * inv(σ)^3) for k in eachindex(idxs) ) ) for j in eachindex(idxs))
            βsum = sum( ((metallicities[idxs[j]]-μ) * tmp_coeffs[j]) for j in eachindex(idxs))
            dLdβ = -sum( fullG[idxs[j]] * variables[i] *
                ( ((metallicities[idxs[j]]-μ) * tmp_coeffs[j]) - tmp_coeffs[j] / A * βsum )
                         for j in eachindex(idxs)) / A / σ^2
            dLdα = dLdβ * age
            G[end-1] += dLdα
            G[end] += dLdβ

        end
        return -loglikelihood(composite, data) # Return the negative loglikelihood
    end
    # elseif G != nothing # Optim.optimize wants only gradient (Does this ever happen?)
    #     @assert axes(G) == axes(models)
    #     # Fill the gradient array
    #     for i in axes(models,1)
    #         @inbounds G[i] = -∇loglikelihood(models[i], composite, data)
    #     end
    # elseif F != nothing # Optim.optimize wants only objective
    #     return -loglikelihood(composite, data) # Return the negative loglikelihood
    # end
end

function fg3!(variables::AbstractVector{<:Number}, models::AbstractVector{T}, data::AbstractMatrix{<:Number}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}, σ::Number) where T <: AbstractMatrix{<:Number}
    unique_logAge = unique(logAge)
    α, β = variables[end-1], variables[end]
    coeffs = reduce(vcat, begin
                  la = unique_logAge[i]
                  μ = α * exp10(la) + β
                  idxs = findall( ==(la), logAge)
                  tmp_coeffs = [_gausspdf(metallicities[j], μ, σ) for j in idxs]
                  A = sum(tmp_coeffs)
                  tmp_coeffs .* variables[i] ./ A 
                    end for i in eachindex(unique_logAge) )
    
    composite = sum( coeffs .* models )
    return -loglikelihood(composite, data)
    
end
# unique_logAge = range(6.6, 10.1; step=0.1)
# unique_MH = range(-2.2, 0.3; step=0.1)
# template_logAge = repeat(unique_logAge; inner=length(unique_MH))
# template_MH = repeat(unique_MH; outer=length(unique_logAge))
# models = [rand(99,99) for i in 1:length(template_logAge)]
# coeffs = rand(length(template_logAge))
# data = sum( coeffs .* models )
# variables = ones(length(unique_logAge)+2)
# C = similar(data)
# G = rand(length(unique_logAge)+2)
# variables[end] = -0.4 # Intercept at present-day
# variables[end-1] = -1.103700353306591e-10 # Slope in MH/yr, with yr being in terms of lookback 
# tmpans = SFH.fg2!(true, G, variables, models, data, C, template_logAge, template_MH, 0.2)
# println(sum(tmpans[1:length(unique_MH)]) ≈ 1) # This should be true if properly normalized
# This checks that the mean MH is correct for the first unique logAge
# println( isapprox(-0.4, sum( tmpans[1:length(unique_MH)] .* unique_MH ) / sum( tmpans[1:length(unique_MH)] ), atol=1e-3 ) )
# import ForwardDiff
# ForwardDiff.gradient( x-> SFH.fg2!(true, G, x, models, data, C, template_logAge, template_MH, 0.2), variables)
# FiniteDifferences is very slow but agrees with ForwardDiff
# import FiniteDifferences
# FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), x-> SFH.fg2!(true, rand(length(variables)), x, models, data, C, template_logAge, template_MH, 0.2), variables)
# G2 = similar(coeffs)
# @benchmark SFH.fg!($true, $G2, $coeffs, $models, $data, $C) # 7.6 ms
# @benchmark SFH.fg2!($true, $G, $variables, $models, $data, $C, $template_logAge, $template_MH, $0.2) # currently 8 ms

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
    x = [ exp(i) for i in logx ]
    # Update the composite model matrix
    composite!( composite, x, models )
    logL = loglikelihood(composite, data) + sum(logx) # + sum(logx) is the Jacobian correction
    ∇logL = [ ∇loglikelihood(models[i], composite, data) * x[i] + 1 for i in eachindex(models,x) ] # The `* x[i] + 1` is the Jacobian correction
    return logL, ∇logL
end

# Version with just one chain; no threading
function hmc_sample(models::AbstractVector{T}, data::AbstractMatrix{<:Number}, nsteps::Integer; composite=Matrix{S}(undef,size(data)), rng::AbstractRNG=default_rng(), kws...) where {S <: Number, T <: AbstractMatrix{S}}
    instance = HMCModel( models, composite, data )
    return DynamicHMC.mcmc_with_warmup(rng, instance, nsteps; kws...)
end

function extract_initialization(state)
    (; Q, κ, ϵ) = state.final_warmup_state
    (; q = Q.q, κ, ϵ)
end

# Version with multiple chains and multithreading
function hmc_sample(models::AbstractVector{T}, data::AbstractMatrix{<:Number}, nsteps::Integer, nchains::Integer; composite=[ Matrix{S}(undef,size(data)) for i in 1:Threads.nthreads() ], rng::AbstractRNG=default_rng(), initialization=(), kws...) where {S <: Number, T <: AbstractMatrix{S}}
    @assert nchains >= 1
    instances = [ HMCModel( models, composite[i], data ) for i in 1:Threads.nthreads() ]
    # Do the warmup
    warmup = DynamicHMC.mcmc_keep_warmup(rng, instances[1], 0;
                                         warmup_stages=DynamicHMC.default_warmup_stages(), initialization=initialization, kws...)
    final_init = extract_initialization(warmup)
    # Do the MCMC
    result_arr = []
    Threads.@threads for i in 1:nchains
        tid = Threads.threadid()
        result = DynamicHMC.mcmc_with_warmup(rng, instances[tid], nsteps; warmup_stages=(),
                                             initialization=final_init, kws...)
        push!(result_arr, result) # Order doesn't matter so push when completed
    end
    return result_arr
end
