# Methods and utilities for fitting star formation histories
# Currently not sure if I should require data and composite model > 0 or != 0 for loglikelihood and ∇loglikelihood; something to look at.
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
    @turbo thread=true for idx in eachindex(composite, data) # LoopVectorization.@turbo gives 2x speedup here
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
    @turbo thread=true for idx in eachindex(model, composite, data) # ~4x speedup from LoopVectorization.@turbo here
        # Setting eps() as minimum of composite greatly improves stability of convergence.
        @inbounds ci = max( composite[idx], eps(T) )
        @inbounds mi = model[idx]
        @inbounds ni = data[idx]
        # Returning NaN is required for Optim.jl but not for SPGBox.jl
        # result += ifelse( (ci > zero(T)) & (ni > zero(T)), -mi * (one(T) - ni/ci), zero(T)) # NaN)
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
        # Setting eps() as minimum of composite greatly improves stability of convergence.
        @inbounds ci = max( composite[idx], eps(T) )
        @inbounds ni = data[idx]
        @inbounds composite[idx] = one(T) - ni/ci
    end
    for k in eachindex(G, models)
        @inbounds model = models[k]
        @assert axes(model) == axes(data) == axes(composite)
        result = zero(T)
        @turbo thread=true for idx in eachindex(model, data, composite)
            @inbounds mi = model[idx]
            @inbounds ni = data[idx]
            @inbounds nici = composite[idx]
            result += ifelse( ni > zero(T), -mi * nici, zero(T) )
        end
        @inbounds G[k] = result
    end
end

"""
    -logL = fg!(F, G, coeffs::AbstractVector{<:Number}, models::AbstractVector{T}, data::AbstractMatrix{<:Number}, composite::AbstractMatrix{<:Number}) where T <: AbstractMatrix{<:Number}

Computes -loglikelihood and its gradient simultaneously for use with Optim.jl and other optimization APIs. See documentation [here](https://julianlsolvers.github.io/Optim.jl/stable/#user/tipsandtricks/). 
"""
@inline function fg!(F, G, coeffs::AbstractVector{<:Number}, models::AbstractVector{T}, data::AbstractMatrix{<:Number}, composite::AbstractMatrix{<:Number}) where T <: AbstractMatrix{<:Number}
    S = promote_type(eltype(coeffs), eltype(eltype(models)), eltype(eltype(data)), eltype(composite))
    # Fill the composite array with the equivalent of sum( coeffs .* models )
    composite!(composite, coeffs, models) 
    if (F != nothing) & (G != nothing) # Optim.optimize wants objective and gradient
        # Calculate logL before ∇loglikelihood! which will overwrite composite
        logL = loglikelihood(composite, data)
        ∇loglikelihood!(G, composite, models, data) # Fill the gradient array
        G .*= -1 # We want the gradient of the negative log likelihood
        return -logL # Return the negative loglikelihood
    elseif G != nothing # Optim.optimize wants only gradient (Does this ever happen?)
        ∇loglikelihood!(G, composite, models, data) # Fill the gradient array
        G .*= -1 # We want the gradient of the negative log likelihood
    elseif F != nothing # Optim.optimize wants only objective
        return -loglikelihood(composite, data) # Return the negative loglikelihood
    end
end

"""
    x0::typeof(logage) = construct_x0(logage::AbstractVector{T}; normalize_value::Number=one(T)) where T <: Number

Generates a vector of initial stellar mass normalizations for input to [`fit_templates`](@ref) or [`hmc_sample`](@ref) with a total stellar mass of `normalize_value` such that the implied star formation rate is constant across the provided `logage` vector that contains the `log10(age [yr])` of each isochrone that you are going to input as models.

# Examples
```julia
julia> x0 = construct_x0(repeat([7.0,8.0,9.0],3); normalize_value=5.0)
9-element Vector{Float64}: ...

julia> sum(x0)
5.05... # Close to `normalize_value`.
```
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

"""
    (unique_logAge, cum_sfh, sfr, mean_MH) = calculate_cum_sfr(coeffs::AbstractVector, logAge::AbstractVector, MH::AbstractVector; normalize_value=1, sorted::Bool=false)

Calculates cumulative star formation history, star formation rates, and mean metallicity evolution as functions of `logAge = log10(age [yr])`.

# Arguments
 - `coeffs::AbstractVector` is a vector of stellar mass coefficients such as those returned by [`fit_templates`](@ref), for example. Actual stellar mass in stellar population `j` is `coeffs[j] * normalize_value`.
 - `logAge::AbstractVector` is a vector giving the `log10(age [yr])` of the stellar populations corresponding to the provided `coeffs`.
 - `MH::AbstractVector` is a vector giving the metallicities of the stellar populations corresponding to the provided `coeffs`.

# Keyword Arguments
 - `normalize_value` is a multiplicative prefactor to apply to all the `coeffs`; same as the keyword in [`partial_cmd_smooth`](@ref).
 - `sorted::Bool` is either `true` or `false` and signifies whether to assume `logAge` is sorted.

# Returns
 - `unique_logAge::Vector` is essentially `unique(sort(logAge))` and provides the x-values you would plot the other returned vectors against.
 - `cum_sfh::Vector` is the normalized cumulative SFH implied by the provided `coeffs`. This is ~1 at the most recent time in `logAge` and decreases as `logAge` increases.
 - `sfr::Vector` gives the star formation rate across each bin in `unique_logAge`. If `coeffs .* normalize_value` are in units of solar masses, then `sfr` is in units of solar masses per year.
 - `mean_MH::Vector` gives the stellar-mass-weighted mean metallicity of the stellar population as a function of `unique_logAge`. 
"""
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
    mstar_arr = Vector{eltype(coeffs)}(undef, length(unique_logAge))
    mean_mh_arr = zeros(promote_type(eltype(MH),eltype(coeffs)), length(unique_logAge))
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
    (-logL, coeffs) = fit_templates_lbfgsb(models::AbstractVector{T}, data::AbstractMatrix{<:Number}; composite=Matrix{S}(undef,size(data)), x0=ones(S,length(models)), factr::Number=1e-12, pgtol::Number=1e-5, iprint::Integer=0, kws...) where {S <: Number, T <: AbstractMatrix{S}}

Finds the coefficients `coeffs` that maximize the Poisson likelihood ratio (equations 7--10 in [Dolphin 2002](http://adsabs.harvard.edu/abs/2002MNRAS.332...91D)) for the composite Hess diagram model `sum(models .* coeffs)` given the provided templates `models` and the observed Hess diagram `data` using the box-constrained LBFGS method provided by [LBFGSB.jl](https://github.com/Gnimuc/LBFGSB.jl). 

# Arguments
 - `models::AbstractVector{AbstractMatrix{<:Number}}`: the list of template Hess diagrams for the simple stellar populations (SSPs) being considered; all must have the same size.
 - `data::AbstractMatrix{<:Number}`: the observed Hess diagram; must match the size of the templates contained in `models`.

# Keyword Arguments
 - `composite`: The working matrix that will be used to store the composite Hess diagram model during computation; must be of the same size as the templates contained in `models` and the observed Hess diagram `data`.
 - `x0`: The vector of initial guesses for the stellar mass coefficients. You should basically always be calculating and passing this keyword argument; we provide [`StarFormationHistories.construct_x0`](@ref) to prepare `x0` assuming constant star formation rate, which is typically a good initial guess.
 - `factr::Number`: Keyword argument passed to `LBFGSB.lbfgsb`; essentially a relative tolerance for convergence based on the inter-iteration change in the objective function.
 - `pgtol::Number`: Keyword argument passed to `LBFGSB.lbfgsb`; essentially a relative tolerance for convergence based on the inter-iteration change in the projected gradient of the objective.
 - `iprint::Integer`: Keyword argument passed to `LBFGSB.lbfgsb` controlling how much information is printed to the terminal; setting to `1` can sometimes be helpful to diagnose convergence issues.
Other `kws...` are passed to `LBFGSB.lbfgsb`.

# Returns
 - `-logL::Number`: the minimum negative log-likelihood found by the optimizer.
 - `coeffs::Vector{<:Number}`: the maximum likelihood estimate for the coefficient vector. 

# Notes
 - It can be helpful to normalize your `models` to contain realistic total stellar masses to aid convergence stability; for example, if the total stellar mass of your population is 10^7 solar masses, then you might normalize your templates to contain 10^3 solar masses. If you are using [`partial_cmd_smooth`](@ref) to construct the templates, you can specify this normalization via the `normalize_value` keyword. 
"""
function fit_templates_lbfgsb(models::AbstractVector{T}, data::AbstractMatrix{<:Number}; composite=Matrix{S}(undef,size(data)), x0=ones(S,length(models)), factr::Number=1e-12, pgtol::Number=1e-5, iprint::Integer=0, kws...) where {S <: Number, T <: AbstractMatrix{S}}
    @assert (size(data) == size(composite)) & all(size(i) == size(data) for i in models)
    G = similar(x0)
    fg(x) = (R = fg!(true,G,x,models,data,composite); return R,G)
    LBFGSB.lbfgsb(fg, x0; lb=zeros(length(models)), ub=fill(Inf,length(models)), factr=factr, pgtol=pgtol, iprint=iprint, kws...)
end


"""
    result = fit_templates(models::AbstractVector{T}, data::AbstractMatrix{<:Number}; composite=Matrix{S}(undef,size(data)), x0=ones(S,length(models)), kws...) where {S <: Number, T <: AbstractMatrix{S}}

Finds both maximum likelihood estimate (MLE) and maximum a posteriori estimate (MAP) for the coefficients `coeffs` such that the composite Hess diagram model is `sum(models .* coeffs)` using the provided templates `models` and the observed Hess diagram `data`. Utilizes the Poisson likelihood ratio (equations 7--10 in [Dolphin 2002](http://adsabs.harvard.edu/abs/2002MNRAS.332...91D)) for the likelihood of the data given the model. See the examples in the documentation for comparisons of the results of this method and [`hmc_sample`](@ref). 

# Arguments
 - `models::AbstractVector{AbstractMatrix{<:Number}}`: the list of template Hess diagrams for the simple stellar populations (SSPs) being considered; all must have the same size.
 - `data::AbstractMatrix{<:Number}`: the observed Hess diagram; must match the size of the templates contained in `models`.

# Keyword Arguments
 - `composite`: The working matrix that will be used to store the composite Hess diagram model during computation; must be of the same size as the templates contained in `models` and the observed Hess diagram `data`.
 - `x0`: The vector of initial guesses for the stellar mass coefficients. You should basically always be calculating and passing this keyword argument; we provide [`StarFormationHistories.construct_x0`](@ref) to prepare `x0` assuming constant star formation rate, which is typically a good initial guess.
Other `kws...` are passed to `Optim.options` to set things like convergence criteria for the optimization. 

# Returns
`result` is a `NamedTuple` containing two `NamedTuple`s, each of which has identical structure. The two components of `result` are `result.map=result[1]`, which contains the results of the MAP optimization, and `result.mle=result[2]`, which contains the results of the MLE optimization. Each of these `NamedTuple`s contain the following fields, accessible as, e.g., `result.map.μ`, `result.map.result`, etc.:
 - `μ::Vector{<:Number}` are the optimized `coeffs` at the maximum.
 - `σ::Vector{<:Number}` are the standard errors on the coeffs `μ` calculated from an estimate of the inverse Hessian matrix evaluated at `μ`. The inverse of the Hessian matrix at the maximum of the likelihood (or posterior) is a estimator for the variance-covariance matrix of the parameters, but is only accurate when the second-order expansion given by the Hessian at the maximum is a good approximation to the function being optimized (i.e., when the optimized function is approximately quadratic around the maximum; see [Dovi et al. 1991](https://doi.org/10.1016/0893-9659(91)90129-J) for more information). We find this is often the case for the MAP estimate, but the errors found for coefficients that are ≈0 in the MLE are typically unrealistically small. For coefficients significantly greater than 0, the `σ` values from the MAP and MLE are typically consistent to 5--10%.
 - `invH::Matrix{<:Number}` is the estimate of the inverse Hessian matrix at `μ` that was used to derive `σ`. The optimization is done under a logarithmic transform, such that `θ[j] = log(coeffs[j])` are the actual parameters optimized, so the entries in the Hessian are actually
```math
H^{(j,k)} ( \\boldsymbol{\\hat \\theta} ) = \\left. \\frac{\\partial^2 \\, J(\\boldsymbol{\\theta})}{\\partial \\theta_j \\, \\partial \\theta_k} \\right\\vert_{\\boldsymbol{\\theta}=\\boldsymbol{\\hat \\theta}}
```
 - `result` is the full object returned by the optimization from `Optim.jl`; this is of type `Optim.MultivariateOptimizationResults`. Remember that the optimization is done with parameters `θ[j] = log(coeffs[j])` when dealing with this raw output. This means that, for example, we calculate `result.map.μ` as `exp.(Optim.minimizer(result.map.result))`.

# Notes
 - This method uses the `BFGS` method from `Optim.jl` internally because it builds and tracks the inverse Hessian matrix approximation which can be used to estimate parameter uncertainties. BFGS is much more memory-intensive than LBFGS (as used for [`StarFormationHistories.fit_templates_lbfgsb`](@ref)) for large numbers of parameters (equivalently, many `models`), so you should consider LBFGS to solve for the MLE along with [`hmc_sample`](@ref) to sample the posterior if you are using a large grid of models (greater than a few hundred).
 - The BFGS implementation we use from Optim.jl uses BLAS operations during its iteration. The OpenBLAS that Julia ships with will often default to running on multiple threads even if Julia itself is started with only a single thread. You can check the current number of BLAS threads with `import LinearAlgebra: BLAS; BLAS.get_num_threads()`. For the problem sizes typical of this function we actually see performance regression with larger numbers of BLAS threads. For this reason you may wish to use BLAS in single-threaded mode; you can set this as `import LinearAlgebra: BLAS; BLAS.set_num_threads(1)`.
"""
function fit_templates(models::AbstractVector{T}, data::AbstractMatrix{<:Number}; composite=Matrix{S}(undef,size(data)), x0=ones(S,length(models)), kws...) where {S <: Number, T <: AbstractMatrix{S}}
    @assert (size(data) == size(composite)) & all(size(i) == size(data) for i in models)
    # log-transform the initial guess vector
    x0 = log.(x0)
    # Make scratch array for assessing transformations
    x = similar(x0)
    function fg_map!(F,G,logx)
        @. x = exp(logx)
        logL = fg!(true,G,x,models,data,composite) - sum(logx) # - sum(logx) is the Jacobian correction
        if G != nothing
            @. G = G * x - 1 # Add Jacobian correction and log-transform to every element of G
        end
        return logL
    end
    function fg_mle!(F,G,logx)
        @. x = exp(logx)
        logL = fg!(true,G,x,models,data,composite)
        if G != nothing
            @. G = G * x # Only correct for log-transform
        end
        return logL
    end
    # Setup for Optim.jl
    # The InitialStatic(1.0,true) alphaguess helps to regularize the optimization and 
    # makes it less sensitive to initial x0.
    bfgs_struct = Optim.BFGS(; alphaguess=LineSearches.InitialStatic(1.0,true),
                             linesearch=LineSearches.HagerZhang())
    # The extended trace will contain the BFGS estimate of the inverse Hessian, aka the
    # covariance matrix, which we can use to make parameter uncertainty estimates
    bfgs_options = Optim.Options(; allow_f_increases=true, store_trace=true, extended_trace=true, kws...)
    # Calculate result
    result_map = Optim.optimize(Optim.only_fg!( fg_map! ), x0, bfgs_struct, bfgs_options)
    # Calculate final result without prior
    result_mle = Optim.optimize(Optim.only_fg!( fg_mle! ), Optim.minimizer(result_map), bfgs_struct, bfgs_options)
    # result = Optim.optimize(Optim.only_fg!( fg_final! ), fill(-10.0,length(models)), fill(Inf,length(models)), Optim.minimizer(result_prior), Optim.Fminbox(bfgs_struct), Optim.Options(; allow_f_increases=true, store_trace=true, extended_trace=true, outer_iterations=1, kws...))
    # NamedTuple of NamedTuples. "map" contains the mean `μ`, standard error `σ`, 
    # the estimate of the inverse Hessian `invH` from BFGS and full result struct `result`
    # for the maximum a posteriori estimate. "mle" contains the same entries but for the maximum
    # likelihood estimate. 
    return  ( map = ( μ = exp.(Optim.minimizer(result_map)),
                            σ = sqrt.(diag(Optim.trace(result_map)[end].metadata["~inv(H)"])) .* exp.(Optim.minimizer(result_map)),
                            invH = result_map.trace[end].metadata["~inv(H)"],
                            result = result_map ),
              mle = ( μ = exp.(Optim.minimizer(result_mle)),
                      σ = sqrt.(diag(Optim.trace(result_mle)[end].metadata["~inv(H)"])) .* exp.(Optim.minimizer(result_mle)),
                      invH = result_mle.trace[end].metadata["~inv(H)"],
                      result = result_mle) )
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

"""
    hmc_sample(models::AbstractVector{T}, data::AbstractMatrix{<:Number}, nsteps::Integer [, nchains::Integer]; composite=Matrix{S}(undef,size(data)), rng::AbstractRNG=default_rng(), kws...) where {S <: Number, T <: AbstractMatrix{S}}

Function to sample the posterior of the coefficients `coeffs` such that the full model of the observational `data` is `sum(models .* coeffs)`. Uses the Poisson likelihood ratio as defined by equations 7--10 of Dolphin 2002 along with a logarithmic transformation of the `coeffs` so that the fitting variables are continuous and differentiable over all reals. Sampling is done using the No-U-Turn sampler as implemented in [DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl), which is a form of dynamic Hamiltonian Monte Carlo.

# Arguments
 - `models::AbstractVector{<:AbstractMatrix{<:Number}}` is a vector of equal-sized matrices that represent the template Hess diagrams for the simple stellar populations that compose the observed Hess diagram.
 - `data::AbstractMatrix{<:Number}` is the Hess diagram for the observed data.
 - `nsteps::Integer` is the number of samples to draw per chain.

# Optional Arguments
 - `nchains::Integer`: If this argument is not provided, this method will return a single chain. If this argument is provided, it will sample `nchains` chains using all available threads and will return a vector of the individual chains. If `nchains` is set, `composite` must be a vector of matrices containing a working matrix for each chain. 

# Keyword Arguments
 - `composite` is the working matrix (or vector of matrices, if the argument `nchains` is provided) that will be used to store the composite Hess diagram model during computation; must be of the same size as the templates contained in `models` and the observed Hess diagram `data`.
 - `rng::AbstractRNG` is the random number generator that will be passed to DynamicHMC.jl. If `nchains` is provided this method will attempt to sample in parallel, requiring a thread-safe `rng` such as that provided by `Random.default_rng()`. 
All other keyword arguments `kws...` will be passed to `DynamicHMC.mcmc_with_warmup` or `DynamicHMC.mcmc_keep_warmup` depending on whether `nchains` is provided.

# Returns
 - If `nchains` is not provided, returns a `NamedTuple` as summarized in DynamicHMC.jl's documentation. In short, the matrix of samples can be extracted and transformed as `exp.( result.posterior_matrix )`. Statistics about the chain can be obtained with `DynamicHMC.Diagnostics.summarize_tree_statistics(result.tree_statistics)`; you want to see a fairly high acceptance rate (>0.5) and the majority of samples having termination criteria being "turning." See DynamicHMC.jl's documentation for more information.

# Examples
```julia
import DynamicHMC
import StatFormationHistories: hmc_sample
import Statistics: mean
# Run sampler using progress meter to monitor progress
# assuming you have constructed some templates `models` and your observational Hess diagram `data`
julia> result = hmc_sample( models, data, 1000; reporter=DynamicHMC.ProgressMeterReport())
# The chain values are stored in result.posterior matrix; extract them with `result.posterior_matrix`
# An exponential transformation is needed since the optimization internally uses a logarithmic 
# transformation and samples log(θ) rather than θ directly. 
julia> mc_matrix = exp.( result.posterior_matrix )
# We can look at some statistics from the chain; want to see high acceptance rate (>0.5) and large % of
# "turning" for termination criteria. 
julia> DynamicHMC.Diagnostics.summarize_tree_statistics(result.tree_statistics)
Hamiltonian Monte Carlo sample of length 1000
  acceptance rate mean: 0.92, 5/25/50/75/95%: 0.65 0.88 0.97 1.0 1.0
  termination: divergence => 0%, max_depth => 0%, turning => 100%
  depth: 0 => 0%, 1 => 64%, 2 => 36%
# mc_matrix has size `(length(models), nsteps)` so each column is an independent
# sample of the SFH as defined by the coefficients and the rows contain the samples
# for each parameter. 
julia> mstar_tot = sum.(eachcol(mc_matrix)) # Total stellar mass of the modelled system per sample
julia> mc_means = mean.(eachrow(mc_matrix)) # Mean of each coefficient evaluated across all samples
# Example with multiple chains sampled in parallel via multi-threading
import Threads
julia> t_result = hmc_sample( models, data, 1000, Threads.nthreads(); reporter=DynamicHMC.ProgressMeterReport())
# Combine the multiple chains into a single matrix and transform
# Can then use the same way as `mc_matrix` above
julia> mc_matrix = exp.( DynamicHMC.pool_posterior_matrices(t_result) )
```
"""
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

######################################################################################
######################################################################################
# Fitting with a metallicity distribution function rather than totally free per logage

"""
    x0::Vector = construct_x0_mdf(logage::AbstractVector{T}; normalize_value::Number=one(T)) where T <: Number

Generates a vector of initial stellar mass normalizations for input to [`StarFormationHistories.fit_templates_mdf`](@ref) or [`StarFormationHistories.hmc_sample_mdf`](@ref) with a total stellar mass of `normalize_value` such that the implied star formation rate is constant across the provided `logage` vector that contains the `log10(age [yr])` of each isochrone that you are going to input as models.

The difference between this function and [`StarFormationHistories.construct_x0`](@ref) is that this function generates an `x0` vector that is of length `length(unique(logage))` (that is, a single normalization factor for each unique entry in `logage`) while [`StarFormationHistories.construct_x0`](@ref) returns an `x0` vector that is of length `length(logage)`; that is, a normalization factor for every entry in `logage`. The order of the coefficients is such that the coefficient `x[i]` corresponds to the entry `unique(logage)[i]`. 

# Notes

# Examples
```julia
julia> construct_x0_mdf([9.0,8.0,7.0]; normalize_value=5.0)
3-element Vector{Float64}:
 4.545454545454545
 0.45454545454545453
 0.050505045454545455

julia> construct_x0_mdf(repeat([9.0,8.0,7.0,8.0];inner=3); normalize_value=5.0)
3-element Vector{Float64}:
 4.545454545454545
 0.45454545454545453
 0.050505045454545455

julia> construct_x0_mdf(repeat([9.0,8.0,7.0,8.0],3); normalize_value=5.0) ≈ construct_x0([9.0,8.0,7.0]; normalize_value=5.0)
true
```
"""
function construct_x0_mdf(logage::AbstractVector{T}; normalize_value::Number=one(T)) where T <: Number
    minlog, maxlog = extrema(logage)
    sfr = normalize_value / (exp10(maxlog) - exp10(minlog)) # Average SFR / yr
    unique_logage = unique(logage)
    idxs = sortperm(unique_logage)
    # unique_logage is sorted, but we want the first element to be zero to properly calculate
    # the dt from present-day to the most recent logage in the vector, so vcat it on
    sorted_ul = vcat([zero(T)], unique_logage[idxs])
    dt = [exp10(sorted_ul[i+1]) - exp10(sorted_ul[i]) for i in 1:length(sorted_ul)-1]
    # return sfr .* dt
    return [ begin
                idx = findfirst( ==(sorted_ul[i+1]), unique_logage )
                sfr * dt[idx]
             end for i in eachindex(dt) ]
end

_gausspdf(x,μ,σ) = exp( -((x-μ)/σ)^2 / 2 )  # Unnormalized, 1-D Gaussian PDF
# _gausstest(x,age,α,β,σ) = inv(σ) * exp( -((x-(α*age+β))/σ)^2 / 2 )
# ForwardDiff.derivative(x -> _gausstest(-1.0, 1e9, -1e-10, x, 0.2), -0.4) = -2.74
# _dgaussdβ(x,age,α,β,σ) = (μ = α*age+β; (x-μ) * exp( -((x-μ)/σ)^2 / 2 ) * inv(σ)^3)
# _dgaussdβ(-1.0,1e9,-1e-10,-0.4,0.2) = -2.74

"""
    coeffs = calculate_coeffs_mdf(variables::AbstractVector{<:Number}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number} [, α::Number, β::Number, σ::Number])

Calculates per-model stellar mass coefficients `coeffs` from the fitting parameters of [`StarFormationHistories.fit_templates_mdf`](@ref) and [`StarFormationHistories.hmc_sample_mdf`](@ref). The `variables` returned by these functions is of length `length(unique(logAge))+3`. The first `length(logAge)` entries are stellar mass coefficients, one per unique entry in `logAge`. The final three elements are α, β, and σ defining a metallicity evolution such that the mean for element `i` of `unique(logAge)` is `μ[i] = α * exp10(unique(logAge)[i]) / 1e9 + β`. The individual weights per each isochrone are then determined via Gaussian weighting with the above mean and the provided `σ`. 
"""
function calculate_coeffs_mdf(variables::AbstractVector{<:Number}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}, α::Number, β::Number, σ::Number)
    S = promote_type(eltype(variables), eltype(logAge), eltype(metallicities), typeof(α), typeof(β), typeof(σ))
    # Compute the coefficients on each model template given the `variables` and the MDF
    unique_logAge = unique(logAge)
    @assert length(variables) == length(unique_logAge)
    coeffs = Vector{S}(undef,length(logAge))
    norm_vals = Vector{S}(undef,length(unique_logAge))
    for i in eachindex(unique_logAge)
        la = unique_logAge[i]
        μ = α * exp10(la) / 1e9 + β # Find the mean metallicity of this age bin
        idxs = findall( ==(la), logAge) # Find all entries that match this logAge
        tmp_coeffs = [_gausspdf(metallicities[j], μ, σ) for j in idxs] # Calculate relative weights
        A = sum(tmp_coeffs)
        norm_vals[i] = A
        # Make sure sum over tmp_coeffs equals 1 and write to coeffs
        coeffs[idxs] .= tmp_coeffs .* variables[i] ./ A
    end
    return coeffs
end
calculate_coeffs_mdf(variables, logAge, metallicities) =
    calculate_coeffs_mdf(view(variables,firstindex(variables):lastindex(variables)-3),
                         logAge, metallicities, variables[end-2], variables[end-1], variables[end])


"""
variables[begin:end-2] are stellar mass coefficients
variables[end-1] is the slope of the age-MH relation in [MH] / [10Gyr; (lookback)], e.g. -1.0
variables[end] is the intercept of the age-MH relation in MH at present-day, e.g. -0.4
"""
@inline function fg_mdf_fixedσ!(F, G, variables::AbstractVector{<:Number}, models::AbstractVector{T}, data::AbstractMatrix{<:Number}, composite::AbstractMatrix{<:Number}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}, σ::Number) where T <: AbstractMatrix{<:Number}
    # `variables` should have length `length(unique(logAge)) + 2`; coeffs for each unique
    # entry in logAge, plus α and β to define the MDF at fixed logAge
    @assert axes(data) == axes(composite)
    S = promote_type(eltype(variables), eltype(eltype(models)), eltype(eltype(data)), eltype(composite), eltype(logAge), eltype(metallicities))
    # Compute the coefficients on each model template given the `variables` and the MDF
    α, β = variables[end-1], variables[end]
    unique_logAge = unique(logAge)
    coeffs = calculate_coeffs_mdf(view(variables,firstindex(variables):lastindex(variables)-2), logAge, metallicities, α, β, σ)
    # Fill the composite array with the equivalent of sum( coeffs .* models )
    # composite = sum( coeffs .* models )
    # return -loglikelihood(composite, data)
    composite!(composite, coeffs, models)
    logL = loglikelihood(composite, data)
    if (F != nothing) & (G != nothing) # Optim.optimize wants objective and gradient
        @assert axes(G) == axes(variables)
        # Calculate the ∇loglikelihood with respect to model coefficients; we will need all of these
        # fullG = [ ∇loglikelihood(models[i], composite, data) for i in axes(models,1) ]
        fullG = Vector{eltype(G)}(undef,length(models))
        ∇loglikelihood!(fullG, composite, models, data)
        # Now need to do the transformation to the `variables` rather than model coefficients
        G[end-1] = zero(eltype(G))
        G[end] = zero(eltype(G))
        for i in axes(G,1)[begin:end-2] 
            la = unique_logAge[i]
            age = exp10(la) / 1e9 # the / 1e9 makes α the slope in MH/Gyr, improves convergence
            μ = α * age + β # Find the mean metallicity of this age bin
            idxs = findall( ==(la), logAge) # Find all entries that match this logAge
            tmp_coeffs = [_gausspdf(metallicities[j], μ, σ) for j in idxs] # Calculate relative weights
            A = sum(tmp_coeffs)
            # This should be correct for any MDF model at fixed logAge
            @inbounds G[i] = -sum( fullG[j] * coeffs[j] / variables[i] for j in idxs )
            βsum = sum( ((metallicities[idxs[j]]-μ) * tmp_coeffs[j]) for j in eachindex(idxs))
            dLdβ = -sum( fullG[idxs[j]] * variables[i] *
                ( ((metallicities[idxs[j]]-μ) * tmp_coeffs[j]) - tmp_coeffs[j] / A * βsum )
                         for j in eachindex(idxs)) / A / σ^2
            dLdα = dLdβ * age
            G[end-1] += dLdα
            G[end] += dLdβ
        end
        return -logL
    elseif F != nothing # Optim.optimize wants only objective
        return -logL
    end
end

function fit_templates_mdf(models::AbstractVector{T},
                           data::AbstractMatrix{<:Number},
                           logAge::AbstractVector{<:Number},
                           metallicities::AbstractVector{<:Number},
                           σ::Number;
                           composite=Matrix{S}(undef,size(data)),
                           x0=vcat(construct_x0_mdf(logAge), [-0.1, -0.5]),
                           kws...) where {S <: Number, T <: AbstractMatrix{S}}
    unique_logage = unique(logAge)
    @assert length(x0) == length(unique_logage)+2
    # Perform logarithmic transformation on the provided x0 for all variables except α and β
    for i in eachindex(x0)[begin:end-2]
        x0[i] = log(x0[i])
    end
    # Make scratch array for assessing transformations
    x = similar(x0)
    # Define wrapper function to pass to Optim.only_fg!
    # It looks like you don't need the Jacobian correction to arrive at the maximum likelihood
    # result, and if you remove the Jacobian corrections it actually converges to the non-log-transformed case.
    # However, the uncertainty estimates from the inverse Hessian don't seem reliable without the
    # Jacobian corrections.
    function fg_mdf_fixedσ!_map(F, G, xvec)
        for i in eachindex(xvec)[begin:end-2] # These are the per-logage stellar mass coefficients
            x[i] = exp(xvec[i])
        end
        x[end-1] = xvec[end-1]  # α
        x[end] = xvec[end]      # β

        logL = fg_mdf_fixedσ!(F, G, x, models, data, composite, logAge, metallicities, σ)
        logL -= sum( @view xvec[begin:end-2] ) # this is the Jacobian correction
        # Add the Jacobian correction for every element of G except α (x[end-2]) and β (x[end-1])
        for i in eachindex(G)[begin:end-2]
            G[i] = G[i] * x[i] - 1
        end
        return logL
    end
    function fg_mdf_fixedσ!_mle(F, G, xvec)
        for i in eachindex(xvec)[begin:end-2] # These are the per-logage stellar mass coefficients
            x[i] = exp(xvec[i])
        end
        x[end-1] = xvec[end-1]  # α
        x[end] = xvec[end]      # β

        logL = fg_mdf_fixedσ!(F, G, x, models, data, composite, logAge, metallicities, σ)
        for i in eachindex(G)[begin:end-2]
            G[i] = G[i] * x[i]
        end
        return logL
    end
    # The InitialStatic(1.0,true) alphaguess helps to regularize the optimization and 
    # makes it less sensitive to initial x0.
    bfgs_struct = Optim.BFGS(; alphaguess=LineSearches.InitialStatic(1.0,true), linesearch=LineSearches.HagerZhang())
    # The extended trace will contain the BFGS estimate of the inverse Hessian, aka the
    # covariance matrix, which we can use to make parameter uncertainty estimates
    bfgs_options = Optim.Options(; allow_f_increases=true, store_trace=true, extended_trace=true, kws...)
    # Calculate results
    result_map = Optim.optimize(Optim.only_fg!( fg_mdf_fixedσ!_map ), x0, bfgs_struct, bfgs_options)
    result_mle = Optim.optimize(Optim.only_fg!( fg_mdf_fixedσ!_mle ), Optim.minimizer(result_map), bfgs_struct, bfgs_options)
    # Transform the resulting variables
    μ_map = deepcopy( Optim.minimizer(result_map) )
    μ_mle = deepcopy( Optim.minimizer(result_mle) )
    for i in eachindex(μ_map,μ_mle)[begin:end-2]
        μ_map[i] = exp(μ_map[i])
        μ_mle[i] = exp(μ_mle[i])
    end
    # Estimate parameter uncertainties from the inverse Hessian approximation
    σ_map = sqrt.(diag(Optim.trace(result_map)[end].metadata["~inv(H)"]))
    σ_mle = sqrt.(diag(Optim.trace(result_mle)[end].metadata["~inv(H)"]))
    # Need to account for the logarithmic transformation
    for i in eachindex(σ_map,σ_mle)[begin:end-2]
        σ_map[i] = μ_map[i] * σ_map[i]
        σ_mle[i] = μ_mle[i] * σ_mle[i]
    end
    return (map = (μ = μ_map, σ = σ_map, result = result_map),
            mle = (μ = μ_mle, σ = σ_mle, result = result_mle))
end

"""
This version of mdf fg! also fits σ
"""
@inline function fg_mdf!(F, G, variables::AbstractVector{<:Number}, models::AbstractVector{T}, data::AbstractMatrix{<:Number}, composite::AbstractMatrix{<:Number}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}) where T <: AbstractMatrix{<:Number}
    # `variables` should have length `length(unique(logAge)) + 2`; coeffs for each unique
    # entry in logAge, plus α and β to define the MDF at fixed logAge
    @assert axes(data) == axes(composite)
    S = promote_type(eltype(variables), eltype(eltype(models)), eltype(eltype(data)), eltype(composite), eltype(logAge), eltype(metallicities))
    # Compute the coefficients on each model template given the `variables` and the MDF
    α, β, σ = variables[end-2], variables[end-1], variables[end]
    unique_logAge = unique(logAge)
    # Calculate the per-template coefficents and normalization values
    coeffs = calculate_coeffs_mdf(view(variables,firstindex(variables):lastindex(variables)-3), logAge, metallicities, α, β, σ)

    # Fill the composite array with the equivalent of sum( coeffs .* models )
    # composite = sum( coeffs .* models )
    # return -loglikelihood(composite, data)
    composite!(composite, coeffs, models)
    logL = loglikelihood(composite, data) # Need to do this before ∇loglikelihood! because it will overwrite composite
    if (F != nothing) & (G != nothing) # Optim.optimize wants objective and gradient
        @assert axes(G) == axes(variables)
        # Calculate the ∇loglikelihood with respect to model coefficients; we will need all of these
        # fullG = [ ∇loglikelihood(models[i], composite, data) for i in axes(models,1) ]
        fullG = Vector{eltype(G)}(undef,length(models))
        ∇loglikelihood!(fullG, composite, models, data)
        # Now need to do the transformation to the `variables` rather than model coefficients
        G[end-2] = zero(eltype(G))
        G[end-1] = zero(eltype(G))
        G[end] = zero(eltype(G))
        for i in axes(G,1)[begin:end-3] 
            la = unique_logAge[i]
            age = exp10(la) / 1e9 # the / 1e9 makes α the slope in MH/Gyr, improves convergence
            μ = α * age + β # Find the mean metallicity of this age bin
            idxs = findall( ==(la), logAge) # Find all entries that match this logAge
            tmp_coeffs = [_gausspdf(metallicities[j], μ, σ) for j in idxs] # Calculate relative weights
            A = sum(tmp_coeffs)
            # This should be correct for any MDF model at fixed logAge
            @inbounds G[i] = -sum( fullG[j] * coeffs[j] / variables[i] for j in idxs )
            βsum = sum( ((metallicities[idxs[j]]-μ) * tmp_coeffs[j]) for j in eachindex(idxs))
            dLdβ = -sum( fullG[idxs[j]] * variables[i] *
                ( ((metallicities[idxs[j]]-μ) * tmp_coeffs[j]) - tmp_coeffs[j] / A * βsum )
                         for j in eachindex(idxs)) / A / σ^2
            dLdα = dLdβ * age
            σsum = sum( tmp_coeffs[j] *
                (metallicities[idxs[j]]-μ)^2 / σ^3 for j in eachindex(idxs))
            dLdσ = -sum( fullG[idxs[j]] * variables[i] *
                (tmp_coeffs[j] * (metallicities[idxs[j]]-μ)^2 / σ^3 -
                tmp_coeffs[j] / A * σsum) for j in eachindex(idxs)) / A
            G[end-2] += dLdα
            G[end-1] += dLdβ
            G[end] += dLdσ
        end
        return -logL
    elseif F != nothing # Optim.optimize wants only objective
        return -logL
    end
end

"""
    fit_templates_mdf(models::AbstractVector{T},
                      data::AbstractMatrix{<:Number},
                      logAge::AbstractVector{<:Number},
                      metallicities::AbstractVector{<:Number} [, σ::Number];
                      composite=Matrix{S}(undef,size(data)),
                      x0=vcat(construct_x0_mdf(logAge), [-0.1, -0.5, 0.3]),
                      kws...) where {S <: Number, T <: AbstractMatrix{S}}

Method that fits a linear combination of the provided Hess diagrams `models` to the observed Hess diagram `data`, constrained to have a linear mean metallicity evolution with the mean metallicity of element `i` of `unique(logAge)` being `μ[i] = α * exp10(unique(logAge)[i]) / 1e9 + β`. `α` is therefore a slope in the units of `metallicities` per Gyr, and `β` is the mean metallicity value of stars being born at present-day. Individual weights per each isochrone are then determined via Gaussian weighting with the above mean and the standard deviation `σ`, which can either be fixed or fit.

This function is designed to work best with a "grid" of stellar models, defined by the outer product of `N` unique entries in `logAge` and `M` unique entries in `metallicities`. See the examples for more information on usage. 

# Arguments
 - `models::AbstractVector{<:AbstractMatrix{<:Number}}` is a vector of equal-sized matrices that represent the template Hess diagrams for the simple stellar populations that compose the observed Hess diagram.
 - `data::AbstractMatrix{<:Number}` is the Hess diagram for the observed data.
 - `logAge::AbstractVector{<:Number}` is the vector containing the effective ages of the stellar populations used to create the templates in `models`, in units of `log10(age [yr])`. For example, if a population has an age of 1 Myr, its entry in `logAge` should be `log10(10^6) = 6.0`.
 - `metallicities::AbstractVector{<:Number}` is the vector containing the effective metallicities of the stellar populations used to create the templates in `models`. This is most commonly a logarithmic abundance like [M/H] or [Fe/H], but you could use a linear abundance like the metal mass fraction Z if you wanted to. There are some notes on the [Wikipedia](https://en.wikipedia.org/wiki/Metallicity) that might be useful. 

# Optional Arguments
 - If provided, `σ::Number` is the fixed width of the Gaussian the defines the metallicity distribution function (MDF) at fixed `logAge`. If this argument is omitted, `σ` will be a free parameter in the fit. 

# Keyword Arguments
 - `composite` is the working matrix that will be used to store the composite Hess diagram model during computation; must be of the same size as the templates contained in `models` and the observed Hess diagram `data`.
 - `x0` is the vector of initial guesses for the stellar mass coefficients per *unique* entry in `logAge`, plus the variables that define the metallicity evolution model. You should basically always be calculating and passing this keyword argument. We provide [`StarFormationHistories.construct_x0_mdf`](@ref) to prepare the first part of `x0` assuming constant star formation rate, which is typically a good initial guess. You then have to concatenate that result with an initial guess for the metallicity evolution parameters. For example, `x0=vcat(construct_x0_mdf(logAge; normalize_value=1e4), [-0.1,-0.5,0.3])`, where `logAge` is a valid argument for this function (see above), and the initial guesses on the parameters are `[α, β, σ] = [-0.1, -0.5, 0.3]`. If the provided `metallicities` are, for example, [M/H] values, then this mean metallicity evolution is μ(t) [dex] = -0.1 [dex/Gyr] * t [Gyr] - 0.5 [dex], and at fixed time, the metallicity distribution function is Gaussian with mean μ(t) and standard deviation σ. If you provide `σ` as an optional argument, then you should not include an entry for it in `x0`. 
Other `kws...` are passed to `Optim.options` to set things like convergence criteria for the optimization.

# Returns
 - This function returns an object of identical structure to the object returned by [`fit_templates`](@ref); please refer to that documentation. `α` and `β` are not optimized under a logarithmic transform, but `σ` is since it must be positive. 

# Notes
 - This method also uses the `BFGS` method from `Optim.jl` internally just like [`fit_templates`](@ref); please see the notes section of that method. 
"""
function fit_templates_mdf(models::AbstractVector{T},
                           data::AbstractMatrix{<:Number},
                           logAge::AbstractVector{<:Number},
                           metallicities::AbstractVector{<:Number};
                           composite=Matrix{S}(undef,size(data)),
                           x0=vcat(construct_x0_mdf(logAge), [-0.1, -0.5, 0.3]),
                           kws...) where {S <: Number, T <: AbstractMatrix{S}}
    unique_logage = unique(logAge)
    @assert length(x0) == length(unique_logage)+3
    # Perform logarithmic transformation on the provided x0 for all variables except α and β
    for i in eachindex(x0)[begin:end-3]
        x0[i] = log(x0[i])
    end
    x0[end] = log(x0[end])
    # Make scratch array for assessing transformations
    x = similar(x0)
    # Define wrapper function to pass to Optim.only_fg!
    # It looks like you don't need the Jacobian correction to arrive at the maximum likelihood
    # result, and if you remove the Jacobian corrections it actually converges to the non-log-transformed case.
    # However, the uncertainty estimates from the inverse Hessian don't seem reliable without the
    # Jacobian corrections.
    function fg_mdf!_map(F, G, xvec)
        for i in eachindex(xvec)[begin:end-3] # These are the per-logage stellar mass coefficients
            x[i] = exp(xvec[i])
        end
        x[end-2] = xvec[end-2]  # α
        x[end-1] = xvec[end-1]  # β
        x[end] = exp(xvec[end]) # σ
        logL = fg_mdf!(F, G, x, models, data, composite, logAge, metallicities)
        logL -= sum( @view xvec[begin:end-3] ) + xvec[end] # This is the Jacobian correction
        # Add the Jacobian correction for every element of G except α (x[end-2]) and β (x[end-1])
        for i in eachindex(G)[begin:end-3]
            G[i] = G[i] * x[i] - 1
        end
        G[end] = G[end] * x[end] - 1
        return logL
    end
    function fg_mdf!_mle(F, G, xvec)
        for i in eachindex(xvec)[begin:end-3] # These are the per-logage stellar mass coefficients
            x[i] = exp(xvec[i])
        end
        x[end-2] = xvec[end-2]  # α
        x[end-1] = xvec[end-1]  # β
        x[end] = exp(xvec[end]) # σ
        logL = fg_mdf!(F, G, x, models, data, composite, logAge, metallicities)
        for i in eachindex(G)[begin:end-3]
            G[i] = G[i] * x[i]
        end
        G[end] = G[end] * x[end]
        return logL
    end
    # Set up options for the optimization
    # The InitialStatic(1.0,true) alphaguess helps to regularize the optimization and 
    # makes it less sensitive to initial x0.
    bfgs_struct = Optim.BFGS(; alphaguess=LineSearches.InitialStatic(1.0,true),
                             linesearch=LineSearches.HagerZhang())
    # The extended trace will contain the BFGS estimate of the inverse Hessian, aka the
    # covariance matrix, which we can use to make parameter uncertainty estimates
    bfgs_options = Optim.Options(; allow_f_increases=true, store_trace=true, extended_trace=true, kws...)
    # Calculate result
    result_map = Optim.optimize(Optim.only_fg!( fg_mdf!_map ), x0, bfgs_struct, bfgs_options)
    result_mle = Optim.optimize(Optim.only_fg!( fg_mdf!_mle ), Optim.minimizer(result_map), bfgs_struct, bfgs_options)
    # Transform the resulting variables
    μ_map = deepcopy( Optim.minimizer(result_map) )
    μ_mle = deepcopy( Optim.minimizer(result_mle) )
    for i in eachindex(μ_map,μ_mle)[begin:end-3]
        μ_map[i] = exp(μ_map[i])
        μ_mle[i] = exp(μ_mle[i])
    end
    μ_map[end] = exp(μ_map[end])
    μ_mle[end] = exp(μ_mle[end])

    # Estimate parameter uncertainties from the inverse Hessian approximation
    σ_map = sqrt.(diag(Optim.trace(result_map)[end].metadata["~inv(H)"]))
    σ_mle = sqrt.(diag(Optim.trace(result_mle)[end].metadata["~inv(H)"]))
    # Need to account for the logarithmic transformation
    for i in eachindex(σ_map,σ_mle)[begin:end-3]
        σ_map[i] = μ_map[i] * σ_map[i]
        σ_mle[i] = μ_mle[i] * σ_mle[i]
    end
    σ_map[end] = μ_map[end] * σ_map[end]
    σ_mle[end] = μ_mle[end] * σ_mle[end]
    return (map = (μ = μ_map, σ = σ_map, result = result_map),
            mle = (μ = μ_mle, σ = σ_mle, result = result_mle))
end

# We can even use the inv(H) = covariance matrix estimate to draw samples to compare to HMC
# import Distributions: MvNormal
# result1, std1, fr = fit_templates_mdf(mdf_templates, h.weights, mdf_template_logAge, mdf_template_MH; x0=vcat(construct_x0_mdf(mdf_template_logAge; normalize_value=1e4),[-0.1,-0.5,0.3]))
# corner.corner(permutedims(rand(MvNormal(result1,LinearAlgebra.Hermitian(fr.trace[end].metadata["~inv(H)"])),10000)[end-2:end,:]))
# Can we also use this inv(H) estimate as input to HMC? I think that's roughly the M matrix.

######################################################################################
## HMC with MDF

struct HMCModelMDF{T,S,V,W,X,Y}
    models::T
    composite::S
    data::V
    logAge::W
    metallicities::X
    G::Y
end

# This model will return loglikelihood and gradient
LogDensityProblems.capabilities(::Type{<:HMCModelMDF}) = LogDensityProblems.LogDensityOrder{1}()
LogDensityProblems.dimension(problem::HMCModelMDF) = length(problem.G)

function LogDensityProblems.logdensity_and_gradient(problem::HMCModelMDF, xvec)
    composite = problem.composite
    models = problem.models
    data = problem.data
    dims = length(models)
    logAge = problem.logAge
    metallicities = problem.metallicities
    G = problem.G
    @assert axes(G) == axes(xvec)
    # Transform the provided x
    # α and β, which are xvec[end-2] and xvec[end-1], are the only variables that are not log-transformed
    x = similar(xvec)
    for i in eachindex(xvec)[begin:end-3] # These are the per-logage stellar mass coefficients
        x[i] = exp(xvec[i])
    end
    x[end-2] = xvec[end-2]  # α
    x[end-1] = xvec[end-1]  # β
    x[end] = exp(xvec[end]) # σ

    # fg_mdf! returns -logL and fills G with -∇logL so we need to negate the signs.
    logL = -fg_mdf!(true, G, x, models, data, composite, logAge, metallicities)
    logL += sum(view(xvec,firstindex(xvec):lastindex(xvec)-3)) + xvec[end] # this is the Jacobian correction
    ∇logL = -G
    # Add the Jacobian correction for every element of ∇logL except α (x[end-2]) and β (x[end-1])
    for i in eachindex(∇logL)[begin:end-3]
        ∇logL[i] = ∇logL[i] * x[i] + 1
    end
    ∇logL[end] = ∇logL[end] * x[end] + 1
    return logL, ∇logL
end

# Version with just one chain; no threading
function hmc_sample_mdf(models::AbstractVector{T}, data::AbstractMatrix{<:Number}, logAge::AbstractVector{<:Number}, metallicities::AbstractVector{<:Number}, nsteps::Integer; composite=Matrix{S}(undef,size(data)), rng::AbstractRNG=default_rng(), kws...) where {S <: Number, T <: AbstractMatrix{S}}
    @assert length(logAge) == length(metallicities)
    instance = HMCModelMDF( models, composite, data, logAge, metallicities,
                            Vector{S}(undef, length(unique(logAge)) + 3) )
    return DynamicHMC.mcmc_with_warmup(rng, instance, nsteps; kws...)
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
