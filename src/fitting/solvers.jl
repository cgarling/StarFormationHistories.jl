# Methods dealing with direct MLE/MAP solvers

# Input validation methods
@inline function _check_matrix_input_sizes(coeffs, models, data, composite)
    # @argcheck size(composite) == size(data)
    @argcheck length(coeffs) == length(models)
    @argcheck all(size(model) == size(data) == size(composite) for model in models)
end
@inline function _check_flat_input_sizes(coeffs, models, data, composite)
    @argcheck axes(coeffs,1) == axes(models,2)
    @argcheck axes(models,1) == axes(data,1) == axes(composite,1)
end

"""
    -logL = fg!(F, G, coeffs::AbstractVector{<:Number}, models::AbstractVector{T}, data::AbstractMatrix{<:Number}, composite::AbstractMatrix{<:Number}) where T <: AbstractMatrix{<:Number}
    -logL = fg!(F, G, coeffs::AbstractVector{<:Number}, models::AbstractMatrix{<:Number}, data::AbstractVector{<:Number}, composite::AbstractVector{<:Number})

Computes -loglikelihood and its gradient simultaneously for use with Optim.jl and other optimization APIs. See documentation [here](https://julianlsolvers.github.io/Optim.jl/stable/#user/tipsandtricks/). Second call signature supports the flattened formats for `models` and `data`. See the notes for the flattened call signature of [`StarFormationHistories.composite!`](@ref) for more details.
"""
@inline function fg!(F, G, coeffs, models, data, composite)
# @inline function fg!(F, G, coeffs::AbstractVector{<:Number}, models::AbstractVector{T}, data::AbstractMatrix{<:Number}, composite::AbstractMatrix{<:Number}) where T <: AbstractMatrix{<:Number}
# @inline function fg!(F, G, coeffs::AbstractVector{<:Number}, models::AbstractMatrix{<:Number}, data::AbstractVector{<:Number}, composite::AbstractVector{<:Number})
    S = promote_type(eltype(coeffs), eltype(eltype(models)), eltype(data), eltype(composite))
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
    (-logL, coeffs) = 
    fit_templates_lbfgsb(models::AbstractVector{T},
                         data::AbstractMatrix{<:Number};
                         x0::AbstractVector{<:Number} = ones(S,length(models)),
                         factr::Number=1e-12,
                         pgtol::Number=1e-5,
                         iprint::Integer=0,
                         kws...) where {S <: Number, T <: AbstractMatrix{S}}

Finds the coefficients `coeffs` that maximize the Poisson likelihood ratio (equations 7--10 in [Dolphin 2002](https://ui.adsabs.harvard.edu/abs/2002MNRAS.332...91D)) for the composite Hess diagram model `sum(models .* coeffs)` given the provided templates `models` and the observed Hess diagram `data` using the box-constrained LBFGS method provided by [LBFGSB.jl](https://github.com/Gnimuc/LBFGSB.jl). 

# Arguments
 - `models::AbstractVector{AbstractMatrix{<:Number}}`: the list of template Hess diagrams for the simple stellar populations (SSPs) being considered; all must have the same size.
 - `data::AbstractMatrix{<:Number}`: the observed Hess diagram; must match the size of the templates contained in `models`.

# Keyword Arguments
 - `x0`: The vector of initial guesses for the stellar mass coefficients. You should basically always be calculating and passing this keyword argument; we provide [`StarFormationHistories.construct_x0`](@ref) to prepare `x0` assuming constant star formation rate, which is typically a good initial guess.
 - `factr::Number`: Keyword argument passed to `LBFGSB.lbfgsb`; essentially a relative tolerance for convergence based on the inter-iteration change in the objective function.
 - `pgtol::Number`: Keyword argument passed to `LBFGSB.lbfgsb`; essentially a relative tolerance for convergence based on the inter-iteration change in the projected gradient of the objective.
 - `iprint::Integer`: Keyword argument passed to `LBFGSB.lbfgsb` controlling how much information is printed to the terminal. Setting to `1` can sometimes be helpful to diagnose convergence issues. Setting to `-1` will disable printing.
Other `kws...` are passed to `LBFGSB.lbfgsb`.

# Returns
 - `-logL::Number`: the minimum negative log-likelihood found by the optimizer.
 - `coeffs::Vector{<:Number}`: the maximum likelihood estimate for the coefficient vector. 

# Notes
 - It can be helpful to normalize your `models` to contain realistic total stellar masses to aid convergence stability; for example, if the total stellar mass of your population is 10^7 solar masses, then you might normalize your templates to contain 10^3 solar masses. If you are using [`partial_cmd_smooth`](@ref) to construct the templates, you can specify this normalization via the `normalize_value` keyword. 
"""
fit_templates_lbfgsb(models::AbstractVector{<:AbstractMatrix{<:Number}}, data::AbstractMatrix{<:Number}; kws...) = fit_templates_lbfgsb(stack_models(models), vec(data); kws...) # Calls to method below
"
    fit_templates_lbfgsb(models::AbstractMatrix{S},
                         data::AbstractVector{<:Number};
                         x0::AbstractVector{<:Number} = ones(S,size(models,2)),
                         factr::Number=1e-12,
                         pgtol::Number=1e-5,
                         iprint::Integer=0,
                         kws...) where S <: Number

This call signature supports the flattened formats for `models` and `data`. See the notes for the flattened call signature of [`StarFormationHistories.composite!`](@ref) for more details.
"
function fit_templates_lbfgsb(models::AbstractMatrix{S}, data::AbstractVector{<:Number}; x0::AbstractVector{<:Number}=ones(S,size(models,2)), factr::Number=1e-12, pgtol::Number=1e-5, iprint::Integer=0, kws...) where S <: Number
    composite = Vector{S}(undef,length(data))
    _check_flat_input_sizes(x0, models, data, composite) # Validate input sizes
    G = similar(x0)
    fg(x) = (R = fg!(true,G,x,models,data,composite); return R,G)
    LBFGSB.lbfgsb(fg, x0; lb=zeros(size(models,2)), ub=fill(Inf,size(models,2)), factr=factr, pgtol=pgtol, iprint=iprint, kws...)
end

"""
    LogTransformFTResult(μ::AbstractVector{<:Number},
                         σ::AbstractVector{<:Number},
                         invH::AbstractMatrix{<:Number},
                         result)

Type for containing the maximum likelihood estimate (MLE) and maximum a posteriori (MAP) results from [`fit_templates`](@ref). The fitted coefficients are available in the `μ` field. Estimates of the standard errors are available in the `σ` field. These have both been transformed from the native logarithmic fitting space into natural units (i.e., stellar mass or star formation rate).

`invH` contains the estimated inverse Hessian of the likelihood / posterior at the maximum point in the logarithmic fitting units. `result` is the full result object returned by the optimization routine.

This type is implemented as a subtype of `Distributions.Sampleable{Multivariate, Continuous}` to enable sampling from an estimate of the likelihood / posterior distribution. We approximate the distribution as a multivariate Gaussian in the native (logarithmically transformed) fitting variables with covariance matrix `invH` and means `log.(μ)`. We find this approximation is good for the MAP result but less robust for the MLE. You can obtain `N::Integer` samples from the distribution by `rand(R, N)` where `R` is an instance of this type; this will return a size `length(μ) x N` matrix, or fail if `invH` is not positive definite.

# Examples
```julia-repl
julia> result = fit_templates(models, data);

julia> typeof(result.map)
StarFormationHistories.LogTransformFTResult{...}

julia> size(rand(result.map, 3)) == (length(models),3)
true
```
"""
struct LogTransformFTResult{S <: AbstractVector{<:Number},
                            T <: AbstractVector{<:Number},
                            U <: AbstractMatrix{<:Number},
                            V} <: Sampleable{Multivariate, Continuous}
    μ::S
    σ::T
    invH::U
    result::V
end
Base.length(result::LogTransformFTResult) = length(result.μ)
function _rand!(rng::AbstractRNG, result::LogTransformFTResult, x::Union{AbstractVector{T}, DenseMatrix{T}}) where T <: Real
    dist = MvNormal(Optim.minimizer(result.result), Hermitian(result.invH))
    _rand!(rng, dist, x)
    @. x = exp(x)
end

"""
    result = fit_templates(models::AbstractVector{T},
                           data::AbstractMatrix{<:Number};
                           x0::AbstractVector{<:Number} = ones(S,length(models)),
                           kws...) where {S <: Number, T <: AbstractMatrix{S}}

Finds both maximum likelihood estimate (MLE) and maximum a posteriori estimate (MAP) for the coefficients `coeffs` such that the composite Hess diagram model is `sum(models .* coeffs)` using the provided templates `models` and the observed Hess diagram `data`. Utilizes the Poisson likelihood ratio (equations 7--10 in [Dolphin 2002](https://ui.adsabs.harvard.edu/abs/2002MNRAS.332...91D)) for the likelihood of the data given the model. See the examples in the documentation for comparisons of the results of this method and [`hmc_sample`](@ref) which samples the posterior via Hamiltonian Monte Carlo. 

# Arguments
 - `models::AbstractVector{AbstractMatrix{<:Number}}`: the list of template Hess diagrams for the simple stellar populations (SSPs) being considered; all must have the same size.
 - `data::AbstractMatrix{<:Number}`: the observed Hess diagram; must match the size of the templates contained in `models`.

# Keyword Arguments
 - `x0`: The vector of initial guesses for the stellar mass coefficients. You should basically always be calculating and passing this keyword argument; we provide [`StarFormationHistories.construct_x0`](@ref) to prepare `x0` assuming constant star formation rate, which is typically a good initial guess.
Other `kws...` are passed to `Optim.options` to set things like convergence criteria for the optimization. 

# Returns
`result` is a `NamedTuple` containing two [`StarFormationHistories.LogTransformFTResult`](@ref). The two components of `result` are `result.map` or `result[1]`, which contains the results of the MAP optimization, and `result.mle` or `result[2]`, which contains the results of the MLE optimization. The documentation for [`StarFormationHistories.LogTransformFTResult`](@ref) contains more information about these types, but briefly they contain the following fields, accessible as, e.g., `result.map.μ`, `result.map.σ`, etc.:
 - `μ::Vector{<:Number}` are the optimized `coeffs` at the maximum.
 - `σ::Vector{<:Number}` are the standard errors on the coeffs `μ` calculated from an estimate of the inverse Hessian matrix evaluated at `μ`. The inverse of the Hessian matrix at the maximum of the likelihood (or posterior) is a estimator for the variance-covariance matrix of the parameters, but is only accurate when the second-order expansion given by the Hessian at the maximum is a good approximation to the function being optimized (i.e., when the optimized function is approximately quadratic around the maximum; see [Dovi et al. 1991](https://doi.org/10.1016/0893-9659(91)90129-J) for more information). We find this is often the case for the MAP estimate, but the errors found for coefficients that are ≈0 in the MLE are typically unrealistically small. For coefficients significantly greater than 0, the `σ` values from the MAP and MLE are typically consistent to 5--10%.
 - `invH::Matrix{<:Number}` is the estimate of the inverse Hessian matrix at `μ` that was used to derive `σ`. The optimization is done under a logarithmic transform, such that `θ[j] = log(coeffs[j])` are the actual parameters optimized, so the entries in the Hessian are actually
```math
H^{(j,k)} ( \\boldsymbol{\\hat \\theta} ) = \\left. \\frac{\\partial^2 \\, J(\\boldsymbol{\\theta})}{\\partial \\theta_j \\, \\partial \\theta_k} \\right\\vert_{\\boldsymbol{\\theta}=\\boldsymbol{\\hat \\theta}}
```
 - `result` is the full object returned by the optimization from `Optim.jl`; this is of type `Optim.MultivariateOptimizationResults`. Remember that the optimization is done with parameters `θ[j] = log(coeffs[j])` when dealing with this raw output. This means that, for example, we calculate `result.map.μ` as `exp.(Optim.minimizer(result.map.result))`.

The special property of the [`StarFormationHistories.LogTransformFTResult`](@ref) type is that you can draw a set of `N::Integer` random parameter samples from the result using the inverse Hessian approximation discussed above by doing `rand(result.map, N)`. This type implements the random sampling API of `Distributions.jl` so the other standard sampling methods should work as well. In our tests these samples compare very favorably against those from [`hmc_sample`](@ref), which samples the posterior via Hamiltonian Monte Carlo and is therefore more robust but much more expensive. We compare these methods in the examples.

# Notes
 - This method uses the `BFGS` method from `Optim.jl` internally because it builds and tracks the inverse Hessian matrix approximation which can be used to estimate parameter uncertainties. BFGS is much more memory-intensive than LBFGS (as used for [`StarFormationHistories.fit_templates_lbfgsb`](@ref)) for large numbers of parameters (equivalently, many `models`), so you should consider LBFGS to solve for the MLE along with [`hmc_sample`](@ref) to sample the posterior if you are using a large grid of models (greater than a few hundred).
 - The BFGS implementation we use from Optim.jl uses BLAS operations during its iteration. The OpenBLAS that Julia ships with will often default to running on multiple threads even if Julia itself is started with only a single thread. You can check the current number of BLAS threads with `import LinearAlgebra: BLAS; BLAS.get_num_threads()`. For the problem sizes typical of this function we actually see performance regression with larger numbers of BLAS threads. For this reason you may wish to use BLAS in single-threaded mode; you can set this as `import LinearAlgebra: BLAS; BLAS.set_num_threads(1)`.
"""
@inline fit_templates(models::AbstractVector{<:AbstractMatrix{<:Number}}, data::AbstractMatrix{<:Number}; kws...) = fit_templates(stack_models(models), vec(data); kws...) # Calls to method below
"""
    fit_templates(models::AbstractMatrix{S},
                  data::AbstractVector{<:Number};
                  x0::AbstractVector{<:Number} = ones(S,length(models)),
                  kws...) where S <: Number

This call signature supports the flattened formats for `models` and `data`. See the notes for the flattened call signature of [`StarFormationHistories.composite!`](@ref) for more details.
"""
function fit_templates(models::AbstractMatrix{S}, data::AbstractVector{<:Number}; x0::AbstractVector{<:Number}=ones(S,size(models,2)), kws...) where S <: Number
    composite = Vector{S}(undef,length(data))
    _check_flat_input_sizes(x0, models, data, composite) # Validate input sizes
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
    # NamedTuple of LogTransformFTResult. "map" contains results for the maximum a posteriori estimate.
    # "mle" contains the same entries but for the maximum likelihood estimate.
    return  ( map = LogTransformFTResult(exp.(Optim.minimizer(result_map)),
                                         sqrt.(diag(Optim.trace(result_map)[end].metadata["~inv(H)"])) .*
                                           exp.(Optim.minimizer(result_map)),
                                         result_map.trace[end].metadata["~inv(H)"],
                                         result_map ),
              mle = LogTransformFTResult(exp.(Optim.minimizer(result_mle)),
                                         sqrt.(diag(Optim.trace(result_mle)[end].metadata["~inv(H)"])) .*
                                           exp.(Optim.minimizer(result_mle)),
                                         result_mle.trace[end].metadata["~inv(H)"],
                                         result_mle ) )
end

"""
    (coeffs::Vector{::eltype(x0)}, result::Optim.MultivariateOptimizationResults) =
    fit_templates_fast(models::AbstractVector{T},
                       data::AbstractMatrix{<:Number};
                       x0::AbstractVector{<:Number} = ones(S,length(models)),
                       kws...)
                       where {S <: Number, T <: AbstractMatrix{S}}

Finds *only* the maximum likelihood estimate (MLE) for the coefficients `coeffs` given the provided `data` such that the best-fit composite Hess diagram model is `sum(models .* coeffs)`. This is a simplification of the main [`fit_templates`](@ref) function, which will return the MLE as well as the maximum a posteriori estimate, and further supports uncertainty quantification. For additional details on arguments to this method, see the documentation for [`fit_templates`](@ref). 

This method optimizes parameters `θ` such that `coeffs = θ.^2` -- this allows for faster convergence than both the [`fit_templates_lbfgsb`](@ref) method, which does not use a variable transformation, and the logarithmic transformation used in [`fit_templates`](@ref). However, the inverse Hessian is not useful for uncertainty estimation under this transformation. As such this method only returns the MLE for `coeffs` as a vector and the result object returned by `Optim.optimize`. While this method offers fewer features than [`fit_templates`](@ref), this method's runtime is typically half as long (or less). As such, this method is recommended for use in performance-sensitive applications like hierarchical models or hyperparameter estimation where the additional features of [`fit_templates`](@ref) are unnecessary. In these applications, the value of the objective function at the derived MLE is typically desired as well; this can be obtained the from `result::Optim.MultivariateOptimizationResults` object as `Optim.minimum(result)`. Note that this will return the *negative* loglikelihood, which is what is minimized in this application.

# Notes
 1. By passing additional convergence keyword arguments supported by `Optim.Options` (see [this guide](https://julianlsolvers.github.io/Optim.jl/stable/#user/config/)), it is possible to converge to the MLE in fewer than 30 iterations with fewer than 100 calls to the likelihood and gradient methods. For example, the main convergence criterion is typically the magnitude of the gradient vector, which by default is `g_abstol=1e-8`, terminating the optimization when the magnitude of the gradient is less than 1e-8. We find results are typically sufficiently accurate with `g_abstol=1e-3`, which often uses half as many objective evaluations as the default value.
"""
fit_templates_fast(models::AbstractVector{<:AbstractMatrix{<:Number}}, data::AbstractMatrix{<:Number}; kws...) = fit_templates_fast(stack_models(models), vec(data); kws...) # Calls to method below
"""
    fit_templates_fast(models::AbstractMatrix{S},
                       data::AbstractVector{<:Number};
                       x0::AbstractVector{<:Number} = ones(S,size(models,2)),
                       kws...)
                       where S <: Number

This call signature supports the flattened formats for `models` and `data`. See the notes for the flattened call signature of [`StarFormationHistories.composite!`](@ref) for more details.
"""
@inline function fit_templates_fast(models::AbstractMatrix{S}, data::AbstractVector{<:Number}; x0::AbstractVector{<:Number}=ones(S,size(models,2)), kws...) where S <: Number
    composite = Vector{S}(undef,length(data))
    _check_flat_input_sizes(x0, models, data, composite) # Validate input sizes
    # Transform the initial guess vector
    x0 = sqrt.(x0)
    # Make scratch array for assessing transformations
    x = similar(x0)
    function fg_mle!(F,G,sqrtx)
        @. x = sqrtx^2
        logL = fg!(true,G,x,models,data,composite)
        if G != nothing
            @. G = G * 2 * sqrtx # Correct for transform
        end
        return logL
    end
    # Setup for Optim.jl
    # The InitialStatic(1.0,true) alphaguess helps to regularize the optimization and 
    # makes it less sensitive to initial x0.
    bfgs_struct = Optim.BFGS(; alphaguess=LineSearches.InitialStatic(1.0,true),
                               linesearch=LineSearches.HagerZhang())
    # We don't need to save the trace of the optimization here
    bfgs_options = Optim.Options(; allow_f_increases=true, kws...)
    # Calculate result
    result_mle = Optim.optimize(Optim.only_fg!( fg_mle! ), x0, bfgs_struct, bfgs_options)
    return Optim.minimizer(result_mle).^2, result_mle # Optim.minimum(result_mle)
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
