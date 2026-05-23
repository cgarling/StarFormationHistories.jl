"""
    FisherOptimizerResult{T <: Number}

Minimal result struct returned by the custom trust-region Fisher scoring optimizer
[`StarFormationHistories._trust_region_fisher_scoring`](@ref). Supports
`Optim.minimizer`, `Optim.minimum`, `Optim.converged`, and `Optim.iterations` so
that it is a drop-in replacement for an Optim result inside
[`BFGSResult`](@ref StarFormationHistories.BFGSResult).
"""
struct FisherOptimizerResult{T <: Number}
    minimizer::Vector{T}
    minimum::T
    converged::Bool
    iterations::Int
    g_norm::T
end
# Extend Optim interface so BFGSResult / CompositeBFGSResult sampling works
Optim.minimizer(r::FisherOptimizerResult) = r.minimizer
Optim.minimum(r::FisherOptimizerResult) = r.minimum
Optim.converged(r::FisherOptimizerResult) = r.converged
Optim.iterations(r::FisherOptimizerResult) = r.iterations

"""
    _tr_subproblem_step(g, H, Δ) -> (s, step_type)

Solves the trust-region subproblem

```math
\\min_{\\|s\\| \\le \\Delta} \\; g^\\top s + \\tfrac{1}{2} s^\\top H s
```

using the **Moré–Sorensen** method via eigendecomposition of `H`. Unlike the dogleg
method, this handles indefinite and highly correlated Hessians correctly by finding the
optimal regularisation parameter λ ≥ 0 such that `‖(H + λI)⁻¹ g‖ = Δ`.

Returns the step vector `s` and `:newton` (interior, λ = 0) or `:boundary` (λ > 0).
"""
function _tr_subproblem_step(g::AbstractVector{T}, H::AbstractMatrix{T}, Δ::T) where {T}
    # Eigendecomposition of the symmetric Hessian: H = Q diag(d) Q'
    F   = eigen(Symmetric(H))
    d   = F.values      # ascending eigenvalues
    Q   = F.vectors     # orthogonal eigenvectors (columns)
    Qg  = Q' * g        # gradient rotated into eigenbasis
    d_min = d[1]

    # ── Interior case ──────────────────────────────────────────────────────────
    # If H is positive-definite and the Newton step fits inside the trust region,
    # accept it directly.
    if d_min > zero(T)
        s_N = -(Q * (Qg ./ d))
        norm(s_N) ≤ Δ && return s_N, :newton
    end

    # ── Boundary case ──────────────────────────────────────────────────────────
    # Find λ ≥ max(0, -d_min) such that ‖s(λ)‖ = Δ, where
    #   s(λ) = -(H + λI)⁻¹ g  ⟺  s̃(λ) = Qg ./ (d .+ λ),  s = -Q s̃
    # ‖s(λ)‖ = ‖s̃(λ)‖ is strictly decreasing in λ on (-d_min, ∞).
    λ_lo = max(zero(T), -d_min + T(1e-8) * (abs(d_min) + one(T)))

    # Upper-bound λ such that ‖s(λ_hi)‖ < Δ.
    λ_hi = λ_lo + norm(g) / Δ + maximum(abs, d)
    for _ in 1:60
        norm(Qg ./ (d .+ λ_hi)) < Δ && break
        λ_hi *= T(2)
    end

    # Bisection (60 iterations → ~18 significant digits)
    for _ in 1:60
        λ_mid = (λ_lo + λ_hi) / 2
        if norm(Qg ./ (d .+ λ_mid)) > Δ
            λ_lo = λ_mid
        else
            λ_hi = λ_mid
        end
        abs(λ_hi - λ_lo) < T(1e-12) * (λ_hi + one(T)) && break
    end

    λ = (λ_lo + λ_hi) / 2
    return -(Q * (Qg ./ (d .+ λ))), :boundary
end

"""
    _trust_region_fisher_scoring(fgh_fn!, x0; kwargs...) -> (x, f, result)

Custom trust-region optimizer that uses the **Fisher information matrix** as the
Hessian approximation (Fisher scoring). Designed specifically for the Poisson
log-likelihood ratio used in resolved-SFH fitting, where the Fisher information is
available analytically via the [`fgh!`](@ref StarFormationHistories.fgh!) family of
functions.

`fgh_fn!(F, G, H, x)` must follow the same convention as `NLSolversBase.only_fgh!`:
  - fills `G` in-place (gradient of the objective) when `G` is not `nothing`;
  - fills `H` in-place (Fisher information / Hessian approximation) when `H` is not
    `nothing`;
  - returns the objective value when `F` is not `nothing`, otherwise returns zero.

The key difference from a generic trust-region Newton method is that the Hessian is
recomputed from the **exact** Fisher information at each accepted step, rather than
accumulated via a quasi-Newton update. This is well-suited to the Poisson problem
where the Fisher information is cheap to evaluate and provides a well-scaled curvature
estimate.

# Algorithm
1. Compute f, g, H at the current iterate via `fgh_fn!`.
2. Solve the trust-region subproblem using the **Moré–Sorensen** method
   ([`_tr_subproblem_step`](@ref StarFormationHistories._tr_subproblem_step)),
   which handles indefinite and correlated Hessians via eigendecomposition.
3. Evaluate f at the proposed iterate (without recomputing g or H).
4. Compute the reduction ratio ρ = (actual decrease) / (predicted decrease).
5. Update the trust-region radius and accept or reject the step.
6. On acceptance, recompute f, g, H at the new iterate.
7. Repeat until ‖g‖∞ ≤ `g_abstol` or `maxiter` steps are taken.

# Keyword Arguments
  - `Δ_init`: initial trust-region radius; defaults to the norm of the initial
    Newton step (clipped to [1e-3, 1.0]).
  - `Δ_max`: maximum trust-region radius (default `1e4`).
  - `η1`: minimum acceptable reduction ratio (default `0.1`).
  - `η2`: threshold above which the trust region is expanded (default `0.75`).
  - `α1`: trust-region shrink factor when ρ < η1 (default `0.25`).
  - `α2`: trust-region expansion factor when ρ > η2 (default `2.0`).
  - `maxiter`: maximum number of iterations (default `1000`).
  - `g_abstol`: convergence criterion on ‖g‖∞ (default `1e-5`).
"""
function _trust_region_fisher_scoring(fgh_fn!, x0::AbstractVector{T};
                                      Δ_init  = nothing,
                                      Δ_max   = T(1e4),
                                      η1      = T(0.1),
                                      η2      = T(0.75),
                                      α1      = T(0.25),
                                      α2      = T(2.0),
                                      maxiter = 1000,
                                      g_abstol = T(1e-5)) where {T}
    n = length(x0)
    x = copy(x0)
    g = Vector{T}(undef, n)
    H = Matrix{T}(undef, n, n)

    f = fgh_fn!(true, g, H, x)

    # Initialise the trust-region radius
    Δ = if !isnothing(Δ_init)
        T(Δ_init)
    else
        # Use the norm of the Newton step as a sensible initial radius
        # Add small regularisation only for this initial step-norm estimate
        max_diag = maximum(abs, diag(H))
        λ0 = max(T(1e-6) * max_diag, T(1e-12))
        H_init = H + Diagonal(fill(λ0, n))
        s0_norm = try
            norm(-(cholesky(Hermitian(H_init)) \ g))
        catch
            T(10.0)
        end
        clamp(s0_norm, T(1.0), T(1e4))
    end

    final_g_norm = norm(g, Inf)
    final_iter   = 0

    for iter in 1:maxiter
        final_iter   = iter
        final_g_norm = norm(g, Inf)
        if final_g_norm ≤ g_abstol
            break
        end

        # Solve the trust-region subproblem with the Moré–Sorensen method
        # (handles indefinite and correlated Hessians directly via eigendecomposition)
        s, _ = _tr_subproblem_step(g, H, Δ)

        # Evaluate the objective at the proposed step (no g or H needed)
        x_new = x .+ s
        f_new = fgh_fn!(true, nothing, nothing, x_new)

        # Predicted decrease from the quadratic model
        Hs = H * s
        predicted = -(dot(g, s) + T(0.5) * dot(s, Hs))

        # Reduction ratio
        actual = f - f_new
        ρ = if abs(predicted) < eps(T) * abs(f)
            actual ≥ 0 ? one(T) : zero(T)
        else
            actual / predicted
        end

        # Update trust-region radius
        step_norm = norm(s)
        if ρ < η1
            # Poor reduction: shrink trust region to a fraction of the actual step
            Δ = α1 * step_norm
        elseif ρ > η2 && step_norm ≥ (1 - T(1e-3)) * Δ
            # Good reduction and step touched the boundary: expand
            Δ = min(α2 * Δ, Δ_max)
        end
        # ρ ∈ [η1, η2]: leave Δ unchanged

        # Accept the step
        if ρ > η1
            x = x_new
            f = fgh_fn!(true, g, H, x)
        end

        # Safety: if Δ collapses to machine precision, stop
        Δ < T(1e-14) && break
    end

    final_g_norm = norm(g, Inf)
    converged    = final_g_norm ≤ g_abstol
    return x, f, FisherOptimizerResult(x, f, converged, final_iter, final_g_norm)
end

"""
    fit_sfh_fisher(MH_model0::AbstractMetallicityModel,
                   disp_model0::AbstractDispersionModel,
                   models::AbstractMatrix{<:Number},
                   data::AbstractVector{<:Number},
                   logAge::AbstractVector{<:Number},
                   metallicities::AbstractVector{<:Number};
                   x0::AbstractVector{<:Number} = <...>
                   kws...)
    fit_sfh_fisher(MH_model0::AbstractMetallicityModel,
                   disp_model0::AbstractDispersionModel,
                   models::AbstractVector{<:AbstractMatrix{<:Number}},
                   data::AbstractMatrix{<:Number},
                   logAge::AbstractVector{<:Number},
                   metallicities::AbstractVector{<:Number};
                   x0::AbstractVector{<:Number} = <...>
                   kws...)

Returns a [`CompositeBFGSResult`](@ref StarFormationHistories.CompositeBFGSResult) by
fitting SSP templates `models` to `data` with a custom **trust-region Fisher scoring**
(TRFS) optimizer. The interface mirrors [`fit_sfh`](@ref) exactly.

Unlike [`fit_sfh`](@ref) (L-BFGS) and [`fit_sfh_newton`](@ref) (Optim
`NewtonTrustRegion`), this method uses a fully custom trust-region implementation that:

1. **Exact Fisher information.** The Hessian approximation is the Fisher information
   matrix in the log-transformed optimization variable space, computed analytically by
   [`fgh!`](@ref). Because the Fisher information equals the expected Hessian of the
   Poisson negative log-likelihood, this is identical to Fisher scoring and provides
   well-scaled search directions throughout the iteration, not just near the solution.

2. **Moré–Sorensen subproblem solver.** The trust-region subproblem is solved via
   eigendecomposition of the Fisher information matrix and bisection on the secular
   equation ([`_tr_subproblem_step`](@ref StarFormationHistories._tr_subproblem_step)).
   This handles indefinite and strongly correlated Hessians correctly, which arise
   frequently in SFH fitting due to coupling between stellar-mass and AMR/dispersion
   parameters.

3. **Correct MAP objective.** The log-prior Jacobian correction (``-\\sum_j \\log R_j``)
   is applied to the objective whenever the function value is requested, including during
   trial-step evaluations inside the trust-region loop. This corrects a subtle issue in
   [`fit_sfh_newton`](@ref) where the correction was inadvertently skipped for trial
   steps, causing the reduction-ratio estimate to be inaccurate for the MAP problem.

# Performance Notes
  - Each accepted step requires one evaluation of `fgh!` (to get f, g, and H) and one
    additional evaluation of f only (for the trial step). The per-iteration cost is
    therefore ``O(n_{\\text{opt}}^2 \\, n_{\\text{pix}})`` for the Fisher information
    plus a cheap ``O(n_{\\text{pix}})`` trial evaluation.
  - For large numbers of optimization variables (≫ 100) consider [`fit_sfh`](@ref).

# Keyword Arguments
All keyword arguments are passed to [`_trust_region_fisher_scoring`](@ref
StarFormationHistories._trust_region_fisher_scoring):
  - `x0`: initial stellar mass coefficients (per age bin), length `length(unique(logAge))`.
  - `g_abstol`: convergence criterion on ‖∇f‖∞ (default `1e-5`).
  - `maxiter`: maximum number of optimizer iterations (default `300`).
  - `Δ_init`, `Δ_max`, `η1`, `η2`, `α1`, `α2`: trust-region tuning parameters; see
    [`_trust_region_fisher_scoring`](@ref StarFormationHistories._trust_region_fisher_scoring).
"""
function fit_sfh_fisher(MH_model0::AbstractMetallicityModel{T},
                        disp_model0::AbstractDispersionModel{U},
                        models::AbstractMatrix{S},
                        data::AbstractVector{<:Number},
                        logAge::AbstractVector{<:Number},
                        metallicities::AbstractVector{<:Number};
                        x0::AbstractVector{<:Number} =
                            construct_x0_mdf(logAge, convert(S, 13.7);
                                             normalize_value=S(1e6)),
                        g_abstol::Real = 1e-5,
                        maxiter::Int   = 1000,
                        kws...) where {T, U, S <: Number}

    unique_logAge = unique(logAge)
    unique_MH     = unique(metallicities)
    Nbins         = length(x0)
    @argcheck length(x0) == length(unique_logAge) "length(x0) != length(unique(logAge))"
    @argcheck size(models, 1) == length(data)
    @argcheck size(models, 2) == length(logAge) == length(metallicities)

    # ── Renormalize x0 (mirrors fit_sfh / fit_sfh_newton) ────────────────────────
    let _full_x0 = calculate_coeffs(MH_model0, disp_model0, x0, logAge, metallicities)
        if any(isnan, _full_x0)
            if MH_model0 isa AbstractMZR
                @warn "Initial `x0` provided to `fit_sfh_fisher` produced NaN " *
                      "coefficients. Attempting to renormalize `x0` by replicating " *
                      "across metallicity bins."
            end
            _full_x0 = repeat(x0, length(unique_MH))
            x0 = renormalize_x0(data, models, x0, _full_x0)
            for i in 1:5
                _full_x0 = calculate_coeffs(MH_model0, disp_model0, x0, logAge, metallicities)
                if any(isnan, _full_x0)
                    @warn "Renormalization of x0 failed to produce valid coefficients " *
                          "after $i iterations. Returning last valid x0." i
                    break
                end
                x0 = renormalize_x0(data, models, x0, _full_x0)
            end
        else
            x0 = renormalize_x0(data, models, x0, _full_x0)
        end
    end

    # ── Log-transform x0 and pack metallicity / dispersion parameters ─────────────
    x0 = map(log, x0)
    par   = (values(fittable_params(MH_model0))..., values(fittable_params(disp_model0))...)
    tf    = (transforms(MH_model0)..., transforms(disp_model0)...)
    free  = SVector(free_params(MH_model0)..., free_params(disp_model0)...)
    x0_mzrdisp = logtransform(par, tf)
    x0 = vcat(x0, x0_mzrdisp[free])

    # Indices of positively-transformed (log) metallicity/dispersion params
    ptf = findall(==(1), tf)
    ptf = ptf[free[ptf]]
    ptf_idx = ptf .+ Nbins

    # Indices of negatively-transformed metallicity/dispersion params
    ntf = findall(==(-1), tf)
    ntf = ntf[free[ntf]]

    # ── Helper: expand optimization vector X (log-space) to full param vector ─────
    function _expand_x(X)
        Nfixed  = count(~, free)
        _Nbins  = lastindex(X) - count(free)
        full_x  = similar(X, _Nbins + nparams(MH_model0) + nparams(disp_model0))
        for i in 1:_Nbins; full_x[i] = exp(X[i]); end
        x_extra  = @view(X[_Nbins+1:end])
        x_zdisp  = exptransform(x_extra, SVector(tf)[free])
        full_x[(_Nbins+1:lastindex(full_x))[free]]   .= x_zdisp
        init_par = SVector(fittable_params(MH_model0)..., fittable_params(disp_model0)...)
        full_x[(_Nbins+1:lastindex(full_x))[.~free]] .= init_par[.~free]
        return full_x
    end

    # ── Helper: copy corrected free-parameter gradient to output G ────────────────
    function _write_free_gradient!(G, G2)
        for i in firstindex(G):Nbins
            G[i] = G2[i]
        end
        free_count = 1
        for i in 1:length(free)
            if free[i]
                G[Nbins + free_count] = G2[Nbins + i]
                free_count += 1
            end
        end
    end

    # ── MAP closure ───────────────────────────────────────────────────────────────
    # Objective = −logL − ∑_j log(R_j)  (log-prior Jacobian for positively-constrained vars)
    # In log-space x_j = log(R_j), so −log(R_j) = −x_j.
    # Gradient of −log(R_j) w.r.t. x_j = −1.
    # The Hessian of −∑ log(R_j) w.r.t. x is zero (the log-prior is linear in x),
    # so the Fisher information is the same for MAP and MLE.
    function fgh_map!(F, G, H_out, X)
        SS       = promote_type(eltype(models), eltype(data))
        composite = similar(data, SS)
        full_x   = _expand_x(X)
        G2       = isnothing(G) ? nothing : similar(full_x)
        nlogL    = fgh!(F, G2, H_out, MH_model0, disp_model0, full_x, models, data,
                        composite, logAge, metallicities)

        # Apply Jacobian/prior correction to the objective whenever F is requested.
        # NOTE: this must be done unconditionally (not gated on G or H being present),
        # so that trial-step evaluations (F=true, G=nothing, H=nothing) return the
        # correct MAP objective value.
        if !isnothing(F)
            for i in vcat(eachindex(full_x)[begin:Nbins], ptf_idx)
                nlogL -= log(full_x[i])
            end
            for i in ntf
                nlogL += log(full_x[i + Nbins])
            end
        end

        # Gradient correction: chain rule + Jacobian term from the log prior
        if !isnothing(G2)
            for i in vcat(eachindex(full_x)[begin:Nbins], ptf_idx)
                G2[i] = G2[i] * full_x[i] - 1
            end
            for i in ntf
                G2[i + Nbins] = -G2[i + Nbins] * full_x[i + Nbins] + 1
            end
        end

        if !isnothing(G)
            @argcheck axes(G) == axes(X)
            _write_free_gradient!(G, G2)
        end
        return isnothing(F) ? zero(SS) : nlogL
    end

    # ── MLE closure ───────────────────────────────────────────────────────────────
    # Objective = −logL (no prior), only chain-rule correction to gradient.
    function fgh_mle!(F, G, H_out, X)
        SS        = promote_type(eltype(models), eltype(data))
        composite = similar(data, SS)
        full_x    = _expand_x(X)
        G2        = isnothing(G) ? nothing : similar(full_x)
        nlogL     = fgh!(F, G2, H_out, MH_model0, disp_model0, full_x, models, data,
                         composite, logAge, metallicities)

        # Chain-rule correction only (no Jacobian prior term)
        if !isnothing(G2)
            for i in vcat(eachindex(full_x)[begin:Nbins], ptf_idx)
                G2[i] *= full_x[i]
            end
            for i in ntf
                G2[i + Nbins] = -G2[i + Nbins] * full_x[i + Nbins]
            end
        end

        if !isnothing(G)
            @argcheck axes(G) == axes(X)
            _write_free_gradient!(G, G2)
        end
        return isnothing(F) ? zero(SS) : nlogL
    end

    # ── Run custom trust-region optimizer ─────────────────────────────────────────
    Topt = promote_type(S, Float64)
    x0_opt = convert(Vector{Topt}, x0)
    x_map_opt, _, map_result = _trust_region_fisher_scoring(
        fgh_map!, x0_opt; g_abstol=Topt(g_abstol), maxiter, kws...)
    x_mle_opt, _, mle_result = _trust_region_fisher_scoring(
        fgh_mle!, x_map_opt; g_abstol=Topt(g_abstol), maxiter, kws...)

    # ── Compute inverse Hessian for uncertainty estimation ────────────────────────
    SS = promote_type(eltype(models), eltype(data))
    H_map = Matrix{SS}(undef, length(x0_opt), length(x0_opt))
    H_mle = Matrix{SS}(undef, length(x0_opt), length(x0_opt))
    fgh_map!(nothing, nothing, H_map, x_map_opt)
    fgh_mle!(nothing, nothing, H_mle, x_mle_opt)
    invH_map = PDMat(Hermitian(inv(Symmetric(H_map))))
    invH_mle = Hermitian(inv(Symmetric(H_mle)))
    try
        invH_mle = PDMat(invH_mle)
    catch
        @debug "Inverse Hessian matrix of MLE estimate is not positive definite" invH_mle
    end

    # ── Back-transform parameters and assemble result ─────────────────────────────
    let μ_map_tmp = x_map_opt, μ_mle_tmp = x_mle_opt
        σ_map_tmp = sqrt.(diag(invH_map))
        σ_mle_tmp = sqrt.(diag(invH_mle))

        μ_map = similar(μ_map_tmp, Nbins + nparams(MH_model0) + nparams(disp_model0))
        μ_mle = similar(μ_map)
        σ_map = similar(σ_map_tmp, length(μ_map))
        σ_mle = similar(σ_map)

        for i in 1:Nbins
            μ_map[i] = exp(μ_map_tmp[i])
            μ_mle[i] = exp(μ_mle_tmp[i])
            σ_map[i] = μ_map[i] * σ_map_tmp[i]
            σ_mle[i] = μ_mle[i] * σ_mle_tmp[i]
        end

        free_count = 0
        for i in Nbins+1:length(μ_map)
            if free[i - Nbins]
                tfi = tf[i - Nbins]
                j   = i - free_count
                if tfi == 1
                    μ_map[i] = exp(μ_map_tmp[j])
                    μ_mle[i] = exp(μ_mle_tmp[j])
                    σ_map[i] = μ_map[j] * σ_map_tmp[j]
                    σ_mle[i] = μ_map[j] * σ_mle_tmp[j]
                elseif tfi == 0
                    μ_map[i] = μ_map_tmp[j]
                    μ_mle[i] = μ_mle_tmp[j]
                    σ_map[i] = σ_map_tmp[j]
                    σ_mle[i] = σ_mle_tmp[j]
                elseif tfi == -1
                    μ_map[i] = -exp(μ_map_tmp[j])
                    μ_mle[i] = -exp(μ_mle_tmp[j])
                    σ_map[i] = -μ_map[j] * σ_map_tmp[j]
                    σ_mle[i] = -μ_map[j] * σ_mle_tmp[j]
                end
            else
                μ_map[i] = par[i - Nbins]
                μ_mle[i] = par[i - Nbins]
                σ_map[i] = 0
                σ_mle[i] = 0
                free_count += 1
            end
        end

        return CompositeBFGSResult(
            BFGSResult(μ_map, σ_map, invH_map, map_result,
                       update_params(MH_model0, @view(μ_map[Nbins+1:Nbins+nparams(MH_model0)])),
                       update_params(disp_model0, @view(μ_map[Nbins+nparams(MH_model0)+1:end]))),
            BFGSResult(μ_mle, σ_mle, invH_mle, mle_result,
                       update_params(MH_model0, @view(μ_mle[Nbins+1:Nbins+nparams(MH_model0)])),
                       update_params(disp_model0, @view(μ_mle[Nbins+nparams(MH_model0)+1:end]))))
    end
end

# Convenience wrapper for non-stacked (vector-of-matrices / 2-D-matrix data) call signature
fit_sfh_fisher(MH_model0::AbstractMetallicityModel, disp_model0::AbstractDispersionModel,
               models::AbstractVector{<:AbstractMatrix{<:Number}},
               data::AbstractMatrix{<:Number},
               logAge::AbstractVector{<:Number},
               metallicities::AbstractVector{<:Number}; kws...) =
    fit_sfh_fisher(MH_model0, disp_model0, stack_models(models), vec(data),
                   logAge, metallicities; kws...)
