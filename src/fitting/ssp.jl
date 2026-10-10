# Fitting a single simple stellar population (SSP) of one age and metallicity, e.g., a star cluster

const SSP_PARAMETERS = (:logAge, :MH, :dmod, :Av, :binary_fraction, :background_fraction)

"""
    SSPModel(isofunc, data::AbstractMatrix{<:Number}, edges, mag_err_funcs, y_index, color_indices, imf,
             completeness_funcs, bias_funcs;
             logAge, MH, dmod, Av=0, binary_model::AbstractBinaryModel=NoBinaries(),
             binary_fraction=binary_system_fraction(binary_model), background_fraction=Uniform(0, 1),
             background=nothing, background_floor::Real=0.05, mask=falses(size(data)),
             mean_mass::Number=mean(imf))

Model of the observed Hess diagram `data` as a single simple stellar population (SSP) of one age and metallicity plus a background, such as a star cluster and its surrounding field. Fit with [`fit_ssp`](@ref) and sample with [`sample_ssp`](@ref).

# Arguments
 - `isofunc(logAge, MH, Av)` returns the isochrone `(m_ini, mags)` for logarithmic age `logAge`, metallicity `MH`, and V-band extinction `Av`, with `m_ini` and `mags` as in [`partial_cmd_smooth`](@ref). It is called from multiple threads, so it must be thread-safe, and it must support every value allowed by the priors on these parameters.
 - `data` is the observed Hess diagram with bin edges `edges`, e.g., from [`bin_cmd`](@ref).
 - `mag_err_funcs`, `y_index`, `color_indices`, `imf`, `completeness_funcs`, and `bias_funcs` are as in [`partial_cmd_smooth`](@ref), for the magnitudes returned by `isofunc`.

# Parameters
The keyword arguments `logAge`, `MH`, `dmod` (distance modulus), `Av`, `binary_fraction`, and `background_fraction` are the parameters of the model. Each is either a `Real`, which fixes the parameter to that value, or a `Distributions.UnivariateDistribution`, which makes it a free parameter with that distribution as its prior. Free parameters are ordered as in `model.free`. `binary_fraction` is the binary system fraction of `binary_model` (see [`with_fraction`](@ref StarFormationHistories.with_fraction)), and `background_fraction` is the fraction of the observed stars in the unmasked bins of `data` that belong to the background.

# Keyword Arguments
 - `binary_model::AbstractBinaryModel` is the binary model, whose binary fraction is replaced by `binary_fraction`.
 - `background` is the shape of the background, a matrix the same size as `data` (e.g., the Hess diagram of a nearby field), or `nothing` for a uniform background.
 - `background_floor` is the fraction of the background spread uniformly over the unmasked bins, so that stars in bins where both `background` and the SSP are zero can be explained by the background rather than distorting the SSP parameters.
 - `mask` is `true` for bins excluded from the fit (see [`hess_mask`](@ref)).
 - `mean_mass` is as in [`partial_cmd_smooth`](@ref).

# Likelihood
The expected number of stars in bin ``i`` is ``A \\, p_i`` with

```math
p_i = (1 - g) \\frac{c_i}{\\sum_j c_j} + g \\, u_i
```

where ``c_i`` is the SSP Hess diagram per solar mass formed (see [`ssp_components`](@ref StarFormationHistories.ssp_components)), ``u_i`` is the background shape normalized to sum to 1, ``g`` is `background_fraction`, and sums run over the unmasked bins. With a scale-invariant prior ``p(A) \\propto 1/A`` on the total expected number of stars ``A``, marginalizing over ``A`` gives the multinomial log-likelihood ``\\sum_i n_i \\log p_i`` for the observed counts ``n_i``, so the fit depends only on the shape of the Hess diagram. The posterior of ``A`` is then ``\\text{Gamma}(N, 1)`` for ``N = \\sum_i n_i`` observed stars, from which [`sample_ssp`](@ref) derives the stellar mass formed in the SSP. This model implements the `LogDensityProblems` interface for use with other samplers.
"""
struct SSPModel{P <: NamedTuple, B <: AbstractBinaryModel}
    # Template construction dominates the cost of each evaluation, so these fields are not specialized
    isofunc
    edges
    mag_err_funcs
    y_index
    color_indices
    imf
    completeness_funcs
    bias_funcs
    mean_mass::Float64
    binary_model::B
    params::P                   # Value (fixed) or prior (free) of each parameter in SSP_PARAMETERS
    free::Vector{Symbol}        # Free parameters, in the order of the parameter vectors θ
    keep::BitVector             # Hess diagram bins used in the fit (not masked)
    idx::Vector{Int}            # Linear indices of the kept bins with observed stars
    counts::Vector{Float64}     # Observed counts in the bins `idx`
    background::Vector{Float64} # Background shape over all bins, normalized to sum to 1 over the kept bins
end
function SSPModel(isofunc, data::AbstractMatrix{<:Number}, edges, mag_err_funcs, y_index, color_indices, imf,
                  completeness_funcs, bias_funcs; logAge, MH, dmod, Av=0, binary_model::AbstractBinaryModel=NoBinaries(),
                  binary_fraction=binary_system_fraction(binary_model), background_fraction=Uniform(0, 1),
                  background=nothing, background_floor::Real=0.05, mask=falses(size(data)), mean_mass::Number=mean(imf))
    @argcheck length(edges) == 2
    @argcheck size(data) == (length(edges[1]) - 1, length(edges[2]) - 1) "`data` must have one entry per bin of `edges`."
    @argcheck all(>=(0), data) "`data` must not be negative."
    @argcheck size(mask) == size(data) "`mask` must be the same size as `data`."
    @argcheck 0 <= background_floor <= 1
    params = (; logAge, MH, dmod, Av, binary_fraction, background_fraction)
    for (k, v) in pairs(params)
        v isa Union{Real, UnivariateDistribution} || throw(ArgumentError("Parameter `$k` must be a number (fixed) or a univariate distribution (free, with that prior), not $(typeof(v))."))
    end
    if binary_model isa NoBinaries && !(binary_fraction isa Real && iszero(binary_fraction))
        throw(ArgumentError("`binary_fraction` requires a binary model other than `NoBinaries`."))
    end
    keep = .!vec(mask)
    any(keep) || throw(ArgumentError("`mask` excludes every bin."))
    if isnothing(background)
        shape = ones(length(keep))
    else
        @argcheck size(background) == size(data) "`background` must be the same size as `data`."
        shape = Float64.(vec(background))
    end
    bg = background_shape(shape, background_floor, keep)
    d = vec(data)
    idx = [i for i in eachindex(d, keep) if keep[i] && d[i] > 0]
    free = [k for k in SSP_PARAMETERS if params[k] isa UnivariateDistribution]
    return SSPModel(isofunc, edges, mag_err_funcs, y_index, color_indices, imf, completeness_funcs, bias_funcs,
                    Float64(mean_mass), binary_model, params, free, keep, idx, Float64.(d[idx]), bg)
end

LogDensityProblems.capabilities(::Type{<:SSPModel}) = LogDensityProblems.LogDensityOrder{0}()
LogDensityProblems.dimension(model::SSPModel) = length(model.free)
LogDensityProblems.logdensity(model::SSPModel, θ) = first(ssp_logdensity(model, θ))

# All parameter values as a NamedTuple, with the free parameters taken from θ in the order of model.free
ssp_params(model::SSPModel, θ) =
    NamedTuple{SSP_PARAMETERS}(Tuple(k in model.free ? θ[findfirst(==(k), model.free)] : model.params[k] for k in SSP_PARAMETERS))

# Prior standard deviations of the free parameters, used as their natural scales when starting the optimizer and walkers
prior_scales(model::SSPModel) = [(s = std(model.params[k]); isfinite(s) && s > 0 ? s : one(s)) for k in model.free]

# Flattened single-star and binary (binary fraction 1) Hess diagrams per solar mass formed for the parameters `p`; the
# binary diagram is `nothing` without binaries
function ssp_parts(model::SSPModel, p)
    m_ini, mags = model.isofunc(p.logAge, p.MH, p.Av)
    single, binary = ssp_components(m_ini, mags, model.mag_err_funcs, model.y_index, model.color_indices, model.imf,
                                    model.completeness_funcs, model.bias_funcs; binary_model=model.binary_model,
                                    dmod=p.dmod, mean_mass=model.mean_mass, edges=model.edges)
    return vec(single.weights), isnothing(binary) ? nothing : vec(binary.weights)
end
# SSP Hess diagram for binary system fraction `bf` from the single-star and binary parts
ssp_mix(model::SSPModel, single, binary::Nothing, bf) = single
function ssp_mix(model::SSPModel, single, binary, bf)
    φ = binary_mass_fraction(with_fraction(model.binary_model, bf), model.imf)
    return (1 - φ) .* single .+ φ .* binary
end
ssp_cluster(model::SSPModel, p) = ssp_mix(model, ssp_parts(model, p)..., p.binary_fraction)

ssp_logprior(model::SSPModel, θ) = sum((logpdf(model.params[k], θ[i]) for (i, k) in enumerate(model.free)); init=0.0)
# Log likelihood of the SSP Hess diagram `c` (per solar mass formed) with background fraction `g`, and the expected number
# of observed SSP stars per solar mass formed, which converts the expected number of SSP stars to stellar mass
function ssp_loglikelihood(model::SSPModel, c, g)
    T = sum(view(c, model.keep))
    T > 0 || return -Inf, T
    # Floored as in `loglikelihood` for bins the model cannot populate, e.g., with background_fraction fixed to 0
    ll = sum(n * log(max((1 - g) * c[i] / T + g * model.background[i], eps())) for (n, i) in zip(model.counts, model.idx); init=0.0)
    return ll, T
end

# Log posterior density (up to a constant) at the free parameters θ, and the expected number of observed SSP stars per
# solar mass formed
function ssp_logdensity(model::SSPModel, θ)
    lp = ssp_logprior(model, θ)
    isfinite(lp) || return -Inf, NaN
    p = ssp_params(model, θ)
    ll, T = ssp_loglikelihood(model, ssp_cluster(model, p), p.background_fraction)
    return lp + ll, T
end

# Unconstrained coordinates z for the free parameters, so that optimizers stay inside the support of the priors:
# logistic for bounded supports, exp for supports bounded below, and the identity otherwise
function to_θ(model::SSPModel, z)
    map(enumerate(model.free)) do (i, k)
        lo, hi = minimum(model.params[k]), maximum(model.params[k])
        isfinite(lo) && isfinite(hi) ? lo + (hi - lo) / (1 + exp(-z[i])) : isfinite(lo) ? lo + exp(z[i]) : z[i]
    end
end
function to_z(model::SSPModel, θ)
    map(enumerate(model.free)) do (i, k)
        lo, hi = minimum(model.params[k]), maximum(model.params[k])
        # Values at a bound map to a finite z just inside it
        x = clamp(θ[i], lo + 1e-9 * (isfinite(hi - lo) ? hi - lo : 1), hi - 1e-9 * (isfinite(hi - lo) ? hi - lo : 1))
        isfinite(lo) && isfinite(hi) ? log((x - lo) / (hi - x)) : isfinite(lo) ? log(x - lo) : x
    end
end
# dθ/dz at z, to convert steps in θ to steps in z; the logistic derivative is written with exp(-|z|) so it does not
# overflow far from z = 0, e.g., for parameters that start at a bound of their prior
function dθdz(model::SSPModel, z)
    map(enumerate(model.free)) do (i, k)
        lo, hi = minimum(model.params[k]), maximum(model.params[k])
        isfinite(lo) && isfinite(hi) ? (e = exp(-abs(z[i])); (hi - lo) * e / (1 + e)^2) : isfinite(lo) ? exp(z[i]) : one(z[i])
    end
end

# Log posterior density at θ with the free binary and background fractions (indices `jmix` into θ) maximized while the
# template parts `single` and `binary` (computed for the other parameters of θ) are held fixed; returns the log density
# and θ with the maximizing fractions. Mixing the fixed parts is cheap, so this profiles out the fractions on the
# starting grid without recomputing templates.
function profile_fractions(model::SSPModel, θ, single, binary, jmix)
    function f(θm)
        lp = ssp_logprior(model, θm)
        isfinite(lp) || return -Inf
        p = ssp_params(model, θm)
        return lp + first(ssp_loglikelihood(model, ssp_mix(model, single, binary, p.binary_fraction), p.background_fraction))
    end
    isempty(jmix) && return f(θ), θ
    z0 = to_z(model, θ)
    θof(u) = (zz = copy(z0); zz[jmix] .= u; to_θ(model, zz))
    result = Optim.optimize(u -> -f(θof(u)), z0[jmix], Optim.NelderMead(), Optim.Options(iterations=500))
    return -Optim.minimum(result), θof(Optim.minimizer(result))
end
# Starting-grid log density at θ: templates for the age and metallicity of θ, with the binary and background fractions
# profiled out
function grid_logdensity(model::SSPModel, θ, jmix)
    isfinite(ssp_logprior(model, θ)) || return -Inf, θ
    return profile_fractions(model, θ, ssp_parts(model, ssp_params(model, θ))..., jmix)
end

# Nelder-Mead in unconstrained coordinates from θ0, with initial simplex steps of half a prior standard deviation (at
# most 1 in z for bounded parameters, which already spans most of their range), restarted from its solution with a fresh
# simplex until the log density improves by less than `restart_tol`
function local_optimize(model::SSPModel, θ0; options, restarts, restart_tol)
    z = to_z(model, θ0)
    bounded = [isfinite(minimum(model.params[k])) for k in model.free]
    logdensity, results = -Inf, []
    for _ in 1:restarts + 1
        s = prior_scales(model) ./ 2 ./ dθdz(model, z)
        s[bounded] .= min.(s[bounded], 1)
        zc = copy(z)
        result = Optim.optimize(u -> -first(ssp_logdensity(model, to_θ(model, zc .+ s .* u))), zeros(length(z)),
                                Optim.NelderMead(initial_simplex=Optim.AffineSimplexer(1.0, 0.0)), options)
        push!(results, result)
        z = zc .+ s .* Optim.minimizer(result)
        improvement = -Optim.minimum(result) - logdensity
        logdensity = -Optim.minimum(result)
        improvement < restart_tol && break
    end
    return to_θ(model, z), logdensity, results
end

# Indices of the local maxima of the grid of log densities `L` (at least their neighbors along each axis, including
# diagonals), best first
function grid_maxima(L::AbstractArray)
    C = CartesianIndices(L)
    nb = oneunit(first(C))
    ismax(I) = isfinite(L[I]) && all(L[J] <= L[I] for J in max(first(C), I - nb):min(last(C), I + nb))
    return sort(filter(ismax, vec(collect(C))); by=I -> L[I], rev=true)
end

"""
    ssp_hess(model::SSPModel, p::NamedTuple)
    ssp_hess(fit::SSPFit)

Returns the expected Hess diagram of `model` for the parameters `p` (a value for every parameter, e.g., `fit.map` from [`fit_ssp`](@ref)) as a matrix the same size as the observed Hess diagram. It is normalized to the number of observed stars in the unmasked bins, which is the posterior mean of the total expected number of stars. Masked bins are included, so the model can be compared to the data inside them.
"""
function ssp_hess(model::SSPModel, p::NamedTuple)
    c = ssp_cluster(model, p)
    T = sum(view(c, model.keep))
    g = p.background_fraction
    return reshape(sum(model.counts) .* ((1 - g) .* c ./ T .+ g .* model.background), length(model.edges[1]) - 1, length(model.edges[2]) - 1)
end

"""
    ssp_mass(model::SSPModel, p::NamedTuple)
    ssp_mass(fit::SSPFit)

Returns the stellar mass formed in the SSP of `model`, in solar masses, for the parameters `p` (a value for every parameter, e.g., `fit.map` from [`fit_ssp`](@ref)), with the total expected number of stars equal to the number of observed stars in the unmasked bins, as in [`ssp_hess`](@ref StarFormationHistories.ssp_hess). [`sample_ssp`](@ref) instead draws the total expected number of stars from its posterior for each sample, so its `mass` samples also include the Poisson uncertainty in the number of stars.
"""
ssp_mass(model::SSPModel, p::NamedTuple) = sum(model.counts) * (1 - p.background_fraction) / sum(view(ssp_cluster(model, p), model.keep))

"""
    SSPFit

Result of [`fit_ssp`](@ref), with fields
 - `model::SSPModel`, the fitted model,
 - `map::NamedTuple`, the value of every parameter at the maximum a posteriori (MAP) solution,
 - `θ::Vector{Float64}`, the free parameters at the MAP, in the order of `model.free`,
 - `logdensity::Float64`, the log posterior density at the MAP (up to a constant),
 - `grid`, `nothing` or a `NamedTuple` with the values of the free `logAge` and `MH` on the starting grid, the log posterior density at each grid point with the free binary and background fractions maximized (`logdensity`), and the free parameters at each grid point (`θ`),
 - `starts`, a vector with one `NamedTuple` per local optimization, with its starting point `θ0`, solution `θ`, and log posterior density `logdensity`,
 - `optim`, the Optim.jl results of the Nelder-Mead runs of the best local optimization.
"""
struct SSPFit{M <: SSPModel, P <: NamedTuple, G, S, O}
    model::M
    map::P
    θ::Vector{Float64}
    logdensity::Float64
    grid::G
    starts::S
    optim::O
end
ssp_hess(fit::SSPFit) = ssp_hess(fit.model, fit.map)
ssp_mass(fit::SSPFit) = ssp_mass(fit.model, fit.map)

"""
    fit_ssp(model::SSPModel; x0=nothing, ngrid::NTuple{2, Integer}=(16, 12), nstarts::Integer=4,
            options::Optim.Options=Optim.Options(iterations=1000, g_tol=0.01), restarts::Integer=3,
            restart_tol::Real=1)::SSPFit

Finds the maximum a posteriori (MAP) parameters of `model` by local optimization with the Nelder-Mead method from one or more starting points. The optimization is done in unconstrained coordinates, so trial points stay within the support of the priors.

If `x0`, a vector of the free parameters in the order of `model.free`, is given, it is the only starting point. Otherwise, free `logAge` and `MH` are evaluated on a grid of `ngrid` values placed at evenly spaced quantiles of their priors, with the other free parameters at their prior medians, except that free binary and background fractions are maximized at each grid point (which does not require recomputing the templates). The local optimization starts from the `nstarts` best local maxima of this grid, in parallel over threads, and the best solution is returned. Multiple starting points guard against converging to a secondary maximum, which the degeneracy between age and metallicity can produce.

Because Nelder-Mead can stall before reaching the maximum, each local optimization is restarted from its solution with a fresh simplex, up to `restarts` times, until the log posterior density improves by less than `restart_tol`. `options` are passed to `Optim.optimize`; by default, each run stops when the standard deviation of the negative log density over the simplex falls below `g_tol = 0.01`. The default tolerances are loose because the MAP only needs to be within the region of the posterior that [`sample_ssp`](@ref) explores; posterior samples typically lie a few units of log density below the maximum. Returns an [`SSPFit`](@ref StarFormationHistories.SSPFit); sample the posterior with [`sample_ssp`](@ref), which uses every local solution in `starts` to find the modes of the posterior.
"""
function fit_ssp(model::SSPModel; x0=nothing, ngrid::NTuple{2, Integer}=(16, 12), nstarts::Integer=4,
                 options::Optim.Options=Optim.Options(iterations=1000, g_tol=0.01), restarts::Integer=3, restart_tol::Real=1)
    free = model.free
    isempty(free) && throw(ArgumentError("`model` has no free parameters."))
    @argcheck nstarts >= 1
    grid = nothing
    if isnothing(x0)
        xmed = [quantile(model.params[k], 0.5) for k in free]
        jmix = [findfirst(==(k), free) for k in (:binary_fraction, :background_fraction) if k in free]
        gridpars = [(k, n) for (k, n) in zip((:logAge, :MH), ngrid) if k in free]
        if isempty(gridpars)
            starts0 = [last(grid_logdensity(model, xmed, jmix))]
        else
            vals = [quantile.(model.params[k], ((1:n) .- 0.5) ./ n) for (k, n) in gridpars]
            jdx = [findfirst(==(k), free) for (k, _) in gridpars]
            points = collect(Iterators.product(vals...))
            lds = Array{Float64}(undef, size(points))
            θs = Array{Vector{Float64}}(undef, size(points))
            Threads.@threads for j in eachindex(points, lds, θs)
                # `θ` is assigned later in this function, so reusing that name here would share it between threads
                θj = copy(xmed)
                θj[jdx] .= points[j]
                lds[j], θs[j] = grid_logdensity(model, θj, jmix)
            end
            maxima = grid_maxima(lds)
            isempty(maxima) && throw(ArgumentError("The log posterior density is not finite anywhere on the starting grid."))
            starts0 = [θs[I] for I in first(maxima, nstarts)]
            grid = NamedTuple{(first.(gridpars)..., :logdensity, :θ)}((vals..., lds, θs))
        end
    else
        @argcheck length(x0) == length(free) "`x0` must have one value per free parameter $free."
        starts0 = [Float64.(x0)]
    end
    for θ0 in starts0
        isfinite(first(ssp_logdensity(model, θ0))) || throw(ArgumentError("The log posterior density is not finite at the starting point $(NamedTuple{Tuple(free)}(θ0))."))
    end
    tasks = [Threads.@spawn local_optimize(model, θ0; options, restarts, restart_tol) for θ0 in starts0]
    sols = fetch.(tasks)
    starts = [(θ0=θ0, θ=sol[1], logdensity=sol[2]) for (θ0, sol) in zip(starts0, sols)]
    any(sol -> isfinite(sol[2]), sols) || error("Every local optimization ended at a point with non-finite log posterior density.")
    θ, logdensity, results = sols[argmax([sol[2] for sol in sols])]
    return SSPFit(model, ssp_params(model, θ), θ, logdensity, grid, starts, results)
end

# Widths of the posterior along each free parameter from the curvature of the log density at θ (with log density `ld0`).
# Each parameter's step `h` is adjusted until the drop in log density over ±h, `d`, is between 2 and 20, well above the
# numerical noise of the templates, and the width is then h / sqrt(d). At a bound of the prior support only the inward
# side is used. Steps are capped at the prior standard deviation, where the posterior is no narrower than the prior.
function posterior_widths(model::SSPModel, θ, ld0)
    scale = prior_scales(model)
    n = length(θ)
    h, w = 1e-3 .* scale, copy(scale)
    done = falses(n)
    for _ in 1:15
        vals = Vector{Float64}(undef, 2n)
        Threads.@threads for q in 1:2n
            i = (q + 1) ÷ 2
            θq = copy(θ)
            θq[i] += isodd(q) ? h[i] : -h[i]
            vals[q] = first(ssp_logdensity(model, θq))
        end
        for i in 1:n
            done[i] && continue
            up, dn = vals[2i - 1], vals[2i]
            d = isfinite(up) && isfinite(dn) ? 2ld0 - up - dn : 2 * (ld0 - max(up, dn))
            if !isfinite(d) || d < 2
                h[i] >= scale[i] && (done[i] = true; continue) # As wide as the prior; keep w = scale
                h[i] = min(4h[i], scale[i])
            elseif d > 20
                h[i] /= 2
            else
                w[i] = h[i] / sqrt(d)
                done[i] = true
            end
        end
        all(done) && break
    end
    return w
end

#########################################
# Posterior sampling: an emcee pilot per mode, then adaptive importance sampling

# Walkers in a ball around θ with radius `radius` times the posterior widths `w`, reflected into the prior support
function ssp_ball(model::SSPModel, θ, w, nwalkers, radius, rng)
    lo, hi = [minimum(model.params[k]) for k in model.free], [maximum(model.params[k]) for k in model.free]
    map(1:nwalkers) do _
        for _ in 1:100
            x = θ .+ radius .* w .* randn(rng, length(θ))
            x = @. ifelse(x < lo, 2lo - x, ifelse(x > hi, 2hi - x, x))
            isfinite(first(ssp_logdensity(model, x))) && return x
        end
        error("No starting position with finite log posterior density found near the solution $(NamedTuple{Tuple(model.free)}(θ)); reduce `radius`.")
    end
end

# Short emcee run (KissMCMC) of `nwalkers` walkers for `nsteps` steps from a ball around θ; returns the mean and covariance
# of the walker positions over the second half of the steps, and the mean acceptance ratio. KissMCMC draws from the
# default random number generator, so the run is done in its own task seeded from `rng`, which makes it reproducible for
# a given number of threads without changing the caller's default generator.
function ssp_pilot(model::SSPModel, θ; nwalkers, nsteps, radius, rng)
    theta0s = ssp_ball(model, θ, posterior_widths(model, θ, first(ssp_logdensity(model, θ))), nwalkers, radius, rng)
    s = rand(rng, UInt64)
    thetas, accept_ratio = fetch(Threads.@spawn begin
        seed!(s)
        KissMCMC.emcee(x -> ssp_logdensity(model, x), theta0s; niter=nsteps * nwalkers, nburnin=(nsteps ÷ 2) * nwalkers,
                       use_progress_meter=false, hasblob=true)
    end)
    X = reduce(hcat, reduce(vcat, thetas))
    μ = vec(sum(X; dims=2)) ./ size(X, 2)
    Σ = (X .- μ) * (X .- μ)' ./ (size(X, 2) - 1)
    return μ, Matrix(Symmetric(Σ)), mean(accept_ratio)
end

# Groups the local solutions `sols` of fit_ssp (with fields θ and logdensity, best first) into modes. Two solutions are the
# same mode if either lies within the 99.9% chi-squared distance, for the number of free parameters, of the other's pilot
# (the Mahalanobis distance with the pilot's mean and covariance). Solutions near an existing mode's pilot need no pilot
# of their own; the others get one, which still joins them to an existing mode if that mode's solution lies within the
# same distance of it (e.g., a small local maximum on a mode's flank, whose pilot moves into the mode).
function ssp_modes(model::SSPModel, sols; nwalkers, nsteps, radius, rng)
    τ2 = quantile(Chisq(length(model.free)), 0.999)
    near(θ, μ, Σ) = (δ = θ .- μ; dot(δ, Symmetric(Σ) \ δ) < τ2)
    modes = @NamedTuple{θ::Vector{Float64}, logdensity::Float64, μ::Vector{Float64}, Σ::Matrix{Float64}, accept_ratio::Float64}[]
    for s in sols
        any(m -> near(s.θ, m.μ, m.Σ), modes) && continue
        μ, Σ, accept_ratio = ssp_pilot(model, s.θ; nwalkers, nsteps, radius, rng)
        any(m -> near(m.θ, μ, Σ), modes) && continue
        push!(modes, (θ=Vector{Float64}(s.θ), logdensity=Float64(s.logdensity), μ=μ, Σ=Σ, accept_ratio=accept_ratio))
    end
    return modes
end

"""
    adaptive_is(logd, μs, Σs; rounds, n, ν=5, inflate=1.5, minweight=0.1, rng)

Adaptive importance sampling of the log density `logd` with the mixture population Monte Carlo (M-PMC) method of
[Cappe2008](@citet), whose proposal is a mixture of multivariate ``t`` distributions that is refitted to the
importance-weighted draws after each of `rounds` rounds. `logd(x)` returns the log density (normalized or not) at the
parameter vector `x` and a `Float64` blob that is stored with each draw.

The mixture has one component per mode, with `ν > 2` degrees of freedom; component ``k`` starts with weight
``\\alpha_k = 1/K``, where ``K`` is `length(μs)`, mean `μs[k]`, and covariance `inflate^2 * Σs[k]`. Each round
 1. draws `n` points ``x_i`` from the mixture with `rng`;
 2. evaluates `logd` at the draws in parallel over threads;
 3. weights each draw by ``w_i \\propto \\pi(x_i) / q(x_i)``, where ``\\pi`` is the target density `exp(logd)` and
    ``q(x) = \\sum_k \\alpha_k \\, t_k(x)`` is the density of the whole mixture;
 4. refits the mixture as in the M-PMC update of [Cappe2008](@citet): the draws are split among the components by their
    responsibilities ``\\rho_{ik} = \\alpha_k t_k(x_i) / q(x_i)``, each ``\\alpha_k`` becomes ``\\sum_i w_i \\rho_{ik}``
    (with normalized ``w_i``), and each mean and covariance becomes the mean and covariance of the draws weighted by
    ``w_i \\rho_{ik}``.

The update differs from that of [Cappe2008](@citet) in ways that keep the proposal robust when it starts far from
the target. The means and covariances are refitted as weighted moments, rather than with their update for ``t``
mixtures, and the covariances are again multiplied by `inflate^2`, so the proposal keeps heavier tails than the target,
which bounds the importance weights. Each ``\\alpha_k`` is kept at least `minweight` (before renormalizing), so that no
mode is starved of draws. A component whose weighted draws are worth fewer than twice the number of parameters in
effective samples keeps its previous mean and covariance.

Returns a vector with one `NamedTuple` per round with fields `Y` (a matrix with the draws as columns), `ld` and `blob`
(the values returned by `logd` at the draws), `lw` (the log importance weights, ``\\log \\pi(x_i) - \\log q(x_i)``),
and the proposal of that round: `α` (the component weights), `μs` (the means), and `Λs` (the scale matrices, which are
the covariances times ``(\\nu - 2) / \\nu``). `amis_combine` combines the draws of all rounds.
"""
function adaptive_is(logd, μs, Σs; rounds, n, ν=5, inflate=1.5, minweight=0.1, rng)
    K, d = length(μs), length(first(μs))
    c = inflate^2 * (ν - 2) / ν # Scale matrix of a t distribution with covariance inflate² Σ
    μs, Λs, α = copy(μs), [c .* Σ for Σ in Σs], fill(1 / K, K)
    out = []
    for r in 1:rounds
        comps = [MvTDist(ν, μs[k], PDMat(Symmetric(Λs[k]))) for k in 1:K]
        ks = rand(rng, Categorical(α), n)
        Y = reduce(hcat, [rand(rng, comps[k]) for k in ks])
        ld, blob = Vector{Float64}(undef, n), Vector{Float64}(undef, n)
        Threads.@threads for i in 1:n
            ld[i], blob[i] = logd(view(Y, :, i))
        end
        lc = [log(α[k]) + logpdf(comps[k], view(Y, :, i)) for k in 1:K, i in 1:n]
        lq = [logsumexp(view(lc, :, i)) for i in 1:n]
        lw = ld .- lq
        any(isfinite, lw) || error("No draw of round $r of adaptive importance sampling has a finite log posterior density.")
        push!(out, (; Y, ld, blob, lw, α=copy(α), μs=copy(μs), Λs=copy(Λs)))
        r == rounds && break
        W = softmax(lw)
        ω = [W .* exp.(view(lc, k, :) .- lq) for k in 1:K]
        α = max.(sum.(ω) ./ sum(sum.(ω)), minweight)
        α ./= sum(α)
        for k in 1:K
            sum(ω[k])^2 / sum(abs2, ω[k]) > 2d || continue
            wk = ω[k] ./ sum(ω[k])
            m = Y * wk
            μs[k], Λs[k] = m, c .* Matrix(Symmetric((Y .- m) * (wk .* (Y .- m)')))
        end
    end
    return out
end

# Log of each component's weighted density, log(α_k t_k(x)), of a round's proposal at the draws X (columns)
function mixture_terms(rd, X, ν)
    comps = [MvTDist(ν, rd.μs[k], PDMat(Symmetric(rd.Λs[k]))) for k in eachindex(rd.α)]
    return [log(rd.α[k]) + logpdf(comps[k], view(X, :, i)) for k in eachindex(comps), i in axes(X, 2)]
end

"""
    amis_combine(rounds, ν)

Combines the draws of all `rounds` returned by [`adaptive_is`](@ref StarFormationHistories.adaptive_is), whose mixture
components have `ν` degrees of freedom, with the weights of adaptive multiple importance sampling (AMIS)
[Cornuet2012](@citep). Rather than by its own round's proposal, each draw ``x_i`` is weighted by the mixture of the
proposals ``q_r`` of all rounds ``r``, each weighted by its share of the ``N`` draws, ``N_r / N``,

```math
w_i = \\frac{\\pi(x_i)}{\\sum_r (N_r / N) \\, q_r(x_i)}.
```

The early rounds, whose proposals were still adapting, then add to the sample instead of being discarded, and a round
whose proposal missed part of the target is covered by the others. As [Cornuet2012](@citet) discuss, because each
proposal depends on the earlier draws, the convergence of AMIS estimators does not follow from that of standard
importance sampling.

Returns a `NamedTuple` with fields `X` (a matrix with the draws of all rounds as columns), `ld` and `blob` (the values
returned by `logd` at the draws), `lw` (the log AMIS weights), and `mode` (for each draw, the component that
contributes most to the combined mixture density at the draw, which assigns the draw to a mode).
"""
function amis_combine(rounds, ν)
    X = reduce(hcat, [rd.Y for rd in rounds])
    N = size(X, 2)
    lterm = reduce((a, b) -> logaddexp.(a, b), [mixture_terms(rd, X, ν) .+ log(size(rd.Y, 2) / N) for rd in rounds])
    ld = reduce(vcat, [rd.ld for rd in rounds])
    lw = ld .- [logsumexp(view(lterm, :, i)) for i in 1:N]
    return (; X, ld, blob=reduce(vcat, [rd.blob for rd in rounds]), lw, mode=[argmax(view(lterm, :, i)) for i in 1:N])
end

"""
    gpd_khat(x)

Estimates the shape ``k`` of a generalized Pareto distribution from the nonnegative exceedances `x` over a threshold,
with the method of [Zhang2009](@citet), which reparameterizes the distribution by the ratio ``\\theta`` of its shape
to its scale (up to sign), estimates ``\\theta`` as the mean over a grid of ``m = 30 + \\lfloor \\sqrt{n} \\rfloor``
values, for ``n`` exceedances, weighted by their profile likelihoods (an approximate posterior mean), and takes ``k``
as its maximum-likelihood value given ``\\theta``. As in the implementation of Pareto smoothed importance sampling
(PSIS) by [Vehtari2024](@citet), ``k`` is then shrunk toward 0.5 by a weakly informative prior worth 10 observations,
which stabilizes the estimate for small ``n``. Larger ``k`` means a heavier tail; the distribution has a finite mean for
``k < 1`` and a finite variance for ``k < 1/2``.
"""
function gpd_khat(x)
    x = sort(x)
    n = length(x)
    m = 30 + floor(Int, sqrt(n))
    xstar = x[max(1, floor(Int, n / 4 + 0.5))]
    θ = [1 / x[n] + (1 - sqrt(m / (j - 0.5))) / 3 / xstar for j in 1:m]
    lθ = map(θ) do t
        k = mean(log1p.(-t .* x))
        n * (log(-t / k) - k - 1)
    end
    θhat = sum(θ .* softmax(lθ))
    k = mean(log1p.(-θhat .* x))
    return (k * n + 0.5 * 10) / (n + 10)
end
"""
    pareto_khat(lw)

Returns the Pareto ``\\hat k`` diagnostic of [Vehtari2024](@citet) for the importance weights with logarithms `lw`: the
shape of a generalized Pareto distribution fitted by [`gpd_khat`](@ref StarFormationHistories.gpd_khat) to the
largest ``M = \\min(S/5, 3\\sqrt{S})`` of the ``S`` finite weights, as exceedances over the next largest weight. It
measures how heavy the tail of the weights is, which sets how reliable estimates from them are: they are reliable for
``\\hat k`` below 0.7, the threshold used by [`sample_ssp`](@ref), and above it a few draws carry most of the weight
and the estimates can be badly biased even when the effective sample size looks adequate. Returns `Inf` if there are
too few finite weights to fit (fewer than 5 in the tail) or if the largest weight is so much larger than the rest of the
tail that they underflow to 0.
"""
function pareto_khat(lw)
    l = filter(isfinite, lw)
    S = length(l)
    M = min(ceil(Int, S / 5), ceil(Int, 3 * sqrt(S)))
    (M >= 5 && S > M) || return Inf
    w = exp.(sort(l) .- maximum(l))
    k = gpd_khat(w[(S - M + 1):S] .- w[S - M])
    return isfinite(k) ? k : Inf # Tail weights that underflow to 0 are infinitely heavy relative to the largest
end

# Convergence diagnostics of the last round `rd` of adaptive_is: the Pareto k̂ of its weights, overall and for each
# component (the weights times its responsibilities), and each component's proposal shift, the Gaussian KL divergence from
# the mean and covariance refitted to the round's weighted draws to those the proposal was built from. A proposal that is
# still moving or growing has not converged, even when its weights look healthy, e.g., when it is too narrow along a curved
# mode and no draws reach the part it misses.
function round_diagnostics(rd, ν, inflate)
    K, n = length(rd.α), size(rd.Y, 2)
    lc = mixture_terms(rd, rd.Y, ν)
    lq = [logsumexp(view(lc, :, i)) for i in 1:n]
    W = softmax(rd.lw)
    khat = max(pareto_khat(rd.lw), maximum(pareto_khat(rd.lw .+ view(lc, k, :) .- lq) for k in 1:K))
    shift = map(1:K) do k
        ω = W .* exp.(view(lc, k, :) .- lq)
        sum(ω) > 0 || return Inf
        ω ./= sum(ω)
        m = rd.Y * ω
        C = Symmetric((rd.Y .- m) * (ω .* (rd.Y .- m)'))
        isposdef(C) || return Inf
        C0 = Symmetric(rd.Λs[k] .* (ν / (ν - 2)) ./ inflate^2)
        δ = m .- rd.μs[k]
        (tr(C0 \ Matrix(C)) + dot(δ, C0 \ δ) - length(m) + logdet(C0) - logdet(C)) / 2
    end
    return (; khat, shift)
end

# Indices of `n` equal-weight draws from the normalized weights `w` by systematic resampling, in random order
function systematic_resample(w, n, rng)
    c = cumsum(w)
    u0 = rand(rng) / n
    idx = Vector{Int}(undef, n)
    j = 1
    for i in 1:n
        u = u0 + (i - 1) / n
        while j < length(c) && c[j] < u
            j += 1
        end
        idx[i] = j
    end
    return shuffle!(rng, idx)
end

"""
    SSPSamples

Result of [`sample_ssp`](@ref), with fields
 - `fit::SSPFit`, the fit that was sampled,
 - `samples::NamedTuple`, equal-weight posterior samples as one vector per column: the free parameters, named as in `fit.model.free`; `mass`, the stellar mass formed in the SSP in solar masses; `background_stars`, the expected number of background stars in the unmasked bins; `lp`, the log posterior density (up to a constant); and `mode`, the index into `modes` of the mode each sample belongs to,
 - `logZ::Float64`, the logarithm of the evidence, the integral of the posterior density over the parameters, up to the same constant as `lp`,
 - `ess::Float64`, the effective sample size of the importance weights; the samples contain repeated draws, and `ess` is the number of independent samples they are worth,
 - `modes`, a vector with one `NamedTuple` per mode: the local solution of [`fit_ssp`](@ref) it was found from (`θ` and `logdensity`), its posterior probability (`weight`), its log evidence (`logZ`), and the effective sample size of its importance weights (`ess`),
 - `diagnostics`, a `NamedTuple` with the Pareto ``\\hat k`` of the importance weights of the last round (`khat`), the proposal shift of each mode in the last round (`shift`, in nats), and the log evidence and fraction of effective draws of each round (`rounds`),
 - `warnings::Vector{String}`, the warnings issued by [`sample_ssp`](@ref).
"""
struct SSPSamples{F <: SSPFit, S <: NamedTuple, M, D}
    fit::F
    samples::S
    logZ::Float64
    ess::Float64
    modes::M
    diagnostics::D
    warnings::Vector{String}
end
function Base.show(io::IO, s::SSPSamples)
    print(io, "SSPSamples: ", length(s.samples.lp), " samples (", round(Int, s.ess), " effective) of ", join(s.fit.model.free, ", "),
          " from ", length(s.modes), length(s.modes) == 1 ? " mode" : " modes", "; log evidence ", @sprintf("%.3f", s.logZ),
          isempty(s.warnings) ? "" : "; $(length(s.warnings)) warning(s)")
end

"""
    sample_ssp(fit::SSPFit; pilot_steps::Integer=300, pilot_walkers::Integer=max(16, 4 * length(fit.θ)),
               is_rounds::Integer=3, is_samples::Integer=2000, nsamples::Integer=is_rounds * is_samples,
               mode_cut::Real=20, radius::Real=0.5, rng::AbstractRNG=default_rng())::SSPSamples

Samples the posterior of the model fit by [`fit_ssp`](@ref), including every mode found by its local optimizations, and
returns an [`SSPSamples`](@ref StarFormationHistories.SSPSamples) with equal-weight samples, the posterior probability
and evidence of each mode, and convergence diagnostics.

 1. **Modes.** The local solutions in `fit.starts` within `mode_cut` of the best log posterior density are grouped into
    modes. From each, an ensemble of `pilot_walkers` walkers (an even number, at least the number of free parameters
    plus 2) of the affine-invariant sampler of [KissMCMC.jl](https://github.com/mauro3/KissMCMC.jl), started in a ball of
    radius `radius` times the posterior width along each parameter, takes `pilot_steps` steps; the mean and covariance of
    the second half describe the mode. Two solutions are the same mode if either lies within the 99.9% chi-squared
    distance (for the number of free parameters) of the other's pilot, so solutions that belong to an existing mode
    usually need no pilot of their own.
 2. **Adaptive importance sampling.** The posterior is sampled with `is_rounds` rounds of `is_samples` draws each from a
    mixture of multivariate ``t`` distributions, one per mode, starting from the pilots' means and covariances. After
    each round, each component is refitted to the importance-weighted draws it is responsible for. At least 3 rounds
    are required, since fewer often fail to adapt the proposal from the pilots. The draws of all rounds are combined
    with the weights of adaptive multiple importance sampling (AMIS), which give each mode's posterior probability and
    evidence directly.
 3. **Samples.** `nsamples` equal-weight samples are drawn from the weighted draws by systematic resampling, so they
    contain repeated draws. The total expected number of stars is drawn from its posterior for each sample (see
    [`SSPModel`](@ref StarFormationHistories.SSPModel)) to give the stellar mass formed and the expected number of
    background stars.

The importance weights also tell whether the settings were adequate. A warning is issued, and recorded in the result, if
the last round's proposal was still adapting (a proposal shift above 0.5 nats or a Pareto ``\\hat k`` of the weights
above 0.7), in which case `is_rounds` or `pilot_steps` should be increased, or if the weights have fewer than 1000
effective samples, in which case `is_samples` should be increased. The evidence and mode probabilities are relative to
the modes that [`fit_ssp`](@ref) found; a mode that none of its local optimizations reached is not represented.

Evaluations of the log posterior density are parallelized over threads. Results are reproducible for a given `rng` and
number of threads.
"""
function sample_ssp(fit::SSPFit; pilot_steps::Integer=300, pilot_walkers::Integer=max(16, 4 * length(fit.θ)),
                    is_rounds::Integer=3, is_samples::Integer=2000, nsamples::Integer=is_rounds * is_samples,
                    mode_cut::Real=20, radius::Real=0.5, rng::AbstractRNG=default_rng())
    model = fit.model
    @argcheck iseven(pilot_walkers) && pilot_walkers >= length(fit.θ) + 2 "`pilot_walkers` must be even and at least the number of free parameters plus 2."
    # In tests on simulated posteriors, 1 or 2 rounds often failed to adapt the proposal from the pilots, and 3 did not
    @argcheck is_rounds >= 3 "`is_rounds` must be at least 3; fewer rounds do not adapt the proposal reliably."
    @argcheck pilot_steps >= 2 && is_samples >= 1 && nsamples >= 1 && mode_cut > 0
    sols = sort([s for s in fit.starts if isfinite(s.logdensity) && s.logdensity > fit.logdensity - mode_cut]; by=s -> -s.logdensity)
    modes = ssp_modes(model, sols; nwalkers=pilot_walkers, nsteps=pilot_steps, radius, rng)
    ν, inflate = 5, 1.5
    rounds = adaptive_is(θ -> ssp_logdensity(model, θ), [m.μ for m in modes], [m.Σ for m in modes]; rounds=is_rounds, n=is_samples, ν, inflate, rng)
    a = amis_combine(rounds, ν)
    W = softmax(a.lw)
    logZ = logsumexp(a.lw) - log(length(a.lw))
    ess = 1 / sum(abs2, W)
    mode_table = map(eachindex(modes)) do k
        Wk = W[a.mode .== k]
        weight = sum(Wk; init=0.0)
        (θ=modes[k].θ, logdensity=modes[k].logdensity, weight=weight, logZ=logZ + log(weight), ess=weight > 0 ? weight^2 / sum(abs2, Wk) : 0.0)
    end
    diag = round_diagnostics(last(rounds), ν, inflate)
    warnings = String[]
    if maximum(diag.shift) > 0.5 || diag.khat > 0.7
        push!(warnings, @sprintf("Adaptive importance sampling has not converged: the largest proposal shift of the last round is %.2f nats (threshold 0.5) and the Pareto k̂ of its weights is %.2f (threshold 0.7). Increase `is_rounds` or `pilot_steps`.", maximum(diag.shift), diag.khat))
    end
    if ess < 1000
        push!(warnings, @sprintf("The importance weights have only %.0f effective samples (threshold 1000), so posterior estimates are imprecise. Increase `is_samples`.", ess))
    end
    foreach(w -> @warn(w), warnings)
    idx = systematic_resample(W, nsamples, rng)
    X = a.X[:, idx]
    g = [ssp_params(model, view(X, :, i)).background_fraction for i in axes(X, 2)]
    A = rand(rng, Gamma(sum(model.counts), 1), nsamples)
    cols = (Tuple(X[j, :] for j in axes(X, 1))..., A .* (1 .- g) ./ a.blob[idx], A .* g, a.ld[idx], a.mode[idx])
    samples = NamedTuple{(model.free..., :mass, :background_stars, :lp, :mode)}(cols)
    diagnostics = (; diag.khat, diag.shift, rounds=[(logZ=logsumexp(rd.lw) - log(length(rd.lw)), ess_fraction=1 / sum(abs2, softmax(rd.lw)) / length(rd.lw)) for rd in rounds])
    return SSPSamples(fit, samples, logZ, ess, mode_table, diagnostics, warnings)
end
