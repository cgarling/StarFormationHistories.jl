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

Because Nelder-Mead can stall before reaching the maximum, each local optimization is restarted from its solution with a fresh simplex, up to `restarts` times, until the log posterior density improves by less than `restart_tol`. `options` are passed to `Optim.optimize`; by default, each run stops when the standard deviation of the negative log density over the simplex falls below `g_tol = 0.01`. The default tolerances are loose because the MAP only needs to be within the region of the posterior that [`sample_ssp`](@ref) explores; posterior samples typically lie a few units of log density below the maximum. Returns an [`SSPFit`](@ref StarFormationHistories.SSPFit); sample the posterior with [`sample_ssp`](@ref).
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

"""
    sample_ssp(fit::SSPFit, nsteps::Integer; nwalkers::Integer=max(16, 4 * length(fit.θ)),
               nburnin::Integer=nsteps ÷ 2, nthin::Integer=1, radius::Real=0.5,
               rng::AbstractRNG=default_rng(), use_progress_meter::Bool=true)

Samples the posterior of the model fit by [`fit_ssp`](@ref) with the affine-invariant ensemble sampler of [KissMCMC.jl](https://github.com/mauro3/KissMCMC.jl), analogous to Python's [emcee](https://emcee.readthedocs.io/en/stable/), parallelized over threads. The `nwalkers` walkers (an even number, at least the number of free parameters plus 2) start in a ball around the MAP solution with a radius of `radius` times the width of the posterior along each parameter, estimated from the curvature of the log posterior density at the MAP. Each walker takes `nsteps` steps, of which the first `nburnin` are discarded, and every `nthin`-th step is kept. The walkers explore the region of the posterior around the MAP and are unlikely to cross into a separate maximum, so the MAP from [`fit_ssp`](@ref) should be in the region of the global maximum.

Returns an `MCMCChains.Chains` with one chain per walker. Its parameters are the free parameters, named as in `fit.model.free`, and the derived quantities
 - `mass`, the stellar mass formed in the SSP in solar masses, and
 - `background_stars`, the expected number of background stars in the unmasked bins.
The log posterior density (up to a constant) of each sample is stored as the internal `lp`, and the acceptance ratio of each walker as `info.accept_ratio`; acceptance ratios much below 0.2 indicate poor mixing. Because the walkers of an ensemble sampler are not independent, convergence diagnostics that compare chains (e.g., the potential scale reduction) are optimistic; check that the effective sample size is adequate for each parameter.

`rng` draws the starting positions and the masses; KissMCMC draws its proposals with the default random number generator, so seed it with `Random.seed!` for reproducible chains.
"""
function sample_ssp(fit::SSPFit, nsteps::Integer; nwalkers::Integer=max(16, 4 * length(fit.θ)),
                    nburnin::Integer=nsteps ÷ 2, nthin::Integer=1, radius::Real=0.5,
                    rng::AbstractRNG=default_rng(), use_progress_meter::Bool=true)
    model = fit.model
    @argcheck iseven(nwalkers) && nwalkers >= length(fit.θ) + 2 "`nwalkers` must be even and at least the number of free parameters plus 2."
    @argcheck 0 <= nburnin < nsteps
    w = posterior_widths(model, fit.θ, fit.logdensity)
    lo, hi = [minimum(model.params[k]) for k in model.free], [maximum(model.params[k]) for k in model.free]
    theta0s = map(1:nwalkers) do _
        for _ in 1:100
            # Reflect positions beyond a bound of the prior support back inside it, for MAPs near a bound
            θ = fit.θ .+ radius .* w .* randn(rng, length(fit.θ))
            θ = @. ifelse(θ < lo, 2lo - θ, ifelse(θ > hi, 2hi - θ, θ))
            isfinite(first(ssp_logdensity(model, θ))) && return θ
        end
        error("No starting position with finite log posterior density found near the MAP solution; reduce `radius`.")
    end
    thetas, accept_ratio, logdensities, blobs = KissMCMC.emcee(θ -> ssp_logdensity(model, θ), theta0s; niter=nsteps * nwalkers,
                                                               nburnin=nburnin * nwalkers, nthin, use_progress_meter, hasblob=true)
    # Array of (sample, parameter, walker); the total expected number of stars is drawn from its Gamma posterior and
    # split between the SSP (converted to mass with the blob T) and the background
    npar = length(model.free)
    val = Array{Float64}(undef, length(first(thetas)), npar + 3, nwalkers)
    Agamma = Gamma(sum(model.counts), 1)
    for w in 1:nwalkers, (j, θ) in enumerate(thetas[w])
        A, g = rand(rng, Agamma), ssp_params(model, θ).background_fraction
        val[j, 1:npar, w] .= θ
        val[j, npar + 1:end, w] .= (A * (1 - g) / blobs[w][j], A * g, logdensities[w][j])
    end
    parameters = [model.free; :mass; :background_stars]
    return MCMCChains.Chains(val, [parameters; :lp], (parameters=parameters, internals=[:lp]); start=nburnin + nthin, thin=nthin,
                             info=(accept_ratio=accept_ratio,))
end
