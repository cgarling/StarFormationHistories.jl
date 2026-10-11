# Templates for simple stellar populations whose stars have a distribution of extinctions (differential extinction).
#
# Each star (an interval of the densified isochrone, or a cell of binary systems) is deposited along its reddening path:
# the magnitudes at each extinction are interpolated linearly between their values at `knots`, so the path is a straight
# segment between consecutive knots. Each knot interval is divided into sub-segments short enough that the star moves by
# at most `max_shift` resolution elements (max(photometric error, bin width)) across one, and each sub-segment is
# deposited as a Gaussian kernel whose covariance includes the spread of extinction across it, with the photometric
# error, completeness, and bias evaluated at the mean extinction of the sub-segment.

#################################
# Extinction distributions

"""
    prepare_extinction(dist; nσ::Real=3)

Checks that `dist` can be used as a distribution of V-band extinctions for [`partial_cmd_smooth`](@ref) and returns it,
truncated if needed. `dist` must be a `Distributions.ContinuousUnivariateDistribution`; distributions with point masses
(e.g., `Distributions.Dirac`) are not supported. A side of `dist` that is unbounded is truncated at the quantile that
corresponds to `nσ` standard deviations of a normal distribution (a tail probability of about 0.00135 for `nσ = 3`), and
the support is truncated at zero, as negative extinctions are unphysical. `dist` is returned unchanged if no truncation
is needed.

```jldoctest; setup = :(using StarFormationHistories: prepare_extinction; using Distributions: Normal, Uniform)
julia> prepare_extinction(Uniform(0.2, 1.2)) == Uniform(0.2, 1.2)
true

julia> round.(extrema(prepare_extinction(Normal(0.7, 0.15))); digits=6)
(0.25, 1.15)
```
"""
function prepare_extinction(dist::ContinuousUnivariateDistribution; nσ::Real=3)
    lo, hi = minimum(dist), maximum(dist)
    ptail = (1 + erf(-nσ / sqrt(2))) / 2 # Φ(-nσ)
    newlo = max(isfinite(lo) ? lo : quantile(dist, ptail), zero(lo))
    newhi = isfinite(hi) ? hi : quantile(dist, 1 - ptail)
    newhi > newlo || throw(ArgumentError("The extinction distribution has no support at non-negative extinctions within its $(nσ)σ range."))
    return (newlo, newhi) == (lo, hi) ? dist : truncated(dist, newlo, newhi)
end
prepare_extinction(dist; nσ::Real=3) = throw(ArgumentError("The extinction distribution must be a continuous univariate distribution; got $(typeof(dist)). Distributions with point masses (e.g., Dirac) are not supported."))

"""
    UniformSum(a::Real, w1::Real, w2::Real)

Distribution of `X + Y` for independent `X ~ Uniform(a, a + w1)` and `Y ~ Uniform(0, w2)`, a trapezoid that rises from
`a` over the smaller width, is flat, and falls to zero at `a + w1 + w2` over the smaller width. This is the
distribution of extinctions in the MATCH model of differential extinction for young stars; see
[`match_extinction`](@ref). Supports `pdf`, `cdf`, `minimum`, `maximum`, `mean`, `var`, and `rand`.
"""
struct UniformSum{T <: Real} <: ContinuousUnivariateDistribution
    a::T
    λ1::T # smaller width
    λ2::T # larger width
    function UniformSum{T}(a, w1, w2) where T <: Real
        @argcheck w1 >= 0 && w2 >= 0 && w1 + w2 > 0 "Widths must be non-negative and not both zero."
        λ1, λ2 = minmax(w1, w2)
        new{T}(a, λ1, λ2)
    end
end
UniformSum(a::Real, w1::Real, w2::Real) = UniformSum{float(promote_type(typeof(a), typeof(w1), typeof(w2)))}(a, w1, w2)
Base.minimum(d::UniformSum) = d.a
Base.maximum(d::UniformSum) = d.a + d.λ1 + d.λ2
function Distributions.pdf(d::UniformSum, x::Real)
    ζ, a, b = x - d.a, d.λ1, d.λ2
    T = typeof(ζ / b)
    return ζ < 0 || ζ > a + b ? zero(T) : ζ < a ? T(ζ / (a * b)) : ζ < b ? T(1 / b) : T((a + b - ζ) / (a * b))
end
Distributions.logpdf(d::UniformSum, x::Real) = log(pdf(d, x))
function Distributions.cdf(d::UniformSum, x::Real)
    ζ, a, b = x - d.a, d.λ1, d.λ2
    T = typeof(ζ / b)
    return ζ <= 0 ? zero(T) : ζ < a ? T(ζ^2 / (2a * b)) : ζ < b ? T((2ζ - a) / (2b)) :
        ζ < a + b ? T(1 - (a + b - ζ)^2 / (2a * b)) : one(T)
end
Distributions.mean(d::UniformSum) = d.a + (d.λ1 + d.λ2) / 2
Distributions.var(d::UniformSum) = (d.λ1^2 + d.λ2^2) / 12
Base.rand(rng::AbstractRNG, d::UniformSum) = d.a + d.λ1 * rand(rng, typeof(d.a)) + d.λ2 * rand(rng, typeof(d.a))

"""
    ExponentialDiskExtinction(Av::Real, dAv::Real, s::Real; f_red::Real=1/2)

Distribution of V-band extinctions for stars in a disk behind a foreground screen of extinction `Av`. The stars are
embedded in a disk of dust with total extinction `dAv`. The stars and dust both have two-sided exponential
vertical profiles, and the scale height of the stars is `s` times that of the dust. Each star has extinction
`Av + dAv U`, where `U` is the fraction of the dust in front of it.
The midplane of the dust is offset from that of the stars so that a fraction `f_red` of the stars lie behind it.

With `s = 1` and `f_red = 1/2` the stars and dust are perfectly mixed and the distribution is a uniform tophat
between `Av` and `Av + dAv`, as in the model of MATCH [Dolphin2002](@citep). For `f_red = 1/2`, `U` has
density ``[2 \\, \\text{min}(u, 1-u)]^{1/s - 1} / s``, which for `s < 1` (stars embedded in the dust) peaks at
`u = 1/2` and for `s > 1` diverges at `u = 0` and `u = 1`, tending to equal point masses at both as `s → ∞` (a thin dust
layer). Supports `pdf`, `cdf`, `quantile`, `minimum`, `maximum`, `mean`, `var`, `rand`, and
`InitialMassFunctions.partial_mean`.

```jldoctest; setup = :(using StarFormationHistories: ExponentialDiskExtinction; using Distributions: cdf)
julia> d = ExponentialDiskExtinction(0.2, 1.0, 3.0);

julia> extrema(d)
(0.2, 1.2)

julia> cdf(d, 0.7) # half of the stars lie in front of the dust midplane
0.5
```
"""
struct ExponentialDiskExtinction{T <: Real} <: ContinuousUnivariateDistribution
    Av::T
    dAv::T
    s::T
    f_red::T
    z0::T # offset of the dust midplane from the stellar midplane, in dust scale heights
    function ExponentialDiskExtinction{T}(Av, dAv, s, f_red) where T <: Real
        @argcheck Av >= 0 && dAv > 0 && s > 0 "Requires Av >= 0, dAv > 0, and s > 0."
        @argcheck 0 < f_red < 1 "f_red must be between 0 and 1."
        # Quantile f_red of the stellar profile, Laplace(0, s)
        z0 = f_red < 1/2 ? s * log(2 * f_red) : -s * log(2 * (1 - f_red))
        new{T}(Av, dAv, s, f_red, z0)
    end
end
ExponentialDiskExtinction(Av::Real, dAv::Real, s::Real; f_red::Real=1/2) =
    ExponentialDiskExtinction{float(promote_type(typeof(Av), typeof(dAv), typeof(s), typeof(f_red)))}(Av, dAv, s, f_red)
Base.minimum(d::ExponentialDiskExtinction) = d.Av
Base.maximum(d::ExponentialDiskExtinction) = d.Av + d.dAv

# Closed forms for U on u <= 1/2, with k = (a, e^c, e^(a c)) for a = 1/s and c = z0; for u > 1/2 they follow from the
# reflection F(u; c) = 1 - F(1 - u; -c), with kf = (a, e^-c, e^(-a c)). The dust has U = 1/2 at height c, and stars
# above (below) height 0 have U <= 1/2 (> 1/2) where the dust is above (below) its midplane, giving the two pieces at
# u = e^c / 2.
# ∫_y^x v^(b-1) dv, accurate as b → 0
powint(x, y, b) = iszero(b) ? log(x / y) : y^b * expm1(b * log(x / y)) / b
diskF(u, (a, ec, eac)) = u <= ec / 2 ? (2u)^a / (2eac) : 1 - eac * (2u)^-a / 2
diskp(u, (a, ec, eac)) = u <= ec / 2 ? a * (2u)^(a - 1) / eac : a * eac * (2u)^(-a - 1)
diskQ(p, (a, ec, eac)) = p <= 1/2 ? ec * (2p)^(1 / a) / 2 : ec * (2 * (1 - p))^(-1 / a) / 2 # for p <= F(1/2)
function diskI(u, (a, ec, eac)) # ∫_0^u F
    τ = min(u, ec / 2)
    I = (2τ)^(a + 1) / (4 * (a + 1) * eac)
    return u > τ ? I + (u - τ) - eac * powint(2u, 2τ, 1 - a) / 4 : I
end
function diskJ(u, (a, ec, eac)) # ∫_0^u t F(t) dt
    τ = min(u, ec / 2)
    J = (2τ)^(a + 2) / (8 * (a + 2) * eac)
    return u > τ ? J + (u^2 - τ^2) / 2 - eac * powint(2u, 2τ, 2 - a) / 8 : J
end
function diskpars(d::ExponentialDiskExtinction)
    a, ec, eac = 1 / d.s, exp(d.z0), exp(d.z0 / d.s)
    return (a, ec, eac), (a, 1 / ec, 1 / eac)
end
# F, ∫_0^u F, and ∫_0^u t F(t) dt on all of [0, 1]
diskFU(u, k, kf) = u <= 1/2 ? diskF(u, k) : 1 - diskF(1 - u, kf)
diskIU(u, k, kf) = u <= 1/2 ? diskI(u, k) : diskI(1//2, k) + (u - 1//2) - diskI(1//2, kf) + diskI(1 - u, kf)
diskJU(u, k, kf) = u <= 1/2 ? diskJ(u, k) :
    diskJ(1//2, k) + (u^2 - 1//4) / 2 - diskI(1//2, kf) + diskI(1 - u, kf) + diskJ(1//2, kf) - diskJ(1 - u, kf)

function Distributions.cdf(d::ExponentialDiskExtinction, x::Real)
    return diskFU(clamp((x - d.Av) / d.dAv, 0, 1), diskpars(d)...)
end
function Distributions.pdf(d::ExponentialDiskExtinction, x::Real)
    k, kf = diskpars(d)
    u = (x - d.Av) / d.dAv
    return 0 <= u <= 1 ? (u <= 1/2 ? diskp(u, k) : diskp(1 - u, kf)) / d.dAv : zero(u)
end
Distributions.logpdf(d::ExponentialDiskExtinction, x::Real) = log(pdf(d, x))
function Distributions.quantile(d::ExponentialDiskExtinction, p::Real)
    k, kf = diskpars(d)
    return d.Av + d.dAv * (p <= 1 - d.f_red ? diskQ(p, k) : 1 - diskQ(1 - p, kf))
end
# E[U] = 1 - ∫_0^1 F and E[U^2] = 1 - 2 ∫_0^1 t F(t) dt
Distributions.mean(d::ExponentialDiskExtinction) = d.Av + d.dAv * (1 - diskIU(1, diskpars(d)...))
function Distributions.var(d::ExponentialDiskExtinction)
    m = 1 - diskIU(1, diskpars(d)...)
    return d.dAv^2 * (1 - 2 * diskJU(1, diskpars(d)...) - m^2)
end
Base.rand(rng::AbstractRNG, d::ExponentialDiskExtinction) = quantile(d, rand(rng, typeof(d.Av)))
function partial_mean(d::ExponentialDiskExtinction, x::Real)
    x <= d.Av && return zero(float(x))
    x >= maximum(d) && return mean(d)
    k, kf = diskpars(d)
    u = (x - d.Av) / d.dAv
    # x F(x) - ∫_Av^x F
    return x * diskFU(u, k, kf) - d.dAv * diskIU(u, k, kf)
end
# Each knot interval of a reddening path is divided into equal sub-segments (see deposit_paths!), whose probability,
# mean, and variance of extinction depend only on the interval and the number of sub-segments. They are tabulated once
# per template and shared by every source, so the extinction distribution is evaluated only a few hundred times.
# The mean and variance across [b0, b1] follow from the cdf alone, by parts: E[X - b0] = ∫(F1 - F) / ΔF and
# E[(X - b0)^2] = ∫2(x - b0)(F1 - F) / ΔF, where the cdf is bounded and continuous even where the density diverges or
# jumps (e.g., ExponentialDiskExtinction with s > 1, or the additional extinction of young stars added to it).
function segment_moments(dist, b0, b1, F0, F1)
    ΔF = F1 - F0
    m1 = quadgk(x -> F1 - cdf(dist, x), b0, b1; rtol=1e-8)[1] / ΔF
    m2 = quadgk(x -> 2 * (x - b0) * (F1 - cdf(dist, x)), b0, b1; rtol=1e-8)[1] / ΔF
    return b0 + m1, max(m2 - m1^2, zero(m2))
end
# (ΔF, Ā, v) of each sub-segment of knot interval n divided into m sub-segments, from the tables of interval n in `cache`
function segment_table!(cache, dist, knots, n, m)
    tabs = cache[n]
    length(tabs) < m && append!(tabs, [eltype(tabs)() for _ in 1:m-length(tabs)])
    if isempty(tabs[m])
        b = range(knots[n], knots[n+1]; length=m + 1)
        F = cdf.(dist, b)
        # Sub-segments without probability are skipped by deposit_paths!
        tabs[m] = [(F[s+1] - F[s], (F[s+1] > F[s] ? segment_moments(dist, b[s], b[s+1], F[s], F[s+1]) : (b[s], 0.0))...)
                   for s in 1:m]
    end
    return tabs[m]
end

"""
    UniformConvolution(base::Distributions.ContinuousUnivariateDistribution, w::Real)

Distribution of `X + Y` for independent `X ~ base` and `Y ~ Uniform(0, w)`, e.g., the additional differential extinction
of young stars in the model of MATCH [Dolphin2003](@citep) added to another distribution of extinctions; see
[`match_extinction`](@ref). `base` must be bounded. The density is at most `1 / w`, so it is finite even where that of
`base` is not. The `cdf` uses `InitialMassFunctions.partial_mean(base, x)`, which is integrated numerically unless `base`
has a closed form, as [`ExponentialDiskExtinction`](@ref) does. Supports `pdf`, `cdf`, `minimum`, `maximum`, `mean`,
`var`, and `rand`.
"""
struct UniformConvolution{T <: Real, D <: ContinuousUnivariateDistribution} <: ContinuousUnivariateDistribution
    base::D
    w::T
    function UniformConvolution{T, D}(base, w) where {T <: Real, D <: ContinuousUnivariateDistribution}
        @argcheck isfinite(minimum(base)) && isfinite(maximum(base)) "`base` must be bounded; truncate it, e.g., with `prepare_extinction`."
        @argcheck w > 0 "The width of the uniform term must be positive."
        new{T, D}(base, w)
    end
end
UniformConvolution(base::ContinuousUnivariateDistribution, w::Real) = UniformConvolution{float(typeof(w)), typeof(base)}(base, w)
Base.minimum(d::UniformConvolution) = minimum(d.base)
Base.maximum(d::UniformConvolution) = maximum(d.base) + d.w
Distributions.pdf(d::UniformConvolution, x::Real) = (cdf(d.base, x) - cdf(d.base, x - d.w)) / d.w
Distributions.logpdf(d::UniformConvolution, x::Real) = log(pdf(d, x))
# (1/w) ∫_{x-w}^x F, with ∫_{-∞}^x F = x F(x) - partial_mean(x)
function Distributions.cdf(d::UniformConvolution, x::Real)
    G(y) = y <= minimum(d.base) ? zero(float(y)) : y * cdf(d.base, y) - partial_mean(d.base, y)
    return clamp((G(x) - G(x - d.w)) / d.w, 0, 1)
end
Distributions.mean(d::UniformConvolution) = mean(d.base) + d.w / 2
Distributions.var(d::UniformConvolution) = Distributions.var(d.base) + d.w^2 / 12
Base.rand(rng::AbstractRNG, d::UniformConvolution) = rand(rng, d.base) + d.w * rand(rng, typeof(d.w))

"""
    match_extinction(Av::Real, dAv::Real, dAvy::Real, logAge::Real; t1::Real=4e7, t2::Real=1e8)

Returns the distribution of V-band extinctions for stars with logarithmic age `logAge` (`log10(age [yr])`) in the
differential extinction model of MATCH [Dolphin2002](@citep). Every star receives an extinction drawn
uniformly from `Av` to `Av + dAv`, and stars younger than `t2` years receive an independent additional extinction drawn
uniformly from 0 to `w(t) dAvy`, where `w(t)` is 1 for ages below `t1` and decreases linearly in age to 0 at `t2`
[Dolphin2003](@citep). Returns a `Distributions.Uniform` when only one of the two terms is nonzero, and a
[`UniformSum`](@ref) otherwise. Requires `dAv + w(t) dAvy > 0`.

    match_extinction(base::Distributions.ContinuousUnivariateDistribution, dAvy::Real, logAge::Real;
                     t1::Real=4e7, t2::Real=1e8)

Adds the additional extinction of young stars, as above, to another distribution of extinctions `base` (e.g., an
[`ExponentialDiskExtinction`](@ref)), returning `base` for stars older than `t2` and a [`UniformConvolution`](@ref)
otherwise.

```jldoctest; setup = :(using StarFormationHistories: match_extinction; using Distributions: Uniform)
julia> match_extinction(0.2, 0.6, 0.5, 9.0) == Uniform(0.2, 0.8) # older than t2; only dAv applies
true

julia> round.(extrema(match_extinction(0.2, 0.6, 0.5, 7.0)); digits=6) # younger than t1; both apply
(0.2, 1.3)
```
"""
function match_extinction(Av::Real, dAv::Real, dAvy::Real, logAge::Real; t1::Real=4e7, t2::Real=1e8)
    @argcheck Av >= 0 && dAv >= 0
    wy = young_width(dAvy, logAge, t1, t2)
    @argcheck dAv + wy > 0 "The extinction distribution has zero width; use the method of `partial_cmd_smooth` without extinction."
    iszero(wy) && return Uniform(Av, Av + dAv)
    iszero(dAv) && return Uniform(Av, Av + wy)
    return UniformSum(Av, dAv, wy)
end
function match_extinction(base::ContinuousUnivariateDistribution, dAvy::Real, logAge::Real; t1::Real=4e7, t2::Real=1e8)
    wy = young_width(dAvy, logAge, t1, t2)
    return iszero(wy) ? base : UniformConvolution(base, wy)
end
# Width of the additional extinction of young stars, tapering linearly in age from dAvy at t1 to 0 at t2
function young_width(dAvy, logAge, t1, t2)
    @argcheck dAvy >= 0 && t2 > t1
    return dAvy * clamp((t2 - exp10(logAge)) / (t2 - t1), 0, 1)
end

#################################
# Photometric models evaluated at one star

# Per-filter models are Tuples of callables (Vectors are converted by `astuple`); anything else is a joint callable
# taking one magnitude per filter. `m` is an SVector with one magnitude per filter.
astuple(f::AbstractVector) = Tuple(f)
astuple(f) = f
point_values(f::Tuple, m::SVector) = SVector(map((fk, mk) -> fk(mk), f, Tuple(m)))
point_values(f, m::SVector) = SVector(f(m...))
# Completeness over the filters of the Hess diagram (`third` is true when the y-axis filter is not in the color)
function point_completeness(f::Tuple, m::SVector, i1, i2, y, third)
    v = map((fk, mk) -> fk(mk), f, Tuple(m))
    return v[i1] * v[i2] * (third ? v[y] : one(v[i1]))
end
point_completeness(f, m::SVector, i1, i2, y, third) = f(m...)

#################################
# Template construction

"""
    result::StatsBase.Histogram =
        partial_cmd_smooth(m_ini::AbstractVector{<:Number},
                           mags::AbstractVector{<:AbstractVector{<:AbstractVector{<:Number}}},
                           knots::AbstractVector{<:Number},
                           extinction::Distributions.ContinuousUnivariateDistribution,
                           mag_err_funcs, y_index, color_indices, imf,
                           completeness_funcs=[one for i in first(mags)],
                           bias_funcs=[zero for i in first(mags)];
                           max_shift::Real=0.5, kws...)

Method for a simple stellar population whose stars have a distribution of V-band extinctions `extinction` (differential
extinction), for example a `Distributions.Uniform` for the model of MATCH [Dolphin2002](@citep) or the result
of [`match_extinction`](@ref). `extinction` is passed through [`prepare_extinction`](@ref), so unbounded distributions
are truncated and distributions with point masses are rejected.

`mags[n]` holds the isochrone magnitudes at extinction `knots[n]`, in the same format as the `mags` argument of the method
without extinction (one vector per filter, each with one entry per element of `m_ini`). The magnitudes are interpolated
linearly in extinction between knots, so the knots should include every extinction at which the magnitudes are not
linear, such as the extinction nodes of the bolometric correction grid (e.g., `BolometricCorrections.Av_nodes`), and
must bracket the support of `extinction`. `knots` must be sorted.

Each isochrone point is deposited along its path through the Hess diagram as extinction increases, in sub-segments
across which it moves by at most `max_shift` times the larger of its photometric error and the bin width. Each
sub-segment is a Gaussian kernel whose covariance includes the spread of extinction across the sub-segment, with the
photometric error, completeness, and bias evaluated at the mean extinction of the sub-segment. The template converges to
the exact mixture of templates over extinction as `max_shift` decreases; the default of 0.5 is accurate enough for fits of
``\\sim 10^5`` stars. Other arguments and keyword arguments are as for the method without extinction, including binary
models.
"""
function partial_cmd_smooth(m_ini::AbstractVector{<:Number},
                            mags::AbstractVector{<:AbstractVector{<:AbstractVector{<:Number}}},
                            knots::AbstractVector{<:Number},
                            extinction::ContinuousUnivariateDistribution,
                            mag_err_funcs, y_index, color_indices, imf,
                            completeness_funcs=[one for i in first(mags)],
                            bias_funcs=[zero for i in first(mags)];
                            dmod::Number=0, normalize_value::Number=1,
                            binary_model::AbstractBinaryModel=NoBinaries(),
                            mean_mass::Number=mean(imf), max_shift::Real=0.5,
                            edges=nothing, xlim=nothing, ylim=nothing, nbins=nothing,
                            xwidth=nothing, ywidth=nothing)
    edges = calculate_edges(edges, xlim, ylim, nbins, xwidth, ywidth)
    bfrac = binary_system_fraction(binary_model)
    @argcheck bfrac <= 1
    single, binary = reddened_hess(m_ini, mags, knots, extinction, mag_err_funcs, y_index, color_indices, imf,
                                   completeness_funcs, bias_funcs, edges, iszero(bfrac) ? nothing : binary_model;
                                   dmod, normalize_value, mean_mass, max_shift)
    isnothing(binary) && return single
    # As in the method without extinction: single-star systems hold 1 - binary_mass_fraction of the mass formed
    return Histogram(edges, single.weights .* (1 - binary_mass_fraction(binary_model, imf)) .+ binary.weights, :left, false)
end

"""
    (single, binary) = ssp_components(m_ini::AbstractVector{<:Number},
                                      mags::AbstractVector{<:AbstractVector{<:AbstractVector{<:Number}}},
                                      knots::AbstractVector{<:Number},
                                      extinction::Distributions.ContinuousUnivariateDistribution,
                                      mag_err_funcs, y_index, color_indices, imf,
                                      completeness_funcs=[one for i in first(mags)],
                                      bias_funcs=[zero for i in first(mags)]; kws...)

Method of [`ssp_components`](@ref StarFormationHistories.ssp_components) for a distribution of extinctions; arguments are
as for the corresponding method of [`partial_cmd_smooth`](@ref).
"""
function ssp_components(m_ini::AbstractVector{<:Number},
                        mags::AbstractVector{<:AbstractVector{<:AbstractVector{<:Number}}},
                        knots::AbstractVector{<:Number},
                        extinction::ContinuousUnivariateDistribution,
                        mag_err_funcs, y_index, color_indices, imf,
                        completeness_funcs=[one for i in first(mags)],
                        bias_funcs=[zero for i in first(mags)];
                        binary_model::AbstractBinaryModel=NoBinaries(),
                        dmod::Number=0, normalize_value::Number=1, mean_mass::Number=mean(imf), max_shift::Real=0.5,
                        edges=nothing, xlim=nothing, ylim=nothing, nbins=nothing, xwidth=nothing, ywidth=nothing)
    edges = calculate_edges(edges, xlim, ylim, nbins, xwidth, ywidth)
    return reddened_hess(m_ini, mags, knots, extinction, mag_err_funcs, y_index, color_indices, imf,
                         completeness_funcs, bias_funcs, edges,
                         binary_model isa NoBinaries ? nothing : with_fraction(binary_model, 1);
                         dmod, normalize_value, mean_mass, max_shift)
end

# Single-star Hess diagram and, if `binary_model` is not `nothing`, the binary Hess diagram for `binary_model`, both with
# extinctions distributed as `extinction`
function reddened_hess(m_ini, mags, knots, extinction, mag_err_funcs, y_index, color_indices, imf, completeness_funcs,
                       bias_funcs, edges, binary_model; dmod, normalize_value, mean_mass, max_shift)
    @argcheck length(color_indices) == 2
    @argcheck max_shift > 0
    dist = prepare_extinction(extinction)
    lo, hi = extrema(dist)
    @argcheck length(knots) == length(mags) >= 2 "`knots` and `mags` must have the same length, at least 2."
    @argcheck issorted(knots) "`knots` must be sorted."
    @argcheck first(knots) <= lo && last(knots) >= hi "`knots` must bracket the support of the extinction distribution, $((lo, hi))."
    NF = length(first(mags))
    @argcheck all(length(mk) == NF for mk in mags) && all(length(m) == length(m_ini) for mk in mags for m in mk)
    @argcheck all(length(f) == NF for f in (mag_err_funcs, completeness_funcs, bias_funcs) if f isa Union{Tuple, AbstractVector})
    @argcheck all(checkbounds(Bool, 1:NF, idx) for idx in union(y_index, color_indices))
    # Knots within the support, with magnitudes at its ends interpolated linearly between the bracketing knots
    interior = [k for k in eachindex(knots) if lo < knots[k] < hi]
    mags_at(A) = let n = clamp(searchsortedlast(knots, A), firstindex(knots), lastindex(knots) - 1),
                     t = (A - knots[n]) / (knots[n+1] - knots[n])
        [m0 .+ t .* (m1 .- m0) for (m0, m1) in zip(mags[n], mags[n+1])]
    end
    kn = [float(lo); float.(knots[interior]); float(hi)]
    km = [[mags_at(lo)]; [mags[k] for k in interior]; [mags_at(hi)]]
    return reddened_hess(Val(NF), m_ini, km, kn, dist, astuple(mag_err_funcs), y_index, color_indices, imf,
                         astuple(completeness_funcs), astuple(bias_funcs), edges, binary_model;
                         dmod, normalize_value, mean_mass, max_shift)
end
function reddened_hess(::Val{NF}, m_ini, mags, knots, dist, errs, y_index, color_indices, imf, comps, biases, edges,
                       binary_model; dmod, normalize_value, mean_mass, max_shift) where NF
    nk = length(knots)
    i1, i2 = color_indices
    # Covariance term of GaussianKernel and the color filter whose error is σx when the y-axis filter is in the color
    c = y_index == i1 ? -1.0 : y_index == i2 ? 1.0 : 0.0
    xi = y_index == i1 ? i2 : i1
    third = !(y_index in color_indices)
    # Ingest all knots and filters together so that every knot is interpolated on the same initial masses
    mi, cols = ingest_isochrone_filters(m_ini, [mags[n][f] for n in 1:nk for f in 1:NF])
    col(n, f) = cols[(n-1)*NF+f]
    # Densify once, with the spacing rule of the method without extinction applied to the least extincted magnitudes,
    # which have the smallest errors
    m0 = [col(1, f) for f in 1:NF]
    err = filter_values(errs, [m .+ dmod for m in m0])
    σx0 = third ? hypot.(err[i1], err[i2]) : err[xi]
    Δmag = min.(step(edges[1]), step(edges[2]), max.(min.(σx0, err[y_index]), 0.01) ./ 2)
    new_mini, spacing = mini_spacing(mi, m0[i1] .- m0[i2], m0[y_index], Δmag, true)
    dense = [[interpolate_ingested(mi, col(n, f), new_mini, dmod) for f in 1:NF] for n in 1:nk] # [knot][filter][point]
    pmags = [[SVector{NF}(ntuple(f -> dense[n][f][j], Val(NF))) for j in eachindex(new_mini)] for n in 1:nk]
    imfw = [dispatch_imf(imf, m) * normalize_value / mean_mass for m in new_mini]
    # Single stars: each densified interval, with its two end points averaged as in the method without extinction
    # Sub-segment tables shared by the single stars and binary systems, one vector of tables per knot interval
    T = eltype(knots)
    cache = [Vector{NTuple{3, T}}[] for _ in 1:nk-1]
    single = zeros(length(edges[1]) - 1, length(edges[2]) - 1)
    nint = length(new_mini) - 1
    deposit_paths!(single, [view(p, 1:nint) for p in pmags], [view(p, 2:nint+1) for p in pmags],
                   spacing .* view(imfw, 1:nint), spacing .* view(imfw, 2:nint+1), knots, dist, errs, comps, biases,
                   i1, i2, y_index, xi, c, third, edges, max_shift, cache)
    single_hist = Histogram(edges, single, :left, false)
    isnothing(binary_model) && return single_hist, nothing
    # Binary systems: as in binary_hess, aggregated in cells of half the bin width by their observed position at the
    # least extincted knot, with each cell's mean magnitudes at every knot defining its path
    m1, m2, w = binary_systems(binary_model, new_mini[begin:4:end], imf; mean_mass)
    b1, b2 = bracket.(Ref(new_mini), m1), bracket.(Ref(new_mini), m2)
    sys = [map(dense[n]) do mag
               F = mag2flux.(mag)
               flux2mag.(lerp_bracket.(Ref(F), b1) .+ lerp_bracket.(Ref(F), b2))
           end for n in 1:nk] # [knot][filter][system]
    lit = findall(isfinite, first(first(sys)))
    sys = [[s[lit] for s in sn] for sn in sys]
    obs = filter_values(biases, sys[1]) .+ sys[1]
    W, _, _, M = aggregate_cells(w[lit] .* normalize_value, obs[i1] .- obs[i2], obs[y_index],
                                 [sys[n][f] for n in 1:nk for f in 1:NF], edges)
    cmags = [[SVector{NF}(ntuple(f -> M[(n-1)*NF+f][k], Val(NF))) for k in eachindex(W)] for n in 1:nk]
    binary = zeros(length(edges[1]) - 1, length(edges[2]) - 1)
    deposit_paths!(binary, cmags, cmags, W, W, knots, dist, errs, comps, biases, i1, i2, y_index, xi, c, third, edges,
                   max_shift, cache)
    return single_hist, Histogram(edges, binary, :left, false)
end

# Deposits sources along their reddening paths. Source k has magnitudes A[n][k] and B[n][k] (SVectors over filters) at
# knot n, which are averaged, and weights w0[k] and w1[k] multiplying the completeness at A and B; a single star is an
# isochrone interval with end points A and B, and a cell of binary systems has A = B. The type parameters force
# specialization on the observational models, which are otherwise only passed to the closure `props` and so would be
# called by dynamic dispatch when they are single joint callables.
function deposit_paths!(data, A, B, w0, w1, knots, dist, errs::E, comps::C, biases::BI, i1, i2, y_index, xi, c, third, edges,
                        max_shift, cache) where {E, C, BI}
    xe, ye = edges
    Δx, Δy, xmax, ymax = step(xe), step(ye), last(xe), last(ye)
    hist = Histogram(edges, data, :left, false)
    erfs = similar(data, size(data, 1) + 1)
    # Observed position, errors in the parameterization of GaussianKernel, and completeness-weighted weight of source k
    # at extinction a in knot interval n
    @inline function props(k, n, a)
        t = (a - knots[n]) / (knots[n+1] - knots[n])
        ma = A[n][k] + t * (A[n+1][k] - A[n][k])
        mb = B[n][k] + t * (B[n+1][k] - B[n][k])
        ea, eb = point_values(errs, ma), point_values(errs, mb)
        oa, ob = ma + point_values(biases, ma), mb + point_values(biases, mb)
        Ca = point_completeness(comps, ma, i1, i2, y_index, third)
        Cb = point_completeness(comps, mb, i1, i2, y_index, third)
        σ = (ea + eb) / 2
        σx = third ? hypot(σ[i1], σ[i2]) : σ[xi]
        return (x=((oa[i1] - oa[i2]) + (ob[i1] - ob[i2])) / 2, y=(oa[y_index] + ob[y_index]) / 2,
                σx=σx, σy=σ[y_index], w=(w0[k] * Ca + w1[k] * Cb) / 2)
    end
    for k in eachindex(w0)
        for n in 1:length(knots)-1
            a0, a1 = knots[n], knots[n+1]
            # Rates of change of color and magnitude with extinction along this knot interval
            dm = ((A[n+1][k] - A[n][k]) + (B[n+1][k] - B[n][k])) / (2 * (a1 - a0))
            dx, dy = dm[i1] - dm[i2], dm[y_index]
            P0 = props(k, n, a0)
            # Motion per unit extinction in resolution elements, at the start of the interval where errors are smallest
            q = max(abs(dx) / max(sqrt(P0.σx^2 + c^2 * P0.σy^2), Δx), abs(dy) / max(P0.σy, Δy))
            # Extinction moves stars fainter and redder, so the path ends once the star is more than 4σ beyond the faint
            # or red edge of the diagram, where it contributes nothing; the errors at the end of the interval are the
            # largest, so this is conservative
            aend = a1
            if dx > 0 && dy > 0
                P1 = props(k, n, a1)
                aend = min(a1, a0 + min((ymax + 4 * P1.σy - P0.y) / dy,
                                        (xmax + 4 * sqrt(P1.σx^2 + c^2 * P1.σy^2) - P0.x) / dx))
                aend <= a0 && break # later knot intervals are fainter and redder still
            end
            # Sub-segments divide the whole knot interval, so that their tables are shared by all sources, and the
            # path stops at the first one that starts beyond aend
            nsub = max(1, ceil(Int, (a1 - a0) * q / max_shift))
            tab = segment_table!(cache, dist, knots, n, nsub)
            for s in 1:nsub
                a0 + (a1 - a0) * (s - 1) / nsub >= aend && break
                ΔF, Ā, v = tab[s]
                ΔF > 0 || continue
                P = props(k, n, Ā)
                wt = P.w * ΔF
                wt > 0 || continue
                # Add the covariance v (dx, dy) (dx, dy)ᵀ of the extinction spread to that of the photometric errors,
                # expressed in the parameterization of GaussianKernel
                vy = P.σy^2 + v * dy^2
                vx = P.σx^2 + c^2 * P.σy^2 + v * dx^2
                cxy = -c * P.σy^2 + v * dx * dy
                addstar!(hist, GaussianKernel(P.x, P.y, sqrt(max(vx - cxy^2 / vy, 0)), sqrt(vy), -cxy / vy, wt); erfs)
            end
        end
    end
    return data
end
