module StarFormationHistories

using Compat: allequal
using ArgCheck: @argcheck
using Distributions: Distribution, Sampleable, Univariate, Continuous, pdf, logpdf,
    cdf, quantile, Multivariate, MvNormal, sampler, Uniform
import Distributions: _rand! # Extending
import DynamicHMC  # For random uncertainties in SFH fits
using Interpolations: interpolate, Gridded, Linear, deduplicate_knots!, extrapolate, Flat
using IrrationalConstants: logten
import LBFGSB # Used for one method in fitting.jl
import LineSearches # For configuration of Optim.jl
# Need mul! for composite!, ∇loglikelihood!;
using LinearAlgebra: BLAS, diag, Hermitian, mul! # hermitianpart
import LogDensityProblems # For interfacing with DynamicHMC
using LoopVectorization: @turbo
import Optim
import NLSolversBase
using PDMats: PDMat
using Printf: @sprintf
import ProgressMeter
using QuadGK: quadgk # For general mean(imf::UnivariateDistribution{Continuous}; kws...)
using Random: AbstractRNG, default_rng, seed!
using Roots: find_zero # For mass_limits in simulate.jl
import Tables # For Tables.rows, process_ASTs
using TaskLocalValues: TaskLocalValue # For threadsafe buffers
import Trapz: trapz
using SpecialFunctions: erf
using StaticArrays: SVector, sacollect
using StatsBase: fit, Histogram, Weights
import StatsBase: mean, median, mode, std # Extending
import KissMCMC
import MCMCChains

# Code inclusion
include("utilities.jl")
include("simulate.jl")
include(joinpath("fitting", "fitting.jl")) # This will include other relevant files

##################################
# Isochrone utilities

"""
    new_mags::Vector = interpolate_mini(m_ini, mags::Vector{<:Real}, new_mini)
    new_mags::Vector{Vector} = interpolate_mini(m_ini, mags::Vector{Vector{<:Real}},
                                                new_mini)
    new_mags::Matrix = interpolate_mini(m_ini, mags::AbstractMatrix, new_mini)

Function to interpolate `mags` as a function of initial mass vector `m_ini` onto a new initial mass vector `new_mini`. `mags` can either be a vector of equal length to `m_ini`, designating magnitudes in a single filter, or a vector of vectors, designating multiple filters, or a matrix with each column designating a different filter. The isochrone points should be in evolutionary order; see [`StarFormationHistories.ingest_isochrone`](@ref) for how `m_ini` is prepared for interpolation.

# Examples
```jldoctest
julia> m_ini = [0.08, 0.10, 0.12, 0.14, 0.16];

julia> mags = [13.545, 12.899, 12.355, 11.459, 10.947];

julia> new_mini = 0.08:0.01:0.16;

julia> result = [13.545, 13.222, 12.899, 12.626999999999999, 12.355, 11.907,
                 11.459, 11.203, 10.947];

julia> StarFormationHistories.interpolate_mini(m_ini, mags, new_mini) ≈ result
true

julia> StarFormationHistories.interpolate_mini(m_ini, [mags, mags], new_mini) ≈
           [result, result] # Vector{Vector{<:Real}}
true

julia> StarFormationHistories.interpolate_mini(m_ini, [mags mags], new_mini) ≈
           [result result] # Matrix{<:Real}
true
```
"""
function interpolate_mini(m_ini::AbstractVector{<:Number}, mags::AbstractVector{<:Number},
                          new_mini::AbstractVector{<:Number})
    m_ini, mags = ingest_isochrone(m_ini, mags) # from simulate.jl include'd above
    return interpolate((m_ini,), mags, Gridded(Linear()))(new_mini)
end
# Filters are ingested together so that every filter is interpolated on the same initial masses
function interpolate_mini(m_ini::AbstractVector{<:Number}, mags::AbstractVector{<:AbstractVector{<:Number}},
                          new_mini::AbstractVector{<:Number})
    m_ini, point_mags = ingest_isochrone(m_ini, [getindex.(mags, i) for i in eachindex(m_ini)])
    return [interpolate((m_ini,), getindex.(point_mags, k), Gridded(Linear()))(new_mini) for k in eachindex(mags)]
end
interpolate_mini(m_ini::AbstractVector{<:Number}, mags::AbstractMatrix{<:Number}, new_mini::AbstractVector{<:Number}) =
    reduce(hcat, interpolate_mini(m_ini, [collect(i) for i in eachcol(mags)], new_mini))

"""
    mini_spacing(m_ini::AbstractVector{<:Number},
                 colors::AbstractVector{<:Number},
                 mags::AbstractVector{<:Number},
                 Δmag,
                 ret_spacing::Bool = false)

Returns a new sampling of stellar masses given the initial mass vector `m_ini` from an isochrone and the corresponding y-axis magnitude vector `mags` and x-axis color vector `color` to be used to construct a model Hess diagram. The isochrone points should be in evolutionary order, and the returned masses are on the scale of the initial masses prepared by [`StarFormationHistories.ingest_isochrone`](@ref). Will compute the new initial mass vector such that the distance between adjacent isochrone points is less than `Δmag`. Will return the change in mass between points `diff(new_mini)` if `ret_spacing==true`.

```julia
julia> mini_spacing([0.08, 0.10, 0.12, 0.14, 0.16],
                    [1.0, 0.99, 0.98, 0.97, 0.96],
                    [13.545, 12.899, 12.355, 11.459, 10.947],
                    0.1, false)
```
"""
function mini_spacing(m_ini::AbstractVector{<:Number},
                      colors::AbstractVector{<:Number},
                      mags::AbstractVector{<:Number},
                      Δmag,
                      ret_spacing::Bool=false)
    @argcheck axes(m_ini) == axes(mags) == axes(colors)
    # Walk along the isochrone in evolutionary order with initial masses increasing along it
    m_ini, points = ingest_isochrone(m_ini, SVector.(colors, mags))
    colors, mags = first.(points), last.(points)
    new_mini = Float64[first(m_ini)]
    # Loop through the indices, testing if adjacent CMD points are
    # different by less than Δm.
    for i in eachindex(m_ini, mags, colors)[begin:end-1]
        Δx = colors[i+1] - colors[i]
        Δy = mags[i+1] - mags[i]
        diffi = hypot(Δx, Δy)
        if diffi > Δmag # Requires interpolation
            npoints = round(Int, diffi / Δmag, RoundUp)
            Δmass = m_ini[i+1] - m_ini[i]
            mass_step = Δmass / npoints
            for j in 1:npoints
                push!(new_mini, m_ini[i] + mass_step*j)
            end
        else # Does not require interpolation
            push!(new_mini, m_ini[i+1])
        end
    end
    # Return only unique entries from new_mini
    new_mini = unique(new_mini)
    if !ret_spacing
        return new_mini
    else
        new_spacing = diff(new_mini)
        return new_mini, new_spacing
    end
end

"""
    dispatch_imf(imf, m) = imf(m)
    dispatch_imf(imf::Distributions.ContinuousUnivariateDistribution, m) =
        Distributions.pdf(imf, m)

Simple function barrier for [`partial_cmd`](@ref) and [`partial_cmd_smooth`](@ref). If you provide a generic functional that takes one argument (mass) and returns the PDF, then it uses the first definition. If you provide a `Distributions.ContinuousUnivariateDistribution`, this will convert the function call into the correct `pdf` call.
"""
dispatch_imf(imf, m) = imf(m)
dispatch_imf(imf::Distribution{Univariate, Continuous}, m) = pdf(imf, m)

# Photometric completeness, error, and bias models are either a Tuple or AbstractVector of single-filter
# callables, or one joint callable taking one magnitude per filter. `mags` holds one magnitude vector per filter.
# Completeness: per-filter models are multiplied over the filters in `idxs`; a joint model returns the joint completeness.
completeness_values(completeness::Union{Tuple, AbstractVector}, mags::AbstractVector{<:AbstractVector}, idxs=eachindex(mags)) = .*((completeness[k].(mags[k]) for k in idxs)...)
completeness_values(completeness, mags::AbstractVector{<:AbstractVector}, idxs=nothing) = completeness.(mags...)
# Errors and biases: returns one vector per filter; a joint model returns one value per filter for each star
filter_values(f::Union{Tuple, AbstractVector}, mags::AbstractVector{<:AbstractVector}) = (@argcheck length(f) == length(mags); [fk.(m) for (fk, m) in zip(f, mags)])
filter_values(f, mags::AbstractVector{<:AbstractVector}) = (r = f.(mags...); [getindex.(r, k) for k in eachindex(mags)])

"""
    mean(imf::Distributions.ContinuousUnivariateDistribution; kws...) =
        quadgk(x->x*pdf(imf,x), extrema(imf)...; kws...)[1]

Simple one-liner that calculates the mean of the provided `imf` distribution using numerical integration via `QuadGK.quadgk` with the passed keyword arguments `kws...`. This is a generic fallback; better to define this explicitly for your IMF model. Requires that `pdf(imf,x)` and `extrema(imf)` be defined.
"""
mean(imf::Distribution{Univariate, Continuous}; kws...) =
     quadgk(x -> x * pdf(imf, x), extrema(imf)...; kws...)[1]

##################################
# KDE models

# Gaussian kernel for a star in a Hess diagram with x-axis color and y-axis magnitude, in data units. y is Gaussian with
# standard deviation σy about y0, and the color given y is Gaussian with standard deviation σx about
# x0 - cov_mult * (y - y0). `cov_mult` is 1 for y=V and x=B-V, -1 for y=B and x=B-V (the y-axis filter also enters the
# color), and 0 for y=R and x=B-V (no covariance, in which case σx is the error in the color). `A` is the total weight.
struct GaussianKernel{T <: Real}
    x0::T
    y0::T
    σx::T
    σy::T
    cov_mult::T
    A::T
    function GaussianKernel{T}(x0, y0, σx, σy, cov_mult, A=1) where T <: Real
        @argcheck cov_mult in (-1, 0, 1)
        new{T}(x0, y0, σx, σy, cov_mult, A)
    end
end
GaussianKernel(x0::Real, y0::Real, σx::Real, σy::Real, cov_mult::Real, A::Real=1) =
    GaussianKernel{promote_type(typeof(x0), typeof(y0), typeof(σx), typeof(σy), typeof(cov_mult), typeof(A))}(x0, y0, σx, σy, cov_mult, A)

# Adds the kernel `obj`, integrated over each pixel, to the Hess diagram `image`, which must have uniformly spaced bin
# edges. The kernel is evaluated within `k` standard deviations of its center along y and, in each row, along the
# color. The integral over each pixel row in y is exact. Each row is oversampled in y by sub-rows across which the
# mean color shifts by at most σx, and within each sub-row the color distribution is approximated as a Gaussian with
# the mean and variance of the color over the sub-row, which is integrated exactly over each pixel in color. Without
# covariance there is one sub-row per row and the integral is exact. The total weight added is exactly A when the
# kernel lies within `image`.
function addstar!(image::Histogram, obj::GaussianKernel; k::Real=4)
    data, (xe, ye) = image.weights, image.edges
    Base.require_one_based_indexing(data)
    all(Base.Fix2(isa, AbstractRange), image.edges) || throw(ArgumentError("The `image::Histogram` provided to `addstar!` must have bin edges `image.edges` that are subtypes of `AbstractRange`."))
    (; x0, y0, σx, σy, cov_mult, A) = obj
    c = cov_mult
    Δx, Δy, x1, y1 = step(xe), step(ye), first(xe), first(ye)
    # Pixel rows within k σy of the center
    jlo = max(1, floor(Int, (y0 - k * σy - y1) / Δy) + 1)
    jhi = min(size(data, 2), floor(Int, (y0 + k * σy - y1) / Δy) + 1)
    # The mean color shifts by |c| Δy / nsub across one sub-row
    nsub = iszero(c) ? 1 : clamp(ceil(Int, abs(c) * Δy / σx), 1, 16)
    erfs = similar(data, size(data, 1) + 1) # erf at the color bin edges of the current sub-row
    # Column range and erf values are recomputed only when the color distribution changes, which without covariance
    # happens only for the first row
    xprev, σprev, ilo, ihi = oftype(float(x0), NaN), oftype(float(σx), NaN), 1, 0
    for j in jlo:jhi, s in 1:nsub
        # Standardized limits of the sub-row in y, its probability P, and the mean μ and variance v of y - y0 within it
        a = (y1 + (j - 1 + (s - 1) / nsub) * Δy - y0) / σy
        b = (y1 + (j - 1 + s / nsub) * Δy - y0) / σy
        P = (erf(b / sqrt(2)) - erf(a / sqrt(2))) / 2
        P > 0 || continue
        φa, φb = exp(-a^2 / 2) / sqrt(2π), exp(-b^2 / 2) / sqrt(2π)
        μ = σy * (φa - φb) / P
        v = σy^2 * (1 + (a * φa - b * φb) / P - ((φa - φb) / P)^2)
        xc, σc = x0 - c * μ, sqrt(σx^2 + c^2 * max(v, 0))
        if (xc != xprev) || (σc != σprev)
            ilo = max(1, floor(Int, (xc - k * σc - x1) / Δx) + 1)
            ihi = min(size(data, 1), floor(Int, (xc + k * σc - x1) / Δx) + 1)
            if ilo <= ihi
                scale = 1 / (sqrt(2) * σc)
                @turbo for i in ilo:ihi+1
                    erfs[i] = erf((x1 + (i - 1) * Δx - xc) * scale)
                end
            end
            xprev, σprev = xc, σc
        end
        ilo > ihi && continue
        w = A * P / 2 # The integral of the color Gaussian over a bin is half the difference of erf at its edges
        @turbo for i in ilo:ihi
            data[i, j] += w * (erfs[i+1] - erfs[i])
        end
    end
    return image
end

#########################################
# 2D Histogram construction and utilities

"""
    midpoints(v::AbstractVector, ranges::Bool=true)
Given an input vector `v`, returns a `length(v)-1` vector containing the midpoints between the original values in `v`. If `ranges == true`, returns the midpoints as a range if the differences between values in `v` (i.e., `diff(v)`) are all approximately equal. 
"""
function midpoints(v::AbstractVector, ranges::Bool=true)
    d = diff(v)
    # Aggressively convert to range if all steps are approximately equal
    if all(Base.Fix1(isapprox, first(d)), d) && ranges
        vstep = first(d)
        start = (v[begin]+vstep/2)
        stop = (v[end]-vstep/2)
        return range(start, stop; length=length(v)-1)
    end
    result = Vector{eltype(v)}(undef, length(v) - 1)
    for i in eachindex(result)
        result[i] = v[i] + d[i] / 2
    end
    return result
end
"""
    midpoints(v::AbstractRange, ranges::Bool=true)
Given an input range `v`, returns a `length(v)-1` range containing the midpoints between the original values in `v`.
"""
midpoints(v::AbstractRange, ranges::Bool=true) = first(v) + step(v)/2:step(v):last(v) - step(v)/2

"""
    calculate_edges(edges, xlim, ylim, nbins, xwidth, ywidth)

Function to calculate the bin edges for 2D histograms. Returns `(xbins, ybins)` with both entries being ranges.

# Keyword Arguments
 - `edges` is a tuple of ranges defining the left-side edges of the bins along the x-axis (edges[1]) and the y-axis (edges[2]). Example: `(-1.0:0.1:1.5, 22:0.1:27.2)`. If `edges` is provided, it will simply be returned.
 - `xlim` is a length-2 indexable object (e.g., a Vector{Float64} or NTuple{2,Float64)) giving the lower and upper bounds on the x-axis corresponding to the provided `colors` array. Example: `[-1.0, 1.5]`. This is only used if `edges==nothing`.
 - `ylim` is like `xlim` but  for the y-axis corresponding to the provided `mags` array. Example `[25, 20]`. This is only used if `edges==nothing`.
 - `nbins::NTuple{2,<:Integer}` is a 2-tuple of integers providing the number of bins to use along the x- and y-axes. This is only used if `edges==nothing`.
 - `xwidth` is the bin width along the x-axis for the `colors` array. This is only used if `edges==nothing` and `nbins==nothing`. Example: `0.1`. 
 - `ywidth` is like `xwidth` but for the y-axis corresponding to the provided `mags` array. Example: `0.1`.
"""
function calculate_edges(edges, xlim, ylim, nbins, xwidth, ywidth)
    if edges !== nothing
        if edges isa Tuple{<:AbstractRange, <:AbstractRange}
            return edges
        else
            throw(ArgumentError("When passing the `edges` keyword directly, it must be of type `Tuple{<:AbstractRange, <:AbstractRange}`; for example, `edges = (-1.0:0.1:1.5, 22:0.1:27.2)`."))
        end
    else 
        xlim, ylim = sort(xlim), sort(ylim)
        # Calculate nbins if it hasn't been provided. 
        if nbins === nothing
            if xwidth !== nothing && ywidth !== nothing

                xbins = round(Int, (xlim[2]-xlim[1])/xwidth)
                ybins = round(Int, (ylim[2]-ylim[1])/ywidth)
                nbins = (xbins, ybins)
            else
                throw(ArgumentError("If the keyword arguments `edges` and `nbins` are not
                                    provided, then `xwidth` and `ywidth` must be provided.")) 
            end
        end
        edges = (range(xlim[1], xlim[2], length=nbins[1]),
                 range(ylim[1], ylim[2], length=nbins[2]))
        return edges
    end
end

"""
    result::StatsBase.Histogram =
       bin_cmd(colors::AbstractVector{<:Number},
               mags::AbstractVector{<:Number};
               weights::AbstractVector{<:Number} = ones(promote_type(eltype(colors),
                                                        eltype(mags)), size(colors)),
               edges  = nothing,
               xlim   = extrema(colors),
               ylim   = extrema(mags),
               nbins  = nothing,
               xwidth = nothing,
               ywidth = nothing)

Returns a `StatsBase.Histogram` type containing the Hess diagram from the provided x-axis photometric `colors` and y-axis photometric magnitudes `mags`. These must all be vectors equal in length. You can either specify the bin edges directly via the `edges` keyword (e.g., `edges = (range(-0.5, 1.6, length=100), range(17.0, 26.0, length=100))`), or you can set the x- and y-limits via `xlim` and `ylim` and the number of bins as `nbins`, or you can omit `nbins` and instead pass the bin width in the x and y directions, `xwidth` and `ywidth`. See below for more info on the keyword arguments. To plot this with `PyPlot.jl` you should do `PyPlot.imshow(permutedims(result.weights), origin="lower", extent=(extrema(result.edges[1])..., extrema(result.edges[2]), kws...)` where `kws...` are any other keyword arguments you wish to pass to `PyPlot.imshow`.

# Keyword Arguments
 - `weights::AbstractVector{<:Number}` is a array of length equal to `colors` and `mags` that contains the probabilistic weights associated with each point. This is passed to `StatsBase.fit` as `StatsBase.Weights(weights)`. The following keyword arguments are passed to [`StarFormationHistories.calculate_edges`](@ref) to determine the bin edges of the histogram.
 - `edges` is a tuple of ranges defining the left-side edges of the bins along the x-axis (edges[1]) and the y-axis (edges[2]). Example: `(-1.0:0.1:1.5, 22:0.1:27.2)`. If `edges` is provided, `weights` is the only other keyword that will be read; `edges` supercedes the other construction methods. 
 - `xlim` is a length-2 indexable object (e.g., a vector or tuple) giving the lower and upper bounds on the x-axis corresponding to the provided `colors` array. Example: `[-1.0, 1.5]`. This is only used if `edges` is not provided. 
 - `ylim` is like `xlim` but  for the y-axis corresponding to the provided `mags` array. Example `[25.0, 20.0]`. This is only used if `edges` is not provided.
 - `nbins::NTuple{2, <:Integer}` is a 2-tuple of integers providing the number of bins to use along the x- and y-axes. This is only used if `edges` is not provided.
 - `xwidth` is the bin width along the x-axis for the `colors` array. This is only used if `edges` and `nbins` are not provided. Example: `0.1`. 
 - `ywidth` is like `xwidth` but for the y-axis corresponding to the provided `mags` array. Example: `0.1`.
"""
function bin_cmd(colors::AbstractVector{<:Number},
                 mags::AbstractVector{<:Number};
                 weights::AbstractVector{<:Number} =
                     ones(promote_type(eltype(colors), eltype(mags)), length(colors)),
                 edges=nothing, xlim=extrema(colors), ylim=extrema(mags), nbins=nothing,
                 xwidth=nothing, ywidth=nothing )
    @argcheck length(colors) == length(mags) == length(weights)
    edges = calculate_edges(edges, xlim, ylim, nbins, xwidth, ywidth)
    return fit(Histogram, (colors, mags), Weights(weights), edges; closed=:left)
end

"""
    result::StatsBase.Histogram =
        bin_cmd_smooth(colors,
                       mags,
                       color_err,
                       mag_err;
                       weights = ones(promote_type(eltype(colors), eltype(mags)),
                                      size(colors)),
                       edges   = nothing,
                       xlim    = extrema(colors),
                       ylim    = extrema(mags),
                       nbins   = nothing,
                       xwidth  = nothing,
                       ywidth  = nothing)

Returns a `StatsBase.Histogram` type containing the Hess diagram where the points have been smoothed using a 2D asymmetric Gaussian with widths given by the provided `color_err` and `mag_err` and weighted by the given `weights`. These arrays must all be equal in size. This is akin to a KDE where each point is broadened by its own probability distribution. Keyword arguments are as explained in [`bin_cmd_smooth`](@ref) and [`StarFormationHistories.calculate_edges`](@ref). To plot this with `PyPlot` you should do `plt.imshow(result.weights', origin="lower", ...)`.

Recommended usage is to make a histogram of your observational data using [`bin_cmd`](@ref), then pass the resulting histogram bins through using the `edges` keyword to [`bin_cmd_smooth`](@ref) and [`partial_cmd_smooth`](@ref) to construct smoothed isochrone models. 
"""
function bin_cmd_smooth(colors, mags, color_err, mag_err, cov_mult::Int=0;
                        weights = ones(promote_type(eltype(colors), eltype(mags)),
                                       size(colors)), edges=nothing,
                        xlim=extrema(colors), ylim=extrema(mags), nbins=nothing,
                        xwidth=nothing, ywidth=nothing)
    @argcheck cov_mult in (-1, 0, 1) # 1 for y=V and x=B-V, -1 for y=B and x=B-V, 0 for y=R and x=B-V
    @argcheck axes(colors) == axes(mags) == axes(color_err) == axes(mag_err) == axes(weights)
    # Calculate edges from provided kws
    edges = calculate_edges(edges, xlim, ylim, nbins, xwidth, ywidth)
    hist = Histogram(edges, zeros(Float64, length(edges[1]) - 1, length(edges[2]) - 1), :left, false)
    for i in eachindex(colors, mags, color_err, mag_err, weights)
        addstar!(hist, GaussianKernel(colors[i], mags[i], color_err[i], mag_err[i], cov_mult, weights[i]))
    end
    return hist
end

###############################
# Constructing binary templates

# Smooth Hess diagram of the binary systems of `model`, normalized to `normalize_value` solar masses formed.
# `m_ini` and `mags` are the densely resampled isochrone used for interpolation; the quadrature over binary
# systems from `binary_systems` uses every `ds`-th mass as its nodes.
function binary_hess(model::AbstractBinaryModel, m_ini::AbstractVector, mags::AbstractVector{<:AbstractVector{T}},
                     mag_err_funcs, y_index, color_indices, imf, completeness_funcs, bias_funcs,
                     edges::Tuple{<:AbstractRange, <:AbstractRange};
                     normalize_value::Number=1, mean_mass::Number=mean(imf), ds::Integer=4) where T <: Real
    Base.require_one_based_indexing(m_ini)
    Base.require_one_based_indexing(mags)
    @argcheck all(length(i) == length(m_ini) for i in mags)
    m1, m2, w = binary_systems(model, m_ini[begin:ds:end], imf; mean_mass)
    # System magnitudes from the summed flux of its stars; stars outside the isochrone contribute no light.
    # Fluxes are interpolated linearly on the dense m_ini grid, which differs negligibly from interpolating
    # magnitudes, and each star's position in m_ini is found once and reused for every filter.
    b1, b2 = bracket.(Ref(m_ini), m1), bracket.(Ref(m_ini), m2)
    sys_mags = map(mags) do mag
        F = mag2flux.(mag)
        flux2mag.(lerp_bracket.(Ref(F), b1) .+ lerp_bracket.(Ref(F), b2))
    end
    lit = findall(isfinite, first(sys_mags))
    sys_mags = [i[lit] for i in sys_mags]
    # Completeness and bias are evaluated at the intrinsic system magnitudes
    idxs = y_index in color_indices ? color_indices : (color_indices..., y_index)
    w = w[lit] .* normalize_value .* completeness_values(completeness_funcs, sys_mags, idxs)
    # bias is defined as (measured - intrinsic), so measured = intrinsic + bias
    obs = sys_mags .+ filter_values(bias_funcs, sys_mags)
    W, x, y, cell_mags = aggregate_cells(w, obs[first(color_indices)] .- obs[last(color_indices)], obs[y_index], sys_mags, edges)
    # Errors are evaluated at each cell's mean intrinsic magnitudes, which vary little within a cell
    σ = filter_values(mag_err_funcs, cell_mags)
    if y_index in color_indices
        # GaussianKernel takes the error of the color filter that is not on the y-axis
        x_c_idx = color_indices[findfirst(!=(y_index), color_indices)]
        σx, cov_mult = σ[x_c_idx], (y_index == first(color_indices)) ? -1 : 1
    else
        σx, cov_mult = sqrt.(σ[first(color_indices)] .^ 2 .+ σ[last(color_indices)] .^ 2), 0
    end
    return bin_cmd_smooth(x, y, σx, σ[y_index], cov_mult; weights=W, edges)
end
# Returns (i, t) such that m = m_ini[i] + t * (m_ini[i+1] - m_ini[i]) for sorted m_ini, or i = 0 if m is outside m_ini
function bracket(m_ini::AbstractVector, m)
    t0 = zero((m - first(m_ini)) / (last(m_ini) - first(m_ini)))
    (m < first(m_ini) || m > last(m_ini)) && return (0, t0)
    i = min(searchsortedlast(m_ini, m), lastindex(m_ini) - 1)
    return (i, oftype(t0, (m - m_ini[i]) / (m_ini[i+1] - m_ini[i])))
end
# Linear interpolation of F at a bracket from `bracket`, or zero outside the grid
lerp_bracket(F::AbstractVector, (i, t)) = iszero(i) ? zero(eltype(F)) : F[i] + t * (F[i+1] - F[i])
# One kernel per system is accurate but slow, so binary_hess aggregates systems in cells half the Hess diagram bin
# width on each axis and places one kernel per cell. Returns the summed weight of each cell, the weighted mean
# observed position (x, y), and the weighted mean intrinsic magnitude in each filter of `mags`. Cells are not
# limited to the Hess diagram so that systems outside it can still scatter into it. Also serves as a type barrier.
function aggregate_cells(w, x, y, mags, edges)
    Δx, Δy = step(edges[1]) / 2, step(edges[2]) / 2
    index = Dict{NTuple{2, Int}, Int}()
    W, X, Y, M = Float64[], Float64[], Float64[], [Float64[] for _ in mags]
    for k in eachindex(w, x, y)
        iszero(w[k]) && continue
        cell = (floor(Int, (x[k] - first(edges[1])) / Δx), floor(Int, (y[k] - first(edges[2])) / Δy))
        c = get!(index, cell) do
            foreach(v -> push!(v, 0), (W, X, Y, M...))
            length(W)
        end
        W[c] += w[k]; X[c] += w[k] * x[k]; Y[c] += w[k] * y[k]
        for f in eachindex(M, mags)
            M[f][c] += w[k] * mags[f][k]
        end
    end
    return W, X ./ W, Y ./ W, [m ./ W for m in M]
end

##########################################
# Constructing single-star model templates

"""
    result::StatsBase.Histogram =
        partial_cmd(m_ini::AbstractVector{<:Number},
                    colors::AbstractVector{<:Number},
                    mags::AbstractVector{<:Number},
                    imf;
                    dmod::Number=0,
                    normalize_value::Number=1,
                    mean_mass::Number=mean(imf),
                    edges=nothing,
                    xlim=extrema(colors),
                    ylim=extrema(mags),
                    nbins=nothing,
                    xwidth=nothing,
                    ywidth=nothing)

Creates an error-free Hess diagram for stars from an isochrone with x-axis photometric `colors`, y-axis photometric magnitudes `mags`, and initial masses `m_ini`. Because this is not smoothed by photometric errors, it is not generally useful but is provided for comparative checks. Most arguments are as in [`bin_cmd`](@ref). The only unique keyword arguments are `normalize_value::Number` which is a multiplicative factor giving the effective stellar mass you want in the Hess diagram, and `mean_mass::Number` which is the mean stellar mass implied by the provided initial mass function `imf`. 
"""
function partial_cmd(m_ini::AbstractVector{<:Number}, colors::AbstractVector{<:Number},
                     mags::AbstractVector{<:Number}, imf;
                     dmod::Number=0, normalize_value::Number=1,
                     mean_mass::Number=mean(imf), edges=nothing, xlim=extrema(colors),
                     ylim=extrema(mags), nbins=nothing, xwidth=nothing, ywidth=nothing)
    # Ingest once so that resampling and interpolation use the same initial masses
    m_ini, points = ingest_isochrone(m_ini, SVector.(colors, mags))
    colors, mags = first.(points), last.(points)
    # Resample the isochrone magnitudes to a denser m_ini array
    new_mini, new_spacing = mini_spacing(m_ini, colors, mags, 0.01, true)
    new_iso_colors = interpolate_mini(m_ini, colors, new_mini)
    new_iso_mags = interpolate_mini(m_ini, mags, new_mini) .+ dmod
    # Approximate the IMF weights on each star in the isochrone as
    # the trapezoidal rule integral across the bin.
    # This is ~equivalent to the difference in the CDF across the bin as long as imf is
    # a properly normalized pdf i.e., if imf is a
    # Distributions.ContinuousUnivariateDistribution,
    # weights[i] = cdf(imf, m_ini[2]) - cdf(imf, m_ini[1])
    weights = Vector{Float64}(undef, length(new_mini) - 1)
    @inbounds @simd for i in eachindex(weights)
        weights[i] = new_spacing[i] *
            (dispatch_imf(imf,new_mini[i]) + dispatch_imf(imf,new_mini[i+1])) / 2
        weights[i] *= normalize_value / mean_mass # Correct normalization
    end
    return bin_cmd(new_iso_colors[begin:end-1], new_iso_mags[begin:end-1];
                   weights=weights, edges=edges, xlim=xlim, ylim=ylim, nbins=nbins,
                   xwidth=xwidth, ywidth=ywidth)
end

    # Approximate the IMF weights on each star in the isochrone as
    # the trapezoidal rule integral across the bin. 
    # This is equivalent to the difference in the CDF across the bin as long as imf
    # is a properly normalized pdf i.e., if imf is a
    # Distributions.ContinuousUnivariateDistribution,
    # weights[i] = cdf(imf, m_ini[2]) - cdf(imf, m_ini[1])
function calculate_weights(mini::AbstractVector, completeness::AbstractVector,
                           imf, normalize_value::Number, mean_mass::Number, mini_spacing::AbstractVector=diff(mini))
    @argcheck length(mini) == length(completeness)
    @argcheck length(mini_spacing) == (length(mini) - 1)
    weights = Vector{Float64}(undef, length(mini) - 1)
    # Trapezoidal rule for the integral of the IMF times the completeness; using the completeness at only one end of
    # each interval is first-order accurate and biased low where the completeness falls toward lower masses
    for i in eachindex(weights)
        weights[i] = mini_spacing[i] *
            (dispatch_imf(imf, mini[i]) * completeness[i] + dispatch_imf(imf, mini[i+1]) * completeness[i+1]) / 2
        weights[i] *= normalize_value / mean_mass # Correct normalization
    end
    return weights
end
    # sum(weights) is now the full integral over the imf pdf from
    # minimum(m_ini) -> maximum(m_ini). This is equivalent to
    # cdf(imf, maximum(m_ini)) - cdf(imf, minimum(m_ini)).

"""
    result::StatsBase.Histogram =
        partial_cmd_smooth(m_ini::AbstractVector{<:Number},
                           mags::AbstractVector{<:AbstractVector{<:Number}},
                           mag_err_funcs,
                           y_index,
                           color_indices,
                           imf,
                           completeness_funcs=[one for i in mags],
                           bias_funcs=[zero for i in mags];
                           dmod::Number=0,
                           normalize_value::Number=1,
                           binary_model::AbstractBinaryModel=NoBinaries(),
                           mean_mass=mean(imf),
                           edges=nothing,
                           xlim=nothing,
                           ylim=nothing,
                           nbins=nothing,
                           xwidth=nothing,
                           ywidth=nothing)

Main function for generating template Hess diagrams from a simple stellar population of stars from an isochrone, including photometric error and completeness.

# Arguments
 - `m_ini::AbstractVector{<:Number}` is a vector containing the initial stellar masses of the stars from the isochrone. The isochrone points must be in evolutionary order (e.g., in order of equivalent evolutionary points), as given by isochrone libraries; see [`StarFormationHistories.ingest_isochrone`](@ref).
 - `mags::AbstractVector{<:AbstractVector{<:Number}}` is a vector of vectors. Each constituent vector with index `i` should have `length(mags[i]) == length(m_ini)`, representing the magnitudes of the isochrone stars in each of the magnitudes considered. In most cases, mags should contain 2 (if y-axis mag is also involved in the x-axis color) or 3 vectors.
 - `mag_err_funcs` computes the 1σ photometric errors for the filters provided in `mags`. It is either a `Vector` or `Tuple` of callables (e.g., a `Function`) with one callable per filter in `mags`, each taking a single *apparent* magnitude and returning a `Number`, or a single joint callable as described in the "Joint Photometric Models" section below.
 - `y_index` gives a valid index (e.g., an `Int` or `CartesianIndex`) into `mags` for the filter you want to have on the y-axis of the Hess diagram. For example, if the `mags` argument contains the B and V band magnitudes as `mags=[B, V]` and you want V on the y-axis, you would set `y_index` as `2`. 
 - `color_indices` is a length-2 indexable object giving the indices into `mags` that are to be used to compute the x-axis color. For example, if the `mags` argument contains the B and V band magnitudes as `mags=[B, V]`, and you want B-V to be the x-axis color, then `color_indices` should be `[1,2]` or `(1,2)` or similar.
 - `imf` is a callable that takes an initial stellar mass as its sole argument and returns the (properly normalized) probability density of your initial mass function model. All the models from [InitialMassFunctions.jl](https://github.com/cgarling/InitialMassFunctions.jl) are valid for `imf`.
 - `completeness_funcs` gives the completeness fractions as a function of *apparent* magnitude. It is either a `Vector` or `Tuple` of callables (e.g., a `Function`) returning the single-filter completeness for the matching filter provided in `mags`, in which case the completeness of a star is the product of the single-filter completeness values in the filters used in the Hess diagram, or a single joint callable as described in the "Joint Photometric Models" section below.
 - `bias_funcs` gives the expected photometric bias as a function of *apparent* magnitude. The photometric bias is defined as the difference between measured and intrinsic magnitude (i.e., `output - input` for artificial star tests). It is either a `Vector` or `Tuple` of callables (e.g., a `Function`) with one callable per filter in `mags`, or a single joint callable as described in the "Joint Photometric Models" section below.

# Keyword Arguments
 - `dmod::Number=0` is the distance modulus in magnitudes to apply to the input `mags`. Leave at `0` if you are providing apparent magnitudes in `mags`.
 - `normalize_value::Number=1` is the total stellar mass of the population you wish to model.
 - `binary_model::AbstractBinaryModel=NoBinaries()` is the model to use for including binary systems; see [`StarFormationHistories.NoBinaries`](@ref), [`StarFormationHistories.RandomBinaryPairs`](@ref), and [`StarFormationHistories.BinaryMassRatio`](@ref).
 - `mean_mass::Number` is the expectation value of the initial mass for a random star drawn from your provided `imf`. This will be computed for you if your provided `imf` is a valid continuous, univariate `Distributions.Distribution` object.
 - `edges` is a tuple of ranges defining the left-side edges of the bins along the x-axis (`edges[1]`) and the y-axis (`edges[2]`). Example: `(-1.0:0.1:1.5, 22:0.1:27.2)`. If `edges` is provided, it overrides the following keyword arguments that offer other ways to specify the extent of the Hess diagram.
 - `xlim` is a length-2 indexable object (e.g., a `Vector` or `Tuple`) giving the lower and upper bounds on the x-axis corresponding to the provided `colors` array. Example: `(-1.0, 1.5)`. This is only used if `edges` is not provided. 
 - `ylim` is as `xlim` but for the y-axis corresponding to the provided `mags` array. Example `(25.0, 20.0)`. This is only used if `edges` is not provided.
 - `nbins::NTuple{2, <:Integer}` is a 2-tuple of integers providing the number of bins to use along the x- and y-axes. This is only used if `edges` is not provided.
 - `xwidth` is the bin width along the x-axis for the `colors` array. This is only used if `edges` and `nbins` are not provided. Example: `0.1`.
 - `ywidth` is as `xwidth` but for the y-axis corresponding to the provided `mags` array. Example: `0.1`.

# Returns
This method returns the Hess diagram as a `StatsBase.Histogram`; you should refer to the StatsBase documentation for more information. In short, if the output of this method is `result`, then the Hess diagram represented as a `Matrix` is available as `result.weights` (this is what you would want for [`fit_templates`](@ref) and similar functions) and the edges of the histogram are available as `result.edges`.

# Joint Photometric Models
Per-filter models assume that the completeness, error, and bias in each filter depend only on the magnitude in that filter. This cannot represent, for example, a catalog cut requiring a minimum signal-to-noise ratio in every filter, which removes stars along a diagonal line in the Hess diagram. Each of `mag_err_funcs`, `completeness_funcs`, and `bias_funcs` may instead be a single joint callable that takes one *apparent* magnitude per filter in `mags`, in the same order as `mags` (e.g., `f(mag_1, mag_2)` for two filters). A joint `completeness_funcs` returns the probability that a star is detected and passes all selection criteria, and a joint `mag_err_funcs` or `bias_funcs` returns one value per filter (e.g., a `Tuple` or `StaticArrays.SVector`). Joint and per-filter models can be mixed; for example, a joint completeness model can be used with per-filter error and bias models. Photometric errors are still assumed to be uncorrelated between filters. See [`process_ASTs`](@ref) for measuring joint models from artificial star tests.
"""
function partial_cmd_smooth(m_ini::AbstractVector{<:Number},
                            mags::AbstractVector{<:AbstractVector{<:Number}},
                            mag_err_funcs, y_index, color_indices, imf,
                            completeness_funcs=[one for i in mags],
                            bias_funcs=[zero for i in mags];
                            dmod::Number=0, normalize_value::Number=1,
                            binary_model::AbstractBinaryModel=NoBinaries(),
                            mean_mass::Number=mean(imf),
                            edges=nothing, xlim=nothing, ylim=nothing, nbins=nothing,
                            xwidth=nothing, ywidth=nothing)
    @argcheck length(color_indices) == 2
    # Per-filter models must have one callable per filter; joint callables are not checked here
    @argcheck all(length(f) == length(mags) for f in (mag_err_funcs, completeness_funcs, bias_funcs) if f isa Union{Tuple, AbstractVector})
    # Calculate edges from provided kws
    edges = calculate_edges(edges, xlim, ylim, nbins, xwidth, ywidth)
    # Verify that the provided y_index and color_indices are valid
    @argcheck all(checkbounds(Bool, mags, idx) for idx in union(y_index, color_indices))
    # Ingest once with all filters so that resampling and interpolation use the same initial masses
    m_ini, point_mags = ingest_isochrone(m_ini, [getindex.(mags, i) for i in eachindex(m_ini)])
    mags = [getindex.(point_mags, k) for k in eachindex(mags)]
    # Resample the isochrone magnitudes to a denser m_ini array
    ymags = mags[y_index]
    colors = mags[first(color_indices)] .- mags[last(color_indices)]
    # Use bin spacing to inform necessary spacing of isochrone points
    Δmag = min(step(edges[1]), step(edges[2]))
    new_mini, new_spacing = mini_spacing(m_ini, colors, ymags, Δmag, true)
    # Interpolate only the mag vectors included in color_indices
    new_iso_mags = [interpolate_mini(m_ini, i, new_mini) .+ dmod for i in mags]
    mag_err = filter_values(mag_err_funcs, new_iso_mags)
    if y_index in color_indices # x-axis color is dependent on y-axis magnitude
        # This case will use GaussianKernel, which expects the color_err
        # to be the single-band error for the filter in the x-axis color which
        # does not appear on the y axis; i.e. if y=V and x=B-V, color_err should
        # simply be σB.
        x_c_idx = color_indices[findfirst(!=(y_index), color_indices)]
        color_err = mag_err[x_c_idx]
        # Calculate vector of completeness values; per-filter models use the product of the two involved magnitudes
        completeness_vec = completeness_values(completeness_funcs, new_iso_mags, color_indices)
        # Calculate weights; output length is one less than input vectors
        weights = calculate_weights(new_mini, completeness_vec, imf, normalize_value, mean_mass, new_spacing)
        cov_mult = (y_index == first(color_indices)) ? -1 : 1
    else # x-axis color is independent of y-axis magnitude
        # x-axis color error is quadrature of the two constituent magnitudes
        color_err = [sqrt(mag_err[first(color_indices)][i]^2 + mag_err[last(color_indices)][i]^2)
                     for i in eachindex(new_mini)]
        # Calculate vector of completeness values; per-filter models use the product of all three involved magnitudes
        completeness_vec = completeness_values(completeness_funcs, new_iso_mags, (color_indices..., y_index))
        # Calculate weights; output length is one less than input vectors
        weights = calculate_weights(new_mini, completeness_vec, imf, normalize_value, mean_mass, new_spacing)
        # 1 for y=V and x=B-V, -1 for y=B and x=B-V, 0 for y=R and x=B-V
        cov_mult = 0
    end

    # bias is defined as (measured - intrinsic), so measured = intrinsic + bias
    bias_iso_mags = new_iso_mags .+ filter_values(bias_funcs, new_iso_mags)
    colors = bias_iso_mags[first(color_indices)] .- bias_iso_mags[last(color_indices)]
    single_star_hist = bin_cmd_smooth(midpoints(colors), midpoints(bias_iso_mags[y_index]),
                                      midpoints(color_err), midpoints(mag_err[y_index]), cov_mult;
                                      weights=weights, edges=edges)

    bfrac = binary_system_fraction(binary_model)
    @argcheck bfrac <= 1
    iszero(bfrac) && return single_star_hist
    binary_hist = binary_hess(binary_model, new_mini, new_iso_mags, mag_err_funcs, y_index, color_indices, imf,
                              completeness_funcs, bias_funcs, edges; normalize_value, mean_mass)
    # Single-star systems hold a fraction 1 - binary_mass_fraction of the stellar mass formed;
    # binary_hist is already normalized and includes binaries with only one star in the isochrone.
    result_mat = single_star_hist.weights .* (1 - binary_mass_fraction(binary_model, imf)) .+ binary_hist.weights
    return Histogram(edges, result_mat, :left, false)
end

# Method exports
# Exports from StarFormationHistories.jl
export mean, bin_cmd, bin_cmd_smooth, partial_cmd, partial_cmd_smooth
# Exports from simulate.jl
export generate_stars_mass, generate_stars_mag, generate_stars_mass_composite,
    generate_stars_mag_composite, model_cmd
# Exports from fitting.jl
export fit_templates,
    hmc_sample,
    GaussianDispersion,
    LinearAMR,
    LogarithmicAMR,
    PowerLawMZR,
    fit_sfh,
    sample_sfh,
    calculate_coeffs


end # module
