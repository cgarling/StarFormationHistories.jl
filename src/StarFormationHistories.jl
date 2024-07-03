module StarFormationHistories

using Distributions: Distribution, Sampleable, Univariate, Continuous, pdf, logpdf,
    quantile, Multivariate, MvNormal, sampler, Uniform # cdf
import Distributions: _rand! # Extending
import DynamicHMC  # For random uncertainties in SFH fits
using Interpolations: interpolate, Gridded, Linear, deduplicate_knots! # extrapolate, Throw
import LBFGSB # Used for one method in fitting.jl
import LineSearches # For configuration of Optim.jl
# Need mul! for composite!, ∇loglikelihood!;
using LinearAlgebra: diag, Hermitian, mul!
import LogDensityProblems # For interfacing with DynamicHMC
using LoopVectorization: @turbo
import LoopVectorization: can_turbo # Extending for our functions
import Optim
using Printf: @sprintf
using QuadGK: quadgk # For general mean(imf::UnivariateDistribution{Continuous}; kws...)
using Random: AbstractRNG, default_rng, rand
using Roots: find_zero # For mass_limits in simulate.jl
using SpecialFunctions: erf
# LoopVectorization has a specialfunctions extension that provides erf(x::AbstractSIMD)
using VectorizationBase: AbstractSIMD, verf # SIMD-capable erf for Float32 and Float64
using StaticArrays: SVector, SMatrix, sacollect
using StatsBase: fit, Histogram, Weights, sample, median
import StatsBase: mean # Extending
import KissMCMC
import MCMCChains

# Code inclusion
include("utilities.jl")
include("simulate.jl")
include("fitting/fitting.jl") # This will include other relevant files

##################################
# Isochrone utilities

"""
    new_mags::Vector = interpolate_mini(m_ini, mags::Vector{<:Real}, new_mini)
    new_mags::Vector{Vector} = interpolate_mini(m_ini, mags::Vector{Vector{<:Real}},
                                                new_mini)
    new_mags::Matrix = interpolate_mini(m_ini, mags::AbstractMatrix, new_mini)

Function to interpolate `mags` as a function of initial mass vector `m_ini` onto a new initial mass vector `new_mini`. `mags` can either be a vector of equal length to `m_ini`, designating magnitudes in a single filter, or a vector of vectors, designating multiple filters, or a matrix with each column designating a different filter.

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
    m_ini, mags = sort_ingested(m_ini, mags) # from simulate.jl include'd above
    return interpolate((m_ini,), mags, Gridded(Linear()))(new_mini)
end
# I don't think these two methods are actually used by anything so not going to optimize,
# although we could use these in partial_cmd_smooth if we wanted. 
interpolate_mini(m_ini::AbstractVector{<:Number},
                 mags::AbstractVector{<:AbstractVector{<:Number}},
                 new_mini::AbstractVector{<:Number}) = 
                     [interpolate((m_ini,), i, Gridded(Linear()))(new_mini) for i in mags]
interpolate_mini(m_ini::AbstractVector{<:Number},
                 mags::AbstractMatrix{<:Number},
                 new_mini::AbstractVector{<:Number}) =
    reduce(hcat, interpolate((m_ini,), i,
                             Gridded(Linear()))(new_mini) for i in eachcol(mags))

"""
    mini_spacing(m_ini::AbstractVector{<:Number},
                 colors::AbstractVector{<:Number},
                 mags::AbstractVector{<:Number},
                 Δmag,
                 ret_spacing::Bool = false)

Returns a new sampling of stellar masses given the initial mass vector `m_ini` from an isochrone and the corresponding y-axis magnitude vector `mags` and x-axis color vector `color` to be used to construct a model Hess diagram. Will compute the new initial mass vector such that the distance between adjacent isochrone points is less than `Δmag`. Will return the change in mass between points `diff(new_mini)` if `ret_spacing==true`.

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
    @assert axes(m_ini) == axes(mags) == axes(colors)
    new_mini = Vector{Float64}(undef, 1)
    new_mini[1] = first(m_ini)
    # Sort the input m_ini and mags. This could be optional.
    idx = sortperm(m_ini)
    m_ini = m_ini[idx]
    mags = mags[idx]
    colors = colors[idx]
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

"""
    mean(imf::Distributions.ContinuousUnivariateDistribution; kws...) =
        quadgk(x->x*pdf(imf,x), extrema(imf)...; kws...)[1]

Simple one-liner that calculates the mean of the provided `imf` distribution using numerical integration via `QuadGK.quadgk` with the passed keyword arguments `kws...`. This is a generic fallback; better to define this explicitly for your IMF model. Requires that `pdf(imf,x)` and `extrema(imf)` be defined.
"""
mean(imf::Distribution{Univariate, Continuous}; kws...) =
     quadgk(x -> x * pdf(imf, x), extrema(imf)...; kws...)[1]

##################################
# KDE models

"Root abstract type for kernels used in Hess diagram modelling."
abstract type AbstractKernel end
"Abstract type for kernels used in Hess diagram modelling that define their parameters \
(e.g., widths, centers, sizes) in the pixel space of underlying Hess diagram matrix."
abstract type PixelSpaceKernel <: AbstractKernel end
"Abstract type for kernels used in Hess diagram modelling that define their parameters \
(e.g., widths, centers, sizes) in real magnitude space -- these kernels require their \
parameters converted into pixel-space prior to injection into the Hess diagram matrix."
abstract type RealSpaceKernel <: AbstractKernel end

##################################
# Code for concrete implementations of PixelSpaceKernel

"""
    gaussian_int_general(Δx::Real, Δy::Real, halfxstep::Real, halfystep::Real,
                         σx::Real, σy::Real, A::Real, B::Real)

Returns the exact analytic integral from `(x - halfxstep, x + halfxstep)` and `(y - halfystep, y + halfystep)` for the asymmetric, *possibly* covariant 2D Gaussian. `A` is a normalization constant which is equal to overall integral of the function, not accounting for an additive background `B`. 

# Notes

Let `Δx` and `Δy` be offsets from the centroid `(x0, y0)` of a 2-D Gaussian such that the midpoint of integration is `(x, y) = (x0, y0) .+ (Δx, Δy)` in the case of no covariance. There are two additional simple covariance patterns;
 - if "B" and "V" are independently sampled from normal distributions, and `y="B"` and `x="B"-"V"`
 - ... and `y="V"` and `x="B"-"V"`
This covariance can be modelled by substituting
`Δx = x - x0 + (Δy * cov_mult)`
with specific values of `cov_mult`:
 - `1` for kernels to be injected into Hess diagrams with filter like `y="V"` and `x="B"-"V"`,
 - `-1` for kernels to be injected into Hess diagrams with filter like `y="V"` and `x="B"-"V"`,
 - `0` for `y="R"` and `x="B-V"`.
As only the offsets are required for the integration, this substitution be done prior to calling this method. This is handled by [`StarFormationHistories.gaussian_psf_covariant](@ref).
"""
@inline function gaussian_int_general(Δx::T, Δy::T, halfxstep::T, halfystep::T,
                                      σx::T, σy::T, A::T, B::T) where T <: Real
    sqrt2 = sqrt(T(2))
    return A / 4 * erf((Δx + halfxstep) / (sqrt2 * σx), (Δx - halfxstep) / (sqrt2 * σx)) *
                   erf((Δy + halfystep) / (sqrt2 * σy), (Δy - halfystep) / (sqrt2 * σy)) + B
end
# 2x speedup in double precision for Julia 1.10 using VectorizationBase.verf
@inline function gaussian_int_general(Δx::T, Δy::T, halfxstep::T, halfystep::T,
                                      σx::T, σy::T, A::T, B::T) where T <: Union{Float32, Float64,
                                                                                 AbstractSIMD{<:Any, Float32},
                                                                                 AbstractSIMD{<:Any, Float64}}
    sqrt2 = sqrt(T(2))
    return A / 4 * (verf((Δx - halfxstep) / (sqrt2 * σx)) - verf((Δx + halfxstep) / (sqrt2 * σx))) *
        (verf((Δy - halfystep) / (sqrt2 * σy)) - verf((Δy + halfystep) / (sqrt2 * σy))) + B
end
@inline gaussian_int_general(args::Vararg{Real,8}) = gaussian_int_general(promote(args...)...)

"""
    GaussianPSFAsymmetric(x0::Real, y0::Real, σx::Real, σy::Real)
    GaussianPSFAsymmetric(x0::Real, y0::Real, σx::Real, σy::Real, A::Real)
    GaussianPSFAsymmetric(x0::Real, y0::Real, σx::Real, σy::Real, A::Real, B::Real)

A [`PixelSpaceKernel`](@ref StarFormationHistories.PixelSpaceKernel) representing the 2D asymmetric Gaussian PSF without rotation (no θ).

# Parameters
 - `x0`, the center of the PSF model along the first matrix dimension
 - `y0`, the center of the PSF model along the second matrix dimension
 - `σx`, the Gaussian `σ` along the first matrix dimension
 - `σy`, the Gaussian `σ` along the first matrix dimension
 - `A`, and additional multiplicative constant in front of the normalized Gaussian
 - `B`, a constant additive background across the PSF
"""
struct GaussianPSFAsymmetric{T <: Real} <: PixelSpaceKernel
    x0::T
    y0::T
    σx::T
    σy::T
    A::T
    B::T
    function GaussianPSFAsymmetric(x0::Real, y0::Real, σx::Real, σy::Real)
        T = promote(x0, y0, σx, σy)
        T_type = eltype(T)
        new{T_type}(T[1], T[2], T[3], T[4], one(T_type), zero(T_type))
    end
    function GaussianPSFAsymmetric(x0::Real, y0::Real, σx::Real, σy::Real, A::Real)
        T = promote(x0, y0, σx, σy, A)
        T_type = eltype(T)
        new{T_type}(T[1], T[2], T[3], T[4], T[5], zero(T_type))
    end
    function GaussianPSFAsymmetric(x0::Real, y0::Real, σx::Real, σy::Real, A::Real, B::Real)
        T = promote(x0, y0, σx, σy, A, B)
        new{eltype(T)}(T...)
    end
end
Base.Broadcast.broadcastable(m::GaussianPSFAsymmetric) = Ref(m)
parameters(model::GaussianPSFAsymmetric) = (model.x0, model.y0, model.σx, model.σy, model.A, model.B)
Base.size(model::GaussianPSFAsymmetric) = (ceil(Int, model.σx * 10), ceil(Int, model.σy * 10))
centroid(model::GaussianPSFAsymmetric) = (model.x0, model.y0)
""" 
    gaussian_psf_asymmetric_integral_halfpix(x::Real, y::Real, x0::Real, y0::Real, σx::Real,
                                             σy::Real, A::Real, B::Real)

Exact analytic integral from `(x - 0.5, x + 0.5)` and `(y - 0.5, y + 0.5)` for the asymmetric, non-rotated 2D Gaussian. `A` is a normalization constant which is equal to overall integral of the function, not accounting for an additive background `B`. 
"""
@inline function gaussian_psf_asymmetric_integral_halfpix(x::T, y::T, x0::T, y0::T,
                                                          σx::T, σy::T, A::T, B::T) where T <: Real
    T <: Integer ? onehalf = 0.5 : onehalf = T(1//2) # Cover all integer case
    Δx = x - x0
    Δy = y - y0
    return gaussian_int_general(Δx, Δy, onehalf, onehalf, σx, σy, A, B)
end
@inline gaussian_psf_asymmetric_integral_halfpix(args::Vararg{Real,8}) = 
    gaussian_psf_asymmetric_integral_halfpix(promote(args...)...)
@inline evaluate(model::GaussianPSFAsymmetric, x::Real, y::Real) = 
    gaussian_psf_asymmetric_integral_halfpix(x, y, parameters(model)...)
# Need to let LoopVectorization know that it can turbo this function
can_turbo(::typeof(evaluate), ::Val{3}) = true

##################################
# Code for concrete implementations of RealSpaceKernel

struct GaussianPSFCovariant{T <: Real} <: RealSpaceKernel
    x0::T
    y0::T
    σx::T
    σy::T
    cov_mult::T # 1 for y=V and x=B-V, -1 for y=B and x=B-V, 0 for y=R and x=B-V
    A::T
    B::T
    GaussianPSFCovariant(x0::Real, y0::Real, σx::Real, σy::Real, cov_mult::Real) = 
        GaussianPSFCovariant(x0, y0, σx, σy, cov_mult, 1, 0)
    GaussianPSFCovariant(x0::Real, y0::Real, σx::Real, σy::Real, cov_mult::Real, A::Real) =
        GaussianPSFCovariant(x0, y0, σx, σy, cov_mult, A, 0)
    function GaussianPSFCovariant(x0::Real, y0::Real, σx::Real, σy::Real, cov_mult::Real, A::Real, B::Real)
        @assert (cov_mult == 1) || (cov_mult == -1) || (cov_mult == 0)
        T = promote(x0, y0, σx, σy, cov_mult, A, B)
        new{eltype(T)}(T...)
    end
end
Base.Broadcast.broadcastable(m::GaussianPSFCovariant) = Ref(m)
parameters(model::GaussianPSFCovariant) = (model.x0, model.y0, model.σx, model.σy,
                                           model.cov_mult, model.A, model.B)
Base.size(model::GaussianPSFCovariant) = (10 * model.σx, 10 * model.σy)
centroid(model::GaussianPSFCovariant) = (model.x0, model.y0)
# This is the PSF but we really want the integral PRF
# @inline function gaussian_psf_covariant(x::Real,y::Real,x0::Real,y0::Real,σx::Real,
#                                         σy::Real,cov_mult::Real,A::Real,B::Real)
#     δy = y - y0
#     δx = x - x0 + (δy * cov_mult)
#     return (A / σy / σx / 2 / π) * exp( -(δy/σy)^2 / 2 ) * exp( -(δx/σx)^2 / 2 ) + B
# end
"""
    gaussian_psf_covariant(x::Real, y::Real, halfxstep::Real, halfystep::Real, x0::Real, 
                           y0::Real, σx::Real, σy::Real, cov_mult::Real, A::Real, B::Real)

Exact analytic integral from `(x - halfxstep, x + halfxstep)` and `(y - halfystep, y + halfystep)` for the asymmetric, covariant 2D Gaussian. `cov_mult` controls the degree of covariance and should be
 - `1` for kernels to be injected into Hess diagrams with filter like `y="V"` and `x="B-V"`,
 - `-1` for kernels to be injected into Hess diagrams with filter like `y="V"` and `x="B-V"`,
 - `0` for `y="R"` and `x="B-V"`.
`A` is a normalization constant which is equal to overall integral of the function, not accounting for an additive background `B`. 
"""
@inline function gaussian_psf_covariant(x::T, y::T, halfxstep::T, halfystep::T, x0::T, y0::T, σx::T,
                                        σy::T, cov_mult::T, A::T, B::T) where T <: Real
    Δy = y - y0
    Δx = x - x0 + (Δy * cov_mult)
    return gaussian_int_general(Δx, Δy, halfxstep, halfystep, σx, σy, A, B)

end
@inline gaussian_psf_covariant(args::Vararg{Real,11}) = 
    gaussian_psf_covariant(promote(args...)...)
@inline evaluate(model::GaussianPSFCovariant, x::Real, y::Real, halfxstep::Real, halfystep::Real) = 
    gaussian_psf_covariant(x, y, halfxstep, halfystep, parameters(model)...)
# Need to let LoopVectorization know that it can turbo this function
can_turbo(::typeof(evaluate), ::Val{5}) = true

#####################################################
# Function to add a kernel to a smoothed Hess diagram

@inline addstar!(image::Histogram, obj::PixelSpaceKernel) = addstar!(image.weights, obj)
@inline addstar!(image::Histogram, obj::PixelSpaceKernel, cutout_size::Tuple{Int,Int}) =
    addstar!(image.weights, obj, cutout_size)
function addstar!(image::AbstractMatrix, obj::PixelSpaceKernel, cutout_size::Tuple{Int,Int}=size(obj))
    @assert length(image) > 1
    x,y = round.(Int, centroid(obj)) # get the center of the object to be inserted, in pixel space
    x_offset = cutout_size[1] ÷ 2
    y_offset = cutout_size[2] ÷ 2
    # Need to eval at pixel midpoint, so add 0.5
    onehalf = eltype(image)(1//2)
    # Define loop indices, verifying safe bounds for @turbo loop
    xind = max(firstindex(image, 1), x - x_offset):min(lastindex(image, 1), x + x_offset)
    yind = max(firstindex(image, 2), y - y_offset):min(lastindex(image, 2), y + y_offset)
    # Double loop over x and y
    if (length(xind) > 1) && (length(yind) > 1) # Don't run if loop indices empty; safety for @turbo
        @turbo for i=xind, j=yind
            @inbounds image[i, j] += evaluate(obj, i + onehalf, j + onehalf) # Add 0.5 to evaluate at pixel midpoint
        end
    end
end

function addstar!(image::Histogram, obj::RealSpaceKernel, cutout_size::NTuple{2,T}=size(obj)) where T <: Number
    data = image.weights
    @assert length(data) > 1
    Base.require_one_based_indexing(data) # Make sure data is 1-indexed
    edges = image.edges # Get histogram edges
    # Need uniform step; easiest to enforce via ranges
    if !all(Base.Fix2(isa, AbstractRange), edges)
        throw(ArgumentError("The `image::Histogram` provided to `addstar!` with a `obj::RealSpaceKernel` object must have bin edges `image.edges` that are subtypes of `AbstractRange`."))
    end
    xrstep = step(edges[1])
    yrstep = step(edges[2])
    xr,yr = centroid(obj) # Get the center of the object to be inserted, in real space
    xp,yp = histogram_pix(xr, edges[1]), histogram_pix(yr, edges[2]) # Convert to fractional pixel space
    xp,yp = round(Int, xp, RoundUp), round(Int, yp, RoundUp) # Round to nearest integer index
    # Convert cutout_size into pixel-space and take half width
    x_offset = round(Int, cutout_size[1] / xrstep / 2, RoundUp)
    y_offset = round(Int, cutout_size[2] / yrstep / 2, RoundUp)
    # Construct iterators over pixel indexes, verifying safe bounds for @turbo loop
    xpiter = max(firstindex(data, 1), xp - x_offset):min(lastindex(data, 1), xp + x_offset)
    ypiter = max(firstindex(data, 2), yp - y_offset):min(lastindex(data, 2), yp + y_offset)
    # Using a range defined by a step length (e.g., xrstep) can result in the range having 1 too
    # few elements as the range is defined according to the following rule:
    # If length is not specified and stop - start is not an integer multiple of step,
    # a range that ends before stop will be produced.
    xriter = range(histogram_data(first(xpiter) + 1//2, edges[1]),
                   histogram_data(last(xpiter) + 1//2, edges[1]); length=length(xpiter))
    yriter = range(histogram_data(first(ypiter) + 1//2, edges[2]),
                   histogram_data(last(ypiter) + 1//2, edges[2]); length=length(ypiter))
    # Half the step widths are needed for evaluate call
    halfxstep, halfystep = xrstep / 2, yrstep / 2
    # Double loop over x and y
    # Above takes ~200ns, so this loop dominates runtime if size(obj) > (2,2) or so
    # for (xind, xreal) = zip(xpiter, xriter), (yind, yreal) = zip(ypiter, yriter)
    if (length(xpiter) > 1) && (length(ypiter) > 1) # Don't run if loop indices empty; safety for @turbo
        @turbo for i=eachindex(xpiter, xriter), j=eachindex(ypiter, yriter)
            @inbounds xind, xreal = xpiter[i], xriter[i]
            @inbounds yind, yreal = ypiter[j], yriter[j]
            @inbounds data[xind, yind] += evaluate(obj, xreal, yreal, halfxstep, halfystep)
        end
    end
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
    histogram_pix(d, edges)
    histogram_pix(d, edges::AbstractRange)

Returns the fractional index (i.e., pixel position) of value `d` given the left-aligned bin `edges`. The specialized form for `edges::AbstractRange` accepts reverse-sorted input (e.g., `edges=1:-0.1:0.0`) but `edges` must be sorted if you are providing an `edges` that is not an `AbstractRange`. See also [`StarFormationHistories.histogram_data`](@ref) for inverse function.

# Examples
```jldoctest
julia> StarFormationHistories.histogram_pix(0.5,0.0:0.1:1.0) ≈ 6
true

julia> (0.0:0.1:1.0)[6] == 0.5
true

julia> StarFormationHistories.histogram_pix(0.55,0.0:0.1:1.0) ≈ 6.5
true

julia> StarFormationHistories.histogram_pix(0.5,1.0:-0.1:0.0) ≈ 6
true

julia> StarFormationHistories.histogram_pix(0.5,collect(0.0:0.1:1.0)) ≈ 6
true
```
"""
histogram_pix(d, edges::AbstractRange) = (d - first(edges)) / step(edges) + 1
function histogram_pix(d, edges)
    idd = searchsortedfirst(edges, d)
    if edges[idd] == d
        return idd
    else
        Δe = edges[idd] - edges[idd-1]
        return (idd-1) + (d - edges[idd-1]) / Δe
    end
end

"""
    histogram_data(x, edges::AbstractRange)

Given a pixel-space value `x`, return the data-space value `d` it corresponds to according to the provided `edges::AbstractRange`. Inverse of [`StarFormationHistories.histogram_pix`](@ref). 
"""
histogram_data(x, edges::AbstractRange) = (x - 1) * step(edges) + first(edges)

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
    @assert length(colors) == length(mags) == length(weights)
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
    @assert cov_mult in (-1, 0, 1) # 1 for y=V and x=B-V, -1 for y=B and x=B-V, 0 for y=R and x=B-V
    @assert axes(colors) == axes(mags) == axes(color_err) == axes(mag_err) == axes(weights)
    # Calculate edges from provided kws
    edges = calculate_edges(edges, xlim, ylim, nbins, xwidth, ywidth)
    # Construct matrix to hold the 2D histogram
    mat = zeros(Float64, length(edges[1])-1, length(edges[2])-1)
    hist = Histogram(edges, mat, :left, false)
    if cov_mult == 0 # Case: y=R and x=B-V
        for i in eachindex(colors)
            # Skip stars that are 3σ away from the histogram region in either x or y. 
            # if (((colors[i] - 3*color_err[i]) > maximum(edges[1]))  |
            #     ((colors[i] + 3*color_err[i]) < minimum(edges[1])))
            #     |
            #     ( ((mags[i] - 3*mag_err[i]) > maximum(edges[2])) | ((mags[i] + 3*mag_err[i]) <
            #                                                         minimum(edges[2])) )
            #     continue
            # end
            # Get the pixel width in each dimension;
            # this currently only works if edges[1] and [2] are AbstractRange. 
            xwidth, ywidth = step(edges[1]), step(edges[2])
            # Convert colors, mags, color_err, and mag_err from magnitude-space to
            # pixel-space in `mat`
            x0 = histogram_pix(colors[i], edges[1])
            y0 = histogram_pix(mags[i], edges[2])
            σx = color_err[i] / xwidth
            σy = mag_err[i] / ywidth
            # Construct the star object
            obj = GaussianPSFAsymmetric(x0, y0, σx, σy, weights[i], 0.0)
            # Insert the star object
            cutout_size = size(obj) # ( round(Int,3σx,RoundUp), round(Int,3σy,RoundUp) )
            addstar!(hist, obj, cutout_size)
        end
    else # Case: (y=V and x=B-V) or (y=B and x=B-V)
        for i in eachindex(colors)
            # Construct the star object
            obj = GaussianPSFCovariant(colors[i], mags[i], color_err[i], mag_err[i], cov_mult, weights[i], 0.0)
            # Insert the star object
            cutout_size = size(obj) # ( round(Int,3σx,RoundUp), round(Int,3σy,RoundUp) )
            addstar!(hist, obj, cutout_size)
        end
    end
    return Histogram(edges, mat, :left, false)
end

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
    @assert length(mini) == length(completeness)
    @assert length(mini_spacing) == (length(mini) - 1)
    weights = Vector{Float64}(undef, length(mini) - 1)
    for i in eachindex(weights)
        weights[i] = mini_spacing[i] *
            (dispatch_imf(imf, mini[i]) + dispatch_imf(imf, mini[i+1])) / 2
        weights[i] *= completeness[i] * normalize_value / mean_mass # Correct normalization
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
                           completeness_funcs=[one for i in mags];
                           dmod::Number=0,
                           normalize_value::Number=1,
                           mean_mass=mean(imf),
                           edges=nothing,
                           xlim=nothing,
                           ylim=nothing,
                           nbins=nothing,
                           xwidth=nothing,
                           ywidth=nothing)

Main function for generating template Hess diagrams from a simple stellar population of stars from an isochrone, including photometric error and completeness.

# Arguments
 - `m_ini::AbstractVector{<:Number}` is a vector containing the initial stellar masses of the stars from the isochrone.
 - `mags::AbstractVector{<:AbstractVector{<:Number}}` is a vector of vectors. Each constituent vector with index `i` should have `length(mags[i]) == length(m_ini)`, representing the magnitudes of the isochrone stars in each of the magnitudes considered. In most cases, mags should contain 2 (if y-axis mag is also involved in the x-axis color) or 3 vectors.
 - `mag_err_funcs` must be an indexable object (e.g., a vector or tuple) that contains callables (e.g., a Function) to compute the 1σ photometric errors in the filters provided in `mags`. Each callable must take a single argument and return a `Number`. The length `mag_err_funcs` must be equal to the length of `mags`.
 - `y_index` gives a valid index (e.g., an `Int` or `CartesianIndex`) into `mags` for the filter you want to have on the y-axis of the Hess diagram. For example, if the `mags` argument contains the B and V band magnitudes as `mags=[B, V]` and you want V on the y-axis, you would set `y_index` as `2`. 
 - `color_indices` is a length-2 indexable object giving the indices into `mags` that are to be used to compute the x-axis color. For example, if the `mags` argument contains the B and V band magnitudes as `mags=[B, V]`, and you want B-V to be the x-axis color, then `color_indices` should be `[1,2]` or `(1,2)` or similar.
 - `imf` is a callable that takes an initial stellar mass as its sole argument and returns the (properly normalized) probability density of your initial mass function model. All the models from [InitialMassFunctions.jl](https://github.com/cgarling/InitialMassFunctions.jl) are valid for `imf`.
 - `completeness_functions` must be an indexable object (e.g., a vector or tuple) that contains callables (e.g., a Function) to compute the single-filter completeness fractions as a function of magnitude. Each callable in this argument must correspond to the matching filter provided in `mags`.

# Keyword Arguments
 - `dmod::Number=0` distance modulus in magnitudes to apply to the input `mags`.
 - `normalize_value::Number=1` gives the total stellar mass of the population you wish to model.
 - `mean_mass::Number` gives the expectation value for a random star drawn from your provided `imf`. This will be computed for you if your provided `imf` is a valid continuous, univariate `Distributions.Distribution` object.
 - `edges` is a tuple of vector-like objects defining the left-side edges of the bins along the x-axis (`edges[1]`) and the y-axis (`edges[2]`). Example: `(-1.0:0.1:1.5, 22:0.1:27.2)`. If `edges` is provided, it overrides the following keyword arguments that offer other ways to specify the extent of the Hess diagram.
 - `xlim`; a length-2 indexable object (e.g., a vector or tuple) giving the lower and upper bounds on the x-axis corresponding to the provided `colors` array. Example: `[-1.0, 1.5]`. This is only used if `edges` is not provided. 
 - `ylim`; as `xlim` but for the y-axis corresponding to the provided `mags` array. Example `[25.0, 20.0]`. This is only used if `edges` is not provided.
 - `nbins::NTuple{2,<:Integer}` is a 2-tuple of integers providing the number of bins to use along the x- and y-axes. This is only used if `edges` is not provided.
 - `xwidth`; the bin width along the x-axis for the `colors` array. This is only used if `edges` and `nbins` are not provided. Example: `0.1`. 
 - `ywidth`; as `xwidth` but for the y-axis corresponding to the provided `mags` array. Example: `0.1`.

# Returns
This method returns the Hess diagram as a `StatsBase.Histogram`; you should refer to the StatsBase documentation for more information. In short, if the output of this method is `result`, then the Hess diagram represented as a `Matrix` is available as `result.weights` (this is what you would want for [`fit_templates`](@ref) and similar functions) and the edges of the histogram are available as `result.edges`.
"""
function partial_cmd_smooth(m_ini::AbstractVector{<:Number},
                            mags::AbstractVector{<:AbstractVector{<:Number}},
                            mag_err_funcs, y_index, color_indices, imf,
                            completeness_funcs=[one for i in mags]; dmod::Number=0,
                            normalize_value::Number=1, mean_mass::Number=mean(imf),
                            edges=nothing, xlim=nothing, ylim=nothing, nbins=nothing,
                            xwidth=nothing, ywidth=nothing)
    @assert length(color_indices) == 2
    @assert length(mags) == length(mag_err_funcs) == length(completeness_funcs)
    # Calculate edges from provided kws
    edges = calculate_edges(edges, xlim, ylim, nbins, xwidth, ywidth)
    # Verify that the provided y_index and color_indices are valid
    for idx in union(y_index, color_indices)
        @assert all(map(x -> y_index in first(axes(x)), (mags, mag_err_funcs, completeness_funcs)))
    end
    # Resample the isochrone magnitudes to a denser m_ini array
    ymags = mags[y_index]
    colors = mags[first(color_indices)] .- mags[last(color_indices)]
    # Use bin spacing to inform necessary spacing of isochrone points
    Δmag = min(step(edges[1]), step(edges[2]))
    new_mini, new_spacing = mini_spacing(m_ini, colors, ymags, Δmag, true)
    # Interpolate only the mag vectors included in color_indices
    new_iso_mags = [interpolate_mini(m_ini, i, new_mini) .+ dmod for i in mags]
    colors = new_iso_mags[first(color_indices)] .- new_iso_mags[last(color_indices)]
    mag_err = [mag_err_funcs[i].(new_iso_mags[i]) for i in eachindex(mags)]
    if y_index in color_indices # x-axis color is dependent on y-axis magnitude
        # This case will use GaussianPSFCovariant, which expects the color_err
        # to be the single-band error for the filter in the x-axis color which
        # does not appear on the y axis; i.e. if y=V and x=B-V, color_err should
        # simply be σB.
        x_c_idx = color_indices[findfirst(x -> x !== y_index, color_indices)]
        color_err = mag_err[x_c_idx]
        # Calculate vector of completeness products; product of the two involved magnitudes
        completeness_vec = completeness_funcs[first(color_indices)].(new_iso_mags[first(color_indices)]) .*
            completeness_funcs[last(color_indices)].(new_iso_mags[last(color_indices)])
        # Calculate weights; output length is one less than input vectors
        weights = calculate_weights(new_mini, completeness_vec, imf, normalize_value, mean_mass, new_spacing)
        cov_mult = (y_index == first(color_indices)) ? -1 : 1
    else # x-axis color is independent of y-axis magnitude
        # x-axis color error is quadrature of the two constituent magnitudes
        color_err = [sqrt(mag_err[first(color_indices)][i]^2 + mag_err[last(color_indices)][i]^2)
                     for i in eachindex(new_mini)]
        # Calculate vector of completeness products; product of all three involved magnitudes
        completeness_vec = completeness_funcs[first(color_indices)].(new_iso_mags[first(color_indices)]) .*
            completeness_funcs[last(color_indices)].(new_iso_mags[last(color_indices)]) .*
            completeness_funcs[y_index].(new_iso_mags[y_index])
            # Calculate weights; output length is one less than input vectors
        weights = calculate_weights(new_mini, completeness_vec, imf, normalize_value, mean_mass, new_spacing)
        # 1 for y=V and x=B-V, -1 for y=B and x=B-V, 0 for y=R and x=B-V
        cov_mult = 0
    end
    
    return bin_cmd_smooth(midpoints(colors), midpoints(new_iso_mags[y_index]),
                          midpoints(color_err), midpoints(mag_err[y_index]), cov_mult;
                          weights=weights, edges=edges)
end


# Method exports
# Exports from StarFormationHistories.jl
export mean, bin_cmd, bin_cmd_smooth, partial_cmd, partial_cmd_smooth
# Exports from simulate.jl
export generate_stars_mass, generate_stars_mag, generate_stars_mass_composite,
    generate_stars_mag_composite, model_cmd
# Exports from fitting.jl
export fit_templates, hmc_sample


end # module
