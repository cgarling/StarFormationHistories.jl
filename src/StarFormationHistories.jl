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
import Optim
using Printf: @sprintf
using QuadGK: quadgk # For general mean(imf::UnivariateDistribution{Continuous}; kws...)
using Random: AbstractRNG, default_rng, rand
using Roots: find_zero # For mass_limits in simulate.jl
using SpecialFunctions: erf
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
                 mags::AbstractVector{<:Number},
                 Δmag,
                 ret_spacing::Bool = false)

Returns a new sampling of stellar masses given the initial mass vector `m_ini` from an isochrone and the corresponding magnitude vector `mags`. Will compute the new initial mass vector such that the absolute difference between adjacent points is less than `Δmag`. Will return the change in mass between points `diff(new_mini)` if `ret_spacing==true`.

```julia
julia> mini_spacing([0.08, 0.10, 0.12, 0.14, 0.16],
                    [13.545, 12.899, 12.355, 11.459, 10.947], 0.1, false)
```
"""
function mini_spacing(m_ini::AbstractVector{<:Number}, mags::AbstractVector{<:Number}, Δmag,
                      ret_spacing::Bool=false)
    @assert axes(m_ini) == axes(mags)
    new_mini = Vector{Float64}(undef,1)
    new_mini[1] = first(m_ini)
    # Sort the input m_ini and mags. This could be optional. 
    idx = sortperm(m_ini)
    m_ini = m_ini[idx]
    mags = mags[idx]
    # Loop through the indices, testing if adjacent magnitudes are
    # different by less than Δm.
    for i in eachindex(m_ini, mags)[begin:end-1]
        diffi = abs(mags[i+1] - mags[i])
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
     quadgk(x->x*pdf(imf,x), extrema(imf)...; kws...)[1]

##################################
# KDE models

"""
    GaussianPSFAsymmetric(x0::Real, y0::Real, σx::Real, σy::Real)
    GaussianPSFAsymmetric(x0::Real, y0::Real, σx::Real, σy::Real, A::Real)
    GaussianPSFAsymmetric(x0::Real, y0::Real, σx::Real, σy::Real, A::Real, B::Real)

Type representing the 2D asymmetric Gaussian PSF without rotation (no θ).

# Parameters
 - `x0`, the center of the PSF model along the first matrix dimension
 - `y0`, the center of the PSF model along the second matrix dimension
 - `σx`, the Gaussian `σ` along the first matrix dimension
 - `σy`, the Gaussian `σ` along the first matrix dimension
 - `A`, and additional multiplicative constant in front of the normalized Gaussian
 - `B`, a constant additive background across the PSF
"""
struct GaussianPSFAsymmetric{T <: Real} 
    x0::T
    y0::T
    σx::T
    σy::T
    A::T
    B::T
    function GaussianPSFAsymmetric(x0::Real,y0::Real,σx::Real,σy::Real)
        T = promote(x0,y0,σx,σy)
        T_type = eltype(T)
        new{T_type}(T[1],T[2],T[3],T[4],one(T_type),zero(T_type))
    end
    function GaussianPSFAsymmetric(x0::Real,y0::Real,σx::Real,σy::Real,A::Real)
        T = promote(x0,y0,σx,σy,A)
        T_type = eltype(T)
        new{T_type}(T[1],T[2],T[3],T[4],T[5],zero(T_type))
    end
    function GaussianPSFAsymmetric(x0::Real,y0::Real,σx::Real,σy::Real,A::Real,B::Real)
        T = promote(x0,y0,σx,σy,A,B)
        new{eltype(T)}(T...)
    end
end
Base.Broadcast.broadcastable(m::GaussianPSFAsymmetric) = Ref(m)
parameters(model::GaussianPSFAsymmetric) = (model.x0, model.y0, model.σx, model.σy,
                                            model.A, model.B)
function Base.size(model::GaussianPSFAsymmetric)
    σx, σy = model.σx, model.σy
    return (ceil(Int,σx * 10), ceil(Int,σy * 10)) 
end
centroid(model::GaussianPSFAsymmetric) = (model.x0, model.y0)  
""" 
    gaussian_psf_asymmetric_integral_halfpix(x::Real,
                                             y::Real,
                                             x0::Real,
                                             y0::Real,
                                             σx::Real,
                                             σy::Real,
                                             A::Real,
                                             B::Real)

Exact analytic integral for the asymmetric, non-rotated 2D Gaussian. `A` is a normalization constant which is equal to overall integral of the function, not accounting for an additive background `B`. 
"""
gaussian_psf_asymmetric_integral_halfpix(x::Real,y::Real,x0::Real,y0::Real,
                                         σx::Real,σy::Real,A::Real,B::Real) = 
    0.25 * A * erf((x+0.5-x0) / (sqrt(2) * σx), (x-0.5-x0) / (sqrt(2) * σx)) *
        erf((y+0.5-y0) / (sqrt(2) * σy), (y-0.5-y0) / (sqrt(2) * σy)) + B
evaluate(model::GaussianPSFAsymmetric, x::Real, y::Real) = 
    gaussian_psf_asymmetric_integral_halfpix(x, y, parameters(model)...)

##################################
# Function to add a star to a smoothed CMD

function addstar!(image::AbstractMatrix, obj, cutout_size::Tuple{Int,Int}=size(obj))
    x,y = round.(Int,centroid(obj)) # get the center of the object to be inserted
    x_offset = cutout_size[1] ÷ 2
    y_offset = cutout_size[2] ÷ 2
    for i in x-x_offset:x+x_offset
        if checkbounds(Bool,image,i)           # check bounds on image
            for j in y-y_offset:y+y_offset
                if checkbounds(Bool,image,i,j) # check bounds on image
                    @inbounds image[i,j] += evaluate(obj,i,j)
                end
            end
        end
    end
end

##################################
# 2D Histogram construction and utilities

"""
    calculate_edges(edges, xlim, ylim, nbins, xwidth, ywidth)

Function to calculate the bin edges for 2D histograms.
Returns `(xbins, ybins)` with both entries being ranges. 

# Keyword Arguments
 - `edges` is a tuple of vectors-like objects defining the left-side edges of the bins along the x-axis (edges[1]) and the y-axis (edges[2]). Example: `(-1.0:0.1:1.5, 22:0.1:27.2)`. If `edges` is provided, it will simply be returned.
 - `xlim`; a length-2 indexable object (e.g., a Vector{Float64} or NTuple{2,Float64)) giving the lower and upper bounds on the x-axis corresponding to the provided `colors` array. Example: `[-1.0, 1.5]`. This is only used if `edges==nothing`.
 - `ylim`; as `xlim` but  for the y-axis corresponding to the provided `mags` array. Example `[25, 20]`. This is only used if `edges==nothing`.
 - `nbins::NTuple{2,<:Integer}` is a 2-tuple of integers providing the number of bins to use along the x- and y-axes. This is only used if `edges==nothing`.
 - `xwidth`; the bin width along the x-axis for the `colors` array. This is only used if `edges==nothing` and `nbins==nothing`. Example: `0.1`. 
 - `ywidth`; as `xwidth` but for the y-axis corresponding to the provided `mags` array. Example: `0.1`.
"""
function calculate_edges(edges, xlim, ylim, nbins, xwidth, ywidth)
    if edges !== nothing
        return edges
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
    histogram_pix(x, edges)
    histogram_pix(x, edges::AbstractRange)

Returns the fractional index (i.e., pixel position) of value `x` given the left-aligned bin `edges`. The specialized form for `edges::AbstractRange` accepts reverse-sorted input (e.g., `edges=1:-0.1:0.0`) but `edges` must be sorted if you are providing an `edges` that is not an `AbstractRange`.

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
histogram_pix(x, edges::AbstractRange) = (x - first(edges)) / step(edges) + 1
function histogram_pix(x, edges)
    idx = searchsortedfirst(edges, x)
    if edges[idx] == x
        return idx
    else
        Δe = edges[idx] - edges[idx-1]
        return (idx-1) + (x - edges[idx-1]) / Δe
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

Returns a `StatsBase.Histogram` type containing the Hess diagram from the provided x-axis photometric `colors` and y-axis photometric magnitudes `mags`. These must all be vectors equal in length. You can either specify the bin edges directly via the `edges` keyword (e.g., `edges = (range(-0.5, 1.6, length=100), range(17.0, 26.0, length=100))`), or you can set the x- and y-limits via `xlim` and `ylim` and the number of bins as `nbins`, or you can omit `nbins` and instead pass the bin width in the x and y directions, `xwidth` and `ywidth`. See below for more info on the keyword arguments. To plot this with `PyPlot` you should do `plt.imshow(result.weights', origin="lower", ...)`.

# Keyword Arguments
 - `weights::AbstractVector{<:Number}` is a array of length equal to `colors` and `mags` that contains the probabilistic weights associated with each point. This is passed to `StatsBase.fit` as `StatsBase.Weights(weights)`. The following keyword arguments are passed to [`StarFormationHistories.calculate_edges`](@ref) to determine the bin edges of the histogram.
 - `edges` is a tuple of vector-like objects defining the left-side edges of the bins along the x-axis (edges[1]) and the y-axis (edges[2]). Example: `(-1.0:0.1:1.5, 22:0.1:27.2)`. If `edges` is provided, `weights` is the only other keyword that will be read; `edges` supercedes the other construction methods. 
 - `xlim`; a length-2 indexable object (e.g., a vector or tuple) giving the lower and upper bounds on the x-axis corresponding to the provided `colors` array. Example: `[-1.0, 1.5]`. This is only used if `edges` is not provided. 
 - `ylim`; as `xlim` but  for the y-axis corresponding to the provided `mags` array. Example `[25.0, 20.0]`. This is only used if `edges` is not provided.
 - `nbins::NTuple{2,<:Integer}` is a 2-tuple of integers providing the number of bins to use along the x- and y-axes. This is only used if `edges` is not provided.
 - `xwidth`; the bin width along the x-axis for the `colors` array. This is only used if `edges` and `nbins` are not provided. Example: `0.1`. 
 - `ywidth`; as `xwidth` but for the y-axis corresponding to the provided `mags` array. Example: `0.1`.
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
                       edges=nothing,
                       xlim=extrema(colors),
                       ylim=extrema(mags),
                       nbins=nothing,
                       xwidth=nothing,
                       ywidth=nothing)

Returns a `StatsBase.Histogram` type containing the Hess diagram where the points have been smoothed using a 2D asymmetric Gaussian with widths given by the provided `color_err` and `mag_err` and weighted by the given `weights`. These arrays must all be equal in size. This is akin to a KDE where each point is broadened by its own probability distribution. Keyword arguments are as explained in [`bin_cmd_smooth`](@ref) and [`StarFormationHistories.calculate_edges`](@ref). To plot this with `PyPlot` you should do `plt.imshow(result.weights', origin="lower", ...)`.

Recommended usage is to make a histogram of your observational data using [`bin_cmd`](@ref), then pass the resulting histogram bins through using the `edges` keyword to [`bin_cmd_smooth`](@ref) and [`partial_cmd_smooth`](@ref) to construct smoothed isochrone models. 
"""
function bin_cmd_smooth(colors, mags, color_err, mag_err;
                        weights = ones(promote_type(eltype(colors), eltype(mags)),
                                       size(colors)), edges=nothing,
                        xlim=extrema(colors), ylim=extrema(mags), nbins=nothing,
                        xwidth=nothing, ywidth=nothing)
    @assert axes(colors) == axes(mags) == axes(color_err) == axes(mag_err) == axes(weights)
    # Calculate edges from provided kws
    edges = calculate_edges(edges, xlim, ylim, nbins, xwidth, ywidth)
    # Construct matrix to hold the 2D histogram
    mat = zeros(Float64, length(edges[1])-1, length(edges[2])-1)
    # Get the pixel width in each dimension;
    # this currently only works if edges[1] and [2] are AbstractRange. 
    xwidth, ywidth = step(edges[1]), step(edges[2])
    for i in eachindex(colors)
        # Skip stars that are 3σ away from the histogram region in either x or y. 
        # if (((colors[i] - 3*color_err[i]) > maximum(edges[1]))  |
        #     ((colors[i] + 3*color_err[i]) < minimum(edges[1])))
        #     |
        #     ( ((mags[i] - 3*mag_err[i]) > maximum(edges[2])) | ((mags[i] + 3*mag_err[i]) <
        #                                                         minimum(edges[2])) )
        #     continue
        # end
        # Convert colors, mags, color_err, and mag_err from magnitude-space to
        # pixel-space in `mat`
        x0 = histogram_pix(colors[i], edges[1]) - 0.5
        y0 = histogram_pix(mags[i], edges[2]) - 0.5
        σx = color_err[i] / xwidth
        σy = mag_err[i] / ywidth
        # Construct the star object
        obj = GaussianPSFAsymmetric(x0, y0, σx, σy, weights[i], 0.0)
        # Insert the star object
        cutout_size = size(obj) # ( round(Int,3σx,RoundUp), round(Int,3σy,RoundUp) )
        addstar!(mat, obj, cutout_size) 
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
    new_mini, new_spacing = mini_spacing(m_ini, mags, 0.01, true)
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
    # Resample the isochrone magnitudes to a denser m_ini array
    # new_mini, new_spacing = mini_spacing(m_ini, imf, 1000, true)
    new_mini, new_spacing = mini_spacing(m_ini, mags[y_index], 0.01, true)
    # Interpolate only the mag vectors included in color_indices
    new_iso_mags = [interpolate_mini(m_ini, i, new_mini) .+ dmod for i in mags]
    colors = new_iso_mags[color_indices[1]] .- new_iso_mags[color_indices[2]]
    mag_err = [mag_err_funcs[color_indices[i]].( new_iso_mags[color_indices[i]] )
               for i in eachindex(mags)]
    color_err = [sqrt( mag_err[color_indices[1]][i]^2 + mag_err[color_indices[2]][i]^2 )
                 for i in eachindex(new_mini)]
    # Approximate the IMF weights on each star in the isochrone as
    # the trapezoidal rule integral across the bin. 
    # This is equivalent to the difference in the CDF across the bin as long as imf
    # is a properly normalized pdf i.e., if imf is a
    # Distributions.ContinuousUnivariateDistribution,
    # weights[i] = cdf(imf, m_ini[2]) - cdf(imf, m_ini[1])
    weights = Vector{Float64}(undef, length(new_mini) - 1)
    @inbounds @simd for i in eachindex(weights)
        weights[i] = new_spacing[i] *
            (dispatch_imf(imf,new_mini[i]) + dispatch_imf(imf,new_mini[i+1])) / 2
        # Incorporate completeness values
        weights[i] *= completeness_funcs[color_indices[1]](new_iso_mags[color_indices[1]][i]) *
            completeness_funcs[color_indices[2]](new_iso_mags[color_indices[2]][i])
        weights[i] *= normalize_value / mean_mass # Correct normalization
    end
    # sum(weights) is now the full integral over the imf pdf from
    # minimum(m_ini) -> maximum(m_ini). This is equivalent to
    # cdf(imf, maximum(m_ini)) - cdf(imf, minimum(m_ini)).
    return bin_cmd_smooth(colors[begin:end-1], new_iso_mags[y_index][begin:end-1],
                          color_err[begin:end-1], mag_err[y_index][begin:end-1];
                          weights=weights, edges=edges, xlim=xlim, ylim=ylim, nbins=nbins,
                          xwidth=xwidth, ywidth=ywidth)
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
