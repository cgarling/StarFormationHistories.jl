module SFH

# Import statements
import StatsBase: fit, Histogram, Weights, sample
import SpecialFunctions: erf
import Interpolations: linear_interpolation
import Distributions: UnivariateDistribution

# Code inclusion

# Code
##################################
##################################
# Isochrone utilities

"""
    interpolate_mini(m_ini, mags::Vector{<:Number}, new_mini)
    interpolate_mini(m_ini, mags::Vector{Vector{<:Number}}, new_mini)
    interpolate_mini(m_ini, mags::AbstractMatrix, new_mini)

Function to interpolate `mags` as a function of initial mass vector `m_ini` onto a new initial mass vector `new_mini`. `mags` can either be a vector of equal length to `m_ini`, designating magnitudes in a single filter, or a vector of vectors, designating multiple filters, or a matrix with each column designating a different filter.
"""
interpolate_mini(m_ini, mags::Vector{<:Number}, new_mini) =
    linear_interpolation(m_ini, mags)(new_mini)
interpolate_mini(m_ini, mags::Vector{Vector{<:Number}}, new_mini) =
    [linear_interpolation(m_ini, i)(new_mini) for i in mags]
interpolate_mini(m_ini, mags::AbstractMatrix, new_mini) =
    reduce(hcat, linear_interpolation(m_ini, i)(new_mini) for i in eachcol(mags))

"""
    mini_spacing(m_ini, imf_func, npoints::Int=1000, ret_spacing::Bool=false)

Returns a new sampling of `npoints` stellar masses given the initial mass vector `m_ini` from an isochrone and an `imf_func(mass)` functional that returns the PDF of the IMF for a given `mass`. The sampling is roughly even but slightly weighted to lower masses according to the IMF.

I am not currently happy with this implementation. It does not always return exactly `npoints`, and the random sampling means that the results are not easily replicable. Ideally we should replace with something else. Maybe we could attempt to equally partition the range of `m_ini` into `npoints` that span equal CDF intervals? Would require passing a `Distribution.UnivariateDistribution` with `quantile` defined. This should be part of a larger overhaul to ONLY use distributions in this package. Some quick code, where `imf` is a `Distributions.UnivariateDistribution`, like those defined in InitialMassFunctions.jl: 
```julia
julia> qvals = range(0.0, cdf(imf, maximum(m_ini)), length=1000)

julia> new_mini = quantile.(imf,qvals)
```
"""
function mini_spacing(m_ini, imf_func, npoints::Int=1000, ret_spacing::Bool=false)
    Δm = diff(m_ini)
    inv_imf_vals = inv.(imf_func.(m_ini[begin:end-1] .+ Δm ./ 2))
    point_intervals = round.(Int, inv_imf_vals ./ sum(inv_imf_vals) * npoints)
    # The minimum value in point_intervals should be 2 for the later call to `range`,
    # so if it's 1 we add 1, if it's zero we add 2. 
    # point_intervals[point_intervals .== 1] .+= 1
    @inbounds for i in eachindex( point_intervals )
        if point_intervals[i] == 0
            point_intervals[i] += 2
        elseif point_intervals[i] == 1
            point_intervals[i] += 1
        end
    end
    # return point_intervals
    # After the `if` clause below we discuss having to chop off the ends of the ranges
    # so as not to duplicate masses, so we want to avoid double-counting here as well. 
    point_sum = sum(point_intervals) - length(point_intervals)
    if point_sum < npoints # Pad out the array so we get the correct length
        nsamp = npoints - point_sum
        randidx = sample(1:length(point_intervals), npoints - point_sum, replace=true)
        # point_intervals[randidx] .+= 1
        @inbounds @simd for i in randidx
            point_intervals[i] += 1
        end
    end
    # range(start, stop, length=n) includes both the beginning and the ending points, so we'll essentially
    # double up on masses if we include the final point, so we have to remove it.
    # This works fine but is fairly slow, if speed becomes a problem we can probably rewrite this
    # as a faster loop. 
    new_mini = reduce(vcat, range(m_ini[i], m_ini[i+1], length=point_intervals[i])[begin:end-1] for i in 1:length(m_ini)-1)

    # Try new implementation
    # Δm = diff(m_ini)
    # inv_imf_vals = inv.(imf_func.(m_ini[begin:end-1] .+ Δm ./ 2))
    # point_intervals = round.(Int, inv_imf_vals ./ sum(inv_imf_vals) * npoints)
    # point_sum = sum(point_intervals) 
    # if point_sum < npoints # Pad out the array so we get the correct length
    #     randidx = sample(1:length(point_intervals), npoints - point_sum, replace=false)
    #     point_intervals[randidx] .+= 1
    # end
    
    # new_mini = Vector{eltype(m_ini)}(undef, sum(point_intervals))
    # current_num = 1
    # for i in 1:length(m_ini)-1
    #     Δx = (m_ini[i+1] - m_ini[i]) / point_intervals[i]
    #     for j in 0:point_intervals[i]-1
    #         new_mini[current_num] = m_ini[i] + Δx * j
    #         current_num += 1 
    #     end
    # end
    
    if !ret_spacing
        return new_mini
    else
        new_spacing = diff(new_mini)
        return new_mini, new_spacing
    end
end

##################################
# KDE models

"""
    GaussianPSFAsymmetric(x0::Real,y0::Real,σx::Real,σy::Real)
    GaussianPSFAsymmetric(x0::Real,y0::Real,σx::Real,σy::Real,A::Real)
    GaussianPSFAsymmetric(x0::Real,y0::Real,σx::Real,σy::Real,A::Real,B::Real)

`struct` for the 2D asymmetric Gaussian PSF without rotation (no θ). This model has an analytic integral which makes it fast; for reference, evaluation time is about 40% that of `GaussianPSF`. It might even be able to be faster if we enforce `σx=σy` in a different model.

# Parameters
 - `x0`, the center of the PSF model along the first matrix dimension
 - `y0`, the center of the PSF model along the second matrix dimension
 - `σx`, the Gaussian `σ` along the first matrix dimension
 - `σy`, the Gaussian `σ` along the first matrix dimension
 - `A`, the multiplicative constant in front of the Gaussian (normalization constant)
 - `B`, a constant additive background across the PSF
"""
struct GaussianPSFAsymmetric{T <: Real} # <: AnalyticPSFModel
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
parameters(model::GaussianPSFAsymmetric) = (model.x0, model.y0, model.σx, model.σy, model.A, model.B)
function Base.size(model::GaussianPSFAsymmetric)
    σx,σy = model.σx,model.σy
    return (ceil(Int,σx) * 10, ceil(Int,σy) * 10)
end
centroid(model::GaussianPSFAsymmetric) = (model.x0, model.y0)  
""" 
    gaussian_psf_asymmetric_integral_halfpix(x::Real,y::Real,x0::Real,y0::Real,σx::Real,σy::Real,A::Real,B::Real)

Exact analytic integral for the asymmetric, non-rotated 2D Gaussian. `A` is a normalization constant which is equal to overall integral of the function, not accounting for an additive background `B`. 
"""
gaussian_psf_asymmetric_integral_halfpix(x::Real,y::Real,x0::Real,y0::Real,σx::Real,σy::Real,A::Real,B::Real) = 
    0.25 * A * erf((x+0.5-x0) / (sqrt(2) * σx), (x-0.5-x0) / (sqrt(2) * σx)) * erf((y+0.5-y0) / (sqrt(2) * σy), (y-0.5-y0) / (sqrt(2) * σy)) + B
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

# Keyword Arguments
 - `edges` is a tuple of vectors-like objects defining the left-side edges of the bins along the x-axis (edges[1]) and the y-axis (edges[2]). Example: `(-1.0:0.1:1.5, 22:0.1:27.2)`. If `edges` is provided, it will simply be returned.
 - `xlim`; a length-2 indexable object (e.g., a Vector{Float64} or NTuple{2,Float64)) giving the lower and upper bounds on the x-axis corresponding to the provided `colors` array. Example: `[-1.0, 1.5]`. This is only used if `edges==nothing`.
 - `ylim`; as `xlim` but  for the y-axis corresponding to the provided `mags` array. Example `[25, 20]`. This is only used if `edges==nothing`.
 - `nbins::NTuple{2,<:Integer}` is a 2-tuple of integers providing the number of bins to use along the x- and y-axes. This is only used if `edges==nothing`.
 - `xwidth`; the bin width along the x-axis for the `colors` array. This is only used if `edges==nothing` and `nbins==nothing`. Example: `0.1`. 
 - `ywidth`; as `xwidth` but for the y-axis corresponding to the provided `mags` array. Example: `0.1`.
"""
function calculate_edges(edges, xlim, ylim, nbins, xwidth, ywidth)
# function calculate_edges(; edges=nothing, xlim=nothing, ylim=nothing, nbins=nothing, xwidth=nothing, ywidth=nothing)
    if edges !== nothing
        return edges
    else # if edges === nothing
        xlim, ylim = sort(xlim), sort(ylim)
        # Calculate nbins if it hasn't been provided. 
        if nbins === nothing
            if xwidth !== nothing && ywidth !== nothing

                xbins, ybins = round(Int, (xlim[2]-xlim[1])/xwidth), round(Int, (ylim[2]-ylim[1])/ywidth)
                nbins = (xbins, ybins)
            else
                throw(ArgumentError("If the keyword arguments `edges` and `nbins` are not provided, then `xwidth` and `ywidth` must be provided.")) 
            end
        end
        edges = (range(xlim[1], xlim[2], length=nbins[1]), range(ylim[1], ylim[2], length=nbins[2]))
        return edges
    end
end

"""
    histogram_pix(x, edges)
    histogram_pix(x, edges::AbstractRange)

Returns the fractional index (i.e., pixel position) of value `x` given the left-aligned bin `edges`. The specialized form for `edges::AbstractRange` accepts reverse-sorted input (e.g., `edges=1:-0.1:0.0`) but `edges` must be sorted if you are providing an `edges` that is not an `AbstractRange`.

# Examples
```jldoctest
julia> SFH.histogram_pix(0.5,0.0:0.1:1.0) ≈ 6
true

julia> (0.0:0.1:1.0)[6] == 0.5
true

julia> SFH.histogram_pix(0.55,0.0:0.1:1.0) ≈ 6.5
true

julia> SFH.histogram_pix(0.5,1.0:-0.1:0.0) ≈ 6
true

julia> SFH.histogram_pix(0.5,collect(0.0:0.1:1.0)) ≈ 6
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
    bin_cmd( colors, mags; weights = ones(promote_type(eltype(colors), eltype(mags)), size(colors)), edges=nothing, xlim=extrema(colors), ylim=extrema(mags), nbins=nothing, xwidth=nothing, ywidth=nothing)

Returns a `StatsBase.Histogram` type containing the unnormalized, 2D binned CMD (i.e., Hess diagram) from the provided x-axis photometric `colors` and y-axis photometric `mags`. These should be equal in size. You can either specify the bin edges directly via the `edges` keyword, or you can set the x- and y-limits via `xlim` and `ylim` and the number of bins as `nbins`, or you can omit `nbins` and instead pass the bin width in the x and y directions, `xwidth` and `ywidth`. See below for more info on the keyword arguments. To plot this with `PyPlot` you should do `plt.imshow(result.weights', origin="lower"...)`.

# Keyword Arguments
 - `weights` is a array of size equal to `colors` and `mags` that contains the probabilistic weights associated with each point. This is passed to `StatsBase.fit` as `StatsBase.Weights(weights)`.
The following keyword arguments are passed to `SFH.calculate_edges` to determine the bin edges of the histogram.
 - `edges` is a tuple of vectors-like objects defining the left-side edges of the bins along the x-axis (edges[1]) and the y-axis (edges[2]). Example: `(-1.0:0.1:1.5, 22:0.1:27.2)`. If `edges` is provided, `weights` is the only other keyword that will be read; `edges` supercedes the other construction methods. 
 - `xlim`; a length-2 indexable object (e.g., a Vector{Float64} or NTuple{2,Float64)) giving the lower and upper bounds on the x-axis corresponding to the provided `colors` array. Example: `[-1.0, 1.5]`. This is only used if `edges` is not provided. 
 - `ylim`; as `xlim` but  for the y-axis corresponding to the provided `mags` array. Example `[25, 20]`. This is only used if `edges` is not provided.
 - `nbins::NTuple{2,<:Integer}` is a 2-tuple of integers providing the number of bins to use along the x- and y-axes. This is only used if `edges` is not provided.
 - `xwidth`; the bin width along the x-axis for the `colors` array. This is only used if `edges` and `nbins` are not provided. Example: `0.1`. 
 - `ywidth`; as `xwidth` but for the y-axis corresponding to the provided `mags` array. Example: `0.1`.
"""
function bin_cmd( colors, mags; weights = ones(promote_type(eltype(colors), eltype(mags)), size(colors)), edges=nothing, xlim=extrema(colors), ylim=extrema(mags), nbins=nothing, xwidth=nothing, ywidth=nothing )
    @assert size(colors) == size(mags)
    edges = calculate_edges(edges, xlim, ylim, nbins, xwidth, ywidth)
    return fit(Histogram, (colors, mags), Weights(weights), edges)
end
"""
    result::StatsBase.Histogram =
    bin_cmd_smooth( colors, mags, color_err, mag_err; weights = ones(promote_type(eltype(colors), eltype(mags)), size(colors)), edges=nothing, xlim=extrema(colors), ylim=extrema(mags), nbins=nothing, xwidth=nothing, ywidth=nothing )

Returns a `StatsBase.Histogram` type containing the unnormalized, 2D CMD (i.e., Hess diagram) where the points have been smoothed using a 2D asymmetric Gaussian with widths given by the provided `color_err` and `mag_err` and weighted by the given `weights`. These should be equal in size. This is akin to a KDE where each point is broadened by its own PRF. Keyword arguments are as explained in [`bin_cmd_smooth`](@ref) and [`SFH.calculate_edges`](@ref). To plot this with `PyPlot` you should do `plt.imshow(result.weights', origin="lower"...)`.

Recommended usage is to make a histogram of your observational data using [`bin_cmd`](@ref), then pass the resulting histogram bins through using the `edges` keyword to [`bin_cmd_smooth`](@ref) and [`partial_cmd_smooth`](@ref) to evaluate smoothed isochrone models. 
"""
function bin_cmd_smooth( colors, mags, color_err, mag_err; weights = ones(promote_type(eltype(colors), eltype(mags)), size(colors)), edges=nothing, xlim=extrema(colors), ylim=extrema(mags), nbins=nothing, xwidth=nothing, ywidth=nothing )
    nstars = size(colors)
    @assert nstars == size(mags) == size(color_err) == size(mag_err) == size(weights)
    # Calculate edges from provided kws
    edges = calculate_edges(edges, xlim, ylim, nbins, xwidth, ywidth)
    # Construct matrix to hold the 2D histogram
    mat = zeros(Float64, length(edges[1])-1, length(edges[2])-1)
    # Get the pixel width in each dimension; this currently only works if edges[1] and [2] are AbstractRange. 
    xwidth, ywidth = step(edges[1]), step(edges[2])
    for i in eachindex(colors)
        # Convert colors, mags, color_err, and mag_err from magnitude-space to pixel-space in `mat`
        x0, y0, σx, σy = histogram_pix(colors[i], edges[1]), histogram_pix(mags[i], edges[2]),
        color_err[i] / xwidth, mag_err[i] / ywidth
        # Construct the star object
        obj = GaussianPSFAsymmetric(x0, y0, σx, σy, weights[i], 0.0)
        # Insert the star object
        cutout_size = size(obj) 
        # cutout_size = ( round(Int,3σx,RoundUp), round(Int,3σy,RoundUp) )
        # cutout_size=(10,10)
        addstar!(mat, obj, cutout_size) #, cutout_size::Tuple{Int,Int}=size(obj))
    end
    return Histogram(edges, mat, :left, false)
end

function partial_cmd( m_ini, colors, mags, imf_func; dmod=0.0, normalize_value=1.0, edges=nothing, xlim=extrema(colors), ylim=extrema(mags), nbins=nothing, xwidth=nothing, ywidth=nothing )
    # Resample the isochrone magnitudes to a denser m_ini array
    new_mini, new_spacing = mini_spacing(m_ini, imf_func, 1000, true)
    new_iso_colors = interpolate_mini(m_ini, colors, new_mini)
    new_iso_mags = interpolate_mini(m_ini, mags, new_mini) .+ dmod
    # Approximate the IMF weights on each star in the isochrone as
    # (weights[i] = (m[i+1] - m[i]) * pdf(imf, m[i])
    # weights = new_spacing .* imf_func.(new_mini[begin:end-1])
    # Might be better to use the actual trapezoidal rule to integrate the PDF across the grid. 
    # (weights[i] = (m[i+1] - m[i]) * ( pdf(imf, m[i]) + pdf(imf, m[i+1]) ) / 2
    # Looks like this amounts to a ~10% difference on average, which is significant actually. 
    weights = Vector{Float64}(undef, length(new_mini) - 1)
    @inbounds @simd for i in eachindex(weights)
        weights[i] = new_spacing[i] * (imf_func(new_mini[i]) + imf_func(new_mini[i+1])) / 2
    end
    weights = weights .* normalize_value ./ sum(weights)
    return bin_cmd( new_iso_colors[begin:end-1], new_iso_mags[begin:end-1]; weights=weights, edges=edges, xlim=xlim, ylim=ylim, nbins=nbins, xwidth=xwidth, ywidth=ywidth )
end

# Need to pass in completeness functions and use them when calculating the weights in addition to the
# IMF weighting. 
function partial_cmd_smooth( m_ini, mags, mag_err_funcs, y_index, color_indices, imf_func, completeness_funcs=[one for i in mags]; dmod=0.0, normalize_value=1.0, edges=nothing, xlim=nothing, ylim=nothing, nbins=nothing, xwidth=nothing, ywidth=nothing )
    # Resample the isochrone magnitudes to a denser m_ini array
    new_mini, new_spacing = mini_spacing(m_ini, imf_func, 1000, true)
    # Interpolate only the mag vectors included in color_indices
    new_iso_mags = [ interpolate_mini(m_ini, i, new_mini) .+ dmod for i in mags ] # dmod included here
    colors = new_iso_mags[color_indices[1]] .- new_iso_mags[color_indices[2]]
    mag_err = [ mag_err_funcs[color_indices[i]].( new_iso_mags[color_indices[i]] ) for i in eachindex(mags) ]
    color_err = [ sqrt( mag_err[color_indices[1]][i]^2 + mag_err[color_indices[2]][i]^2 ) for i in eachindex(new_mini) ]
    # Approximate the IMF weights on each star in the isochrone as
    # the trapezoidal rule integral across the bin. 
    # (weights[i] = (m[i+1] - m[i]) * ( pdf(imf, m[i]) + pdf(imf, m[i+1]) ) / 2
    # This is equivalent to the difference in the CDF across the bin as long as imf_func is a properly normalized pdf
    # i.e., if imf is a Distributions.UnivariateDistribution, weights[i] = cdf(imf, m_ini[2]) - cdf(imf, m_ini[1])
    weights = Vector{Float64}(undef, length(new_mini) - 1)
    # weights2 = Vector{Float64}(undef, length(new_mini) - 1)
    @inbounds @simd for i in eachindex(weights)
        weights[i] = new_spacing[i] * (imf_func(new_mini[i]) + imf_func(new_mini[i+1])) / 2
        # Incorporate completness values
        weights[i] *= completeness_funcs[color_indices[1]](new_iso_mags[color_indices[1]][i]) *
            completeness_funcs[color_indices[2]](new_iso_mags[color_indices[2]][i])
    end
    # return weights, weights2
    # weights3 = weights .* weights2
    # @assert all( weights3 .== weights ) # This returns true so it's fine
    # sum(weights) is now the full integral over the imf_func pdf from minimum(m_ini) -> maximum(m_ini).
    # This is equivalent to cdf(imf, maximum(m_ini)) - cdf(imf, minimum(m_ini)).
    # Previously we were dividing by sum(weights) here but I think that is wrong. 
    weights .= weights .* normalize_value # ./ sum(weights)
    return bin_cmd_smooth( colors[begin:end-1], new_iso_mags[y_index][begin:end-1],
                           color_err[begin:end-1], mag_err[y_index][begin:end-1]; weights=weights,
                           edges=edges, xlim=xlim, ylim=ylim, nbins=nbins, xwidth=xwidth, ywidth=ywidth )
end


# Method exports
export bin_cmd, bin_cmd_smooth, partial_cmd, partial_cmd_smooth

end # module
