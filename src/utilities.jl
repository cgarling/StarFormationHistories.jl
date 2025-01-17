#### Distance Utilities
"""
    distance_modulus(distance)
Finds distance modulus for distance in parsecs.

```math
μ = 5 \\times \\log_{10}(d) - 5
```
"""
distance_modulus(distance) = 5 * log10(distance) - 5 
"""
    distance_modulus_to_distance(dist_mod)
Converts distance modulus to distance in parsecs.

```math
d = 10^{μ/5 + 1}
```
"""
distance_modulus_to_distance(dist_mod) = exp10(dist_mod/5 + 1)
"""
    arcsec_to_pc(arcsec, dist_mod)
Converts on-sky angle in arcseconds to physical separation based on distance modulus under the small-angle approximation.

```math
r ≈ 10^{μ/5 + 1} \\times \\text{atan}(θ/3600)
```
"""
arcsec_to_pc(arcsec, dist_mod) = exp10(dist_mod/5 + 1) * atan(deg2rad(arcsec/3600))
"""
    pc_to_arcsec(pc, dist_mod)
Inverse of [`arcsec_to_pc`](@ref StarFormationHistories.arcsec_to_pc).

```math
θ ≈ \\text{tan}\\left( r / 10^{μ/5 + 1} \\right) \\times 3600
```
"""
pc_to_arcsec(pc, dist_mod) = rad2deg(tan(pc / exp10(dist_mod/5 + 1))) * 3600
"""
    angular_transformation_distance(angle, distance0, distance1)
Transforms an angular separation in arcseconds at distance `distance0` in parsecs to another distance `distance1` in parsecs. Uses the small angle approximation. 
"""
function angular_transformation_distance(angle, distance0, distance1)
    pc0 = arcsec_to_pc(angle, distance_modulus(distance0))
    return pc_to_arcsec(pc0, distance_modulus(distance1))
end

#### Luminosity Utilities
"""
    mag2flux(m, zpt=0)
Convert a magnitude `m` to a flux assuming a photometric zeropoint of `zpt`, defined as
the magnitude of an object that produces one count (or data number, DN) per second.
```jldoctest; setup = :(import StarFormationHistories: mag2flux)
julia> mag2flux(15.0, 25.0) ≈ exp10(4 * (25.0 - 15.0) / 10)
true
```
"""
mag2flux(m, zpt=0) = exp10(4 * (zpt - m) / 10)
"""
    flux2mag(f, zpt=0)
Convert a flux `f` to a magnitude assuming a photometric zeropoint of `zpt`, defined as
the magnitude of an object that produces one count (or data number, DN) per second.
```jldoctest; setup = :(import StarFormationHistories: flux2mag)
julia> flux2mag(10000.0, 25.0) ≈ 25.0 - 5 * log10(10000.0) / 2
true
```
"""
flux2mag(f, zpt=0) = zpt - 5 * log10(f) / 2
"""
    magerr(f, σf)
Returns an error in magnitudes given a flux and a flux uncertainty.

```jldoctest; setup = :(import StarFormationHistories: magerr)
julia> magerr(100.0, 1.0) ≈ 2.5 / log(10) * (1.0 / 100.0)
true
```
"""
magerr(f, σf) = 5//2 * σf / f / logten
"""
    fluxerr(f, σm)
Returns an error in flux given a flux and a magnitude uncertainty.

```jldoctest; setup = :(import StarFormationHistories: fluxerr)
julia> fluxerr(100.0, 0.01) ≈ (0.01 * 100.0) / 2.5 * log(10)
true
```
"""
fluxerr(f, σm) = σm * f * logten / 5//2
"""
    snr_magerr(σm)
Returns a signal-to-noise ratio ``(f/σf)`` given an uncertainty in magnitudes.

```jldoctest; setup = :(import StarFormationHistories: snr_magerr)
julia> snr_magerr(0.01) ≈ 2.5 / log(10) / 0.01
true
```
"""
snr_magerr(σm) = 5//2 / σm / logten
"""
    magerr_snr(snr)
Returns a magnitude uncertainty given a signal-to-noise ratio ``(f/σf)``.

```jldoctest; setup = :(import StarFormationHistories: magerr_snr)
julia> magerr_snr(100.0) ≈ 2.5 / log(10) / 100.0
true
```
"""
magerr_snr(snr) = 5//2 / snr / logten
# Absolute magnitude of Sun in V-band is 4.83 = 483//100
L_from_MV(absmagv) = mag2flux(absmagv, 483//100)
MV_from_L(lum) = flux2mag(lum, 483//100)

#### Metallicity utilities
"""
    Y_from_Z(Z, Y_p=0.2485, γ=1.78)
Calculates the helium mass fraction (Y) for a star given its metal mass fraction (Z) using the approximation `Y = Y_p + γ * Z`, with `Y_p` being the primordial helium abundance `Y_p=0.2485` as assumed for [PARSEC](http://stev.oapd.inaf.it/cmd/) isochrones and `γ=1.78` matching the input scaling for PARSEC as well. 
"""
Y_from_Z(Z, Y_p = 0.2485, γ = 1.78) = Y_p + γ * Z
"""
    X_from_Z(Z[, Yp, γ])
Calculates the hydrogen mass fraction (X) for a star given its metal mass fraction (Z) via `X = 1 - (Z + Y)`, with the helium mass fraction `Y` approximated via [`StarFormationHistories.Y_from_Z`](@ref). You may also provide the primordial helium abundance `Y_p` and `γ` such that `Y = Y_p + γ * Z`; these are passed through to `X_from_Z`. 
"""
X_from_Z(Z) = 1 - (Y_from_Z(Z) + Z)
X_from_Z(Z, Y_p) = 1 - (Y_from_Z(Z, Y_p) + Z)
X_from_Z(Z, Y_p, γ) = 1 - (Y_from_Z(Z, Y_p, γ) + Z)
"""
    MH_from_Z(Z, solZ=0.01524; Y_p = 0.2485, γ = 1.78)
Calculates [M/H] = log(Z/X) - log(Z/X)⊙. Given the provided solar metal mass fraction `solZ`, it calculates the hydrogen mass fraction X for both the Sun and the provided `Z` with [`StarFormationHistories.X_from_Z`](@ref). You may also provide the primordial helium abundance `Y_p` and `γ` such that `Y = Y_p + γ * Z`; these are passed through to `X_from_Z`. 

The present-day solar Z is measured to be 0.01524 ([Caffau et al. 2011](https://ui.adsabs.harvard.edu/abs/2011SoPh..268..255C/abstract)), but for PARSEC isochrones an [M/H] of 0 corresponds to Z=0.01471. This is because of a difference between the Sun's initial and present helium content caused by diffusion. If you want to reproduce PARSEC's scaling, you should set `solZ=0.01471`.

This function is an approximation and may not be suitable for precision calculations.
"""
function MH_from_Z(Z, solZ=0.01524; Y_p = 0.2485, γ = 1.78)
    # @assert all(var > 0 for var in (Z, solZ, Y_p)) "Metal mass fraction `Z`, solar metal mass fraction `solZ`, and primordial helium mass fraction `Y_p` must be greater than 0."
    X = X_from_Z(Z, Y_p, γ)
    # return log10(Z / X) - log10(solZ / X_from_Z(solZ, Y_p, γ))
    # By default log10 will throw an error for negative argument; such negative arguments
    # can be manifested during the linesearch procedure of a BFGS solve. To prevent errors
    # and enable the fit to proceed, we return NaN if X <= 0.
    # Also fuse expression into one log10 call for efficiency.
    return X > 0 ? log10(Z / (X * solZ) * X_from_Z(solZ, Y_p, γ)) : NaN
end
"""
    dMH_dZ(Z, solZ=0.01524; Y_p = 0.2485, γ = 1.78)
Partial derivative of [`MH_from_Z`](@ref StarFormationHistories.MH_from_Z) with respect to the input metal mass fraction `Z`. Used for [`LogarithmicAMR`](@ref StarFormationHistories.LogarithmicAMR).
"""
function dMH_dZ(Z, solZ=0.01524; Y_p = 0.2485, γ = 1.78)
    # @assert Z > 0 "Metal mass fraction `Z` must be greater than 0."
    return (Y_p - 1) / (logten * Z * (Y_p + Z + γ * Z - 1))
end
"""
    Z_from_MH(MH, solZ=0.01524; Y_p = 0.2485, γ = 1.78)
Calculates metal mass fraction `Z` assuming that the solar metal mass fraction is `solZ` and using the PARSEC relation for the helium mass fraction `Y = Y_p + γ * Z` with primordial helium abundance `Y_p = 0.2485`, and `γ = 1.78`.
"""
function Z_from_MH(MH, solZ=0.01524; Y_p = 0.2485, γ = 1.78)
    # [M/H] = log(Z/X)-log(Z/X)☉ with Z☉ = solz
    # Z/X = exp10( [M/H] + log(Z/X)☉ )
    # X = 1 - Y - Z
    # Y ≈ Y_p + γ * Z for parsec (see Y_from_Z above)
    # so X ≈ 1 - (Y_p + γ * Z) - Z = 1 - Y_p - (1 + γ) * Z
    # Substitute into line 2,
    # Z / (1 - Y_p - (1 + γ) * Z) = exp10( [M/H] + log(Z/X)☉ )
    # Z = (1 - Y_p - (1 + γ) * Z) * exp10( [M/H] + log(Z/X)☉ )
    # let A = exp10( [M/H] + log(Z/X)☉ )
    # Z = (1 - Y_p) * A - (1 + γ) * Z * A
    # Z + (1 + γ) * Z * A = (1 - Y_p) * A
    # Z (1 + (1 + γ) * A) = (1 - Y_p) * A
    # Z = (1 - Y_p) * A / (1 + (1 + γ) * A)
    # Originally had X_from_Z(solZ) without passing through the Y_p. Don't remember why
    zoverx = exp10(MH + log10(solZ / X_from_Z(solZ, Y_p, γ)))
    return (1 - Y_p) * zoverx / (1 + (1 + γ) * zoverx)
end
"""
    dZ_dMH(MH, solZ=0.01524; Y_p = 0.2485, γ = 1.78)
Partial derivative of [`Z_from_MH`](@ref StarFormationHistories.Z_from_MH) with respect to the input metal abundance `MH`. Used for [`LogarithmicAMR`](@ref StarFormationHistories.LogarithmicAMR).
"""
function dZ_dMH(MH, solZ=0.01524; Y_p = 0.2485, γ = 1.78)
    prefac = exp10(MH) * solZ
    X = X_from_Z(solZ, Y_p, γ)
    return -prefac * X * (Y_p - 1) * logten /
        (X + prefac * (1 + γ))^2
end

# PARSEC says that the solar Z is 0.0152 and Z/X = 0.0207, but they don't quite agree
# when assuming their provided Y=0.2485+1.78Z. We'll adopt their solZ here, but this
# should probably not be used for precision calculations.

#### Error and Completeness Utilities
"""
    η(m) = Martin2016_complete(m, A, m50, ρ)

Completeness model of [Martin et al. 2016](https://ui.adsabs.harvard.edu/abs/2016ApJ...833..167M/abstract) implemented as their Equation 7:

```math
\\eta(m) = \\frac{A}{1 + \\text{exp} \\left( \\frac{m - m_{50}}{\\rho} \\right)}
```

`m` is the magnitude of interest, `A` is the maximum completeness, `m50` is the magnitude at which the data are 50% complete, and `ρ` is an effective slope modifier.
"""
Martin2016_complete(m,A,m50,ρ) = A / (1 + exp((m - m50) / ρ))

"""
    exp_photerr(m, a, b, c, d)

Exponential model for photometric errors of the form

```math
\\sigma(m) = a^{b \\times \\left( m-c \\right)} + d
```

Reported values for some HST data were `a=1.05, b=10.0, c=32.0, d=0.01`. 
"""
exp_photerr(m, a, b, c, d) = a^(b * (m - c)) + d

"""
    process_ASTs(ASTs::Union{DataFrames.DataFrame,
                             TypedTables.Table},
                 inmag::Symbol,
                 outmag::Symbol,
                 bins::AbstractVector{<:Real},
                 selectfunc;
                 statistic=StatsBase.median)

Processes a table of artificial stars to calculate photometric completeness, bias, and error across the provided `bins`. This method has no default implementation and is implemented in package extensions that rely on either `DataFrames.jl` or `TypedTables.jl` being loaded into your Julia session to load the relevant method. This method therefore requires Julia 1.9 or greater to use.

# Arguments
 - `ASTs` is the table of artificial stars to be analyzed.
 - `inmag` is the column name in symbol format (e.g., :F606Wi) that corresponds to the intrinsic (input) magnitudes of the artificial stars.
 - `outmag` is the column name in symbol format (e.g., :F606Wo) that corresponds to the measured (output) magnitude of the artificial stars.
 - `bins` give the bin edges to be used when computing the binned statistics.
 - `selectfunc` is a method that takes a single row from `ASTs`, corresponding to a single artificial star, and returns a boolean that is `true` if the star is considered successfully measured.

# Keyword Arguments
 - `statistic` is the method that will be used to determine the bias and error, i.e., `bias = statistic(out .- in)` and `error = statistic(abs.(out .- in))`. By default we use `StatsBase.median`, but you could instead use a simple or sigma-clipped mean if so desired.

# Returns
This method returns a `result` of type `NTuple{4,Vector{Float64}}`. Each vector is of length `length(bins)-1`. `result` contains the following elements, each of which are computed over the provided `bins` considering only artificial stars for which `selectfunc` returned `true`:
 - `result[1]` contains the mean input magnitude of the stars in each bin.
 - `result[2]` contains the completeness value measured for each bin, defined as the fraction of input stars in each bin for which `selectfunc` returned `true`.
 - `result[3]` contains the photometric bias measured for each bin, defined as `statistic(out .- in)`, where `out` are the measured (output) magnitudes and `in` are the intrinsic (input) magnitudes.
 - `result[4]` contains the photometric error measured for each bin, defined as `statistic(abs.(out .- in))`, with `out` and `in` defined as above.

# Examples
Let
 - `F606Wi` be a vector containing the input magnitudes of your artificial stars
 - `F606Wo` be a vector containing the measured magnitudes of the artificial stars, where a value of 99.999 indicates a non-detection.
 - `flag` be a vector of booleans that indicates whether the artificial star passed additional quality cuts (star-galaxy separation, etc.)
You could call this method as

```julia
import TypedTables: Table
process_ASTs(Table(input=F606Wi, output=F606Wo, good=flag),
             :input, :output, minimum(F606Wi):0.1:maximum(F606Wi),
             x -> (x.good==true) & (x.output != 99.999))
```

See also the tests in `test/utilities/process_ASTs_test.jl`.
"""
function process_ASTs end

# Numerical utilities
"""
    vecs_to_svecs(vecs::Vararg{T, N}) where {T <: AbstractVector, N}
    vecs_to_svecs(x::AbstractVector{<:AbstractVector})
Convert a vector of length `a` of vectors of length `b` to a length `b` vector of length `a` `StaticArray.SVectors`. This data format can be put into `Interpolations.interpolate` as the y-value for simultaneous interpolation of multiple y-values given one x value. This function is type unstable.
```jldoctest; setup = :(import StarFormationHistories: vecs_to_svecs; import StaticArrays: SVector)
julia> vecs_to_svecs([1,2], [3,4]) == [SVector(1,3), SVector(2,4)]
true
julia> vecs_to_svecs([[1,2], [3,4]]) == [SVector(1,3), SVector(2,4)]
true
```
"""
vecs_to_svecs() = SVector{0,Float64}[] # Covers ambiguity in case vararg length is 0
vecs_to_svecs(vecs::Vararg{T, N}) where {T <: AbstractVector, N} = [SVector{N, eltype(T)}(tup) for tup in zip(vecs...)]
vecs_to_svecs(x::AbstractVector{<:AbstractVector}) = vecs_to_svecs(x...)

"""
    tups_to_mat(tups::Vararg{T, N}) where {M, S, N, T <: NTuple{M, S}}

Takes a sequence of `N` `NTuples`, each of which has length `M` and element type `S`, and converts them into a matrix of size `(M, N)`.

```jldoctest; setup = :(import StarFormationHistories: tups_to_mat)
julia> tups_to_mat((1,2,3), (4,5,6)) == [[1,2,3] [4,5,6]]
true
```
"""
# function tups_to_mat(tups::Vararg{T, N}) where {M, S, N, T <: NTuple{M, S}}
function tups_to_mat(tups::T...) where {M, S, T <: NTuple{M, S}}
    N = length(tups)
    mat = Matrix{S}(undef, M, N)
    for i in 1:N
        mat[:,i] .= tups[i]
    end
    return mat
end
# Cover length-0 case to prevent unbound arguments
tups_to_mat() = Matrix{Float64}(undef, (0, 0))
"""
    tups_to_mat(tups::Vararg{T, N}) where {T <: Tuple, N}

Implementation of `tups_to_mat` for `tups` which do not all have identical element types.

```jldoctest; setup = :(import StarFormationHistories: tups_to_mat)
julia> tups_to_mat((1.0,2,3), (4,5,6)) == Float64[[1,2,3] [4,5,6]]
true
```
"""
# function tups_to_mat(tups::Vararg{Tuple, N}) where N # Don't specialize on N...
function tups_to_mat(tups::Tuple...)
    @assert allequal(length, tups) "All tuples passed as arguments to `tups_to_mat` must have same length."
    N = length(tups)
    S = reduce(promote_type, promote_type(typeof.(tup)...) for tup in tups)
    M = length(first(tups))
    mat = Matrix{S}(undef, M, N)
    for i in 1:N
        tup = tups[i]
        for j in 1:M
            mat[j,i] = convert(S, tup[j])
        end
    end
    return mat
end
tups_to_mat(itr...) = tups_to_mat(Tuple(i) for i in itr)
"""
    tups_to_mat(itr)

Takes a length-`N` iterable whose elements have length `M` and converts it into a matrix of size `(M, N)`. Often useful for `itr::AbstractArray`.

```jldoctest; setup = :(import StarFormationHistories: tups_to_mat)
julia> tups_to_mat([(1,2,3), (4,5,6)]) == [[1,2,3] [4,5,6]]
true
```
"""
tups_to_mat(itr) = tups_to_mat(itr...)

# """
#     tups_to_mat(tups::AbstractArray{T}) where {T <: Tuple}

# Takes an `AbstractArray` which has `N` elements that are `Tuple`s of identical length `M` and converts it into a matrix of size `(M, N)`.

# ```jldoctest; setup = :(import StarFormationHistories: tups_to_mat)
# julia> tups_to_mat([(1,2,3), (4,5,6)]) == [[1,2,3] [4,5,6]]
# true
# ```
# """
# tups_to_mat(tups::AbstractArray{T}) where {T <: Tuple} = tups_to_mat(tups...)

# function tups_to_mat(itr)
#     iter = iterate(itr)
#     M = length(iter[1])
#     N = length(itr)
#     # S = mapreduce(typeof, promote_type, iter[1])
#     S = reduce(promote_type, promote_type(typeof.(tup)...) for tup in itr)
#     # return reshape(reinterpret(S, itr), (N,:))
#     mat = Matrix{S}(undef, M, N)
#     # for i in 1:N
#     #     ii = iterate(itr, i)
#     #     for j in 1:M
#     #         mat[j,i] = convert(S, tup[j])
#     #     end
#     # end
#     # state = firstiter[2]
#     # mat[:,state] .= firstiter[1]
#     while !isnothing(iter)
#         for j in 1:M
#             mat[j,iter[2]-1] = convert(S, iter[1][j])
#         end
#         iter = iterate(itr, iter[2])
#         # iter[2] |> display
#     end
#     return mat
# end

# vecs_to_svecs(vecs::Vararg{<:AbstractArray{<:Number},N}) where {N} = vecs
# old implementation
# vecs_to_svecs(x::AbstractVector{<:AbstractVector}) =
#     reinterpret(SVector{length(x),eltype(first(x))}, vec(permutedims(hcat(x...))))
# julia> vecs_to_svecs([[1,2], [3,4]])
# 2-element Vector{SVector{2, Int64}}:
#  [1, 3]
#  [2, 4]
# function estimate_mode(data::AbstractVector{<:Real})
#     bw = KernelDensity.default_bandwidth(data) / 10
#     KDE = KernelDensity.kde(data; bandwidth=bw)
#     return KDE.x[findmax(KDE.density)[2]]
# end
