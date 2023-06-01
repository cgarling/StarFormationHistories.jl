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
arcsec_to_pc(arcsec, dist_mod) = exp10(dist_mod/5 + 1) * atan( deg2rad(arcsec/3600) )
"""
    pc_to_arcsec(pc, dist_mod)
Inverse of [`StarFormationHistories.arcsec_to_pc`](@ref).

```math
θ ≈ \\text{tan}\\left( r / 10^{μ/5 + 1} \\right) \\times 3600
```
"""
pc_to_arcsec(pc, dist_mod) = rad2deg( tan( pc / exp10(dist_mod/5 + 1) ) ) * 3600
"""
    angular_transformation_distance(angle, distance0, distance1)
Transforms an angular separation in arcseconds at distance `distance0` in parsecs to another distance `distance1` in parsecs. Uses the small angle approximation. 
"""
function angular_transformation_distance(angle, distance0, distance1)
    pc0 = arcsec_to_pc(angle, distance_modulus(distance0))
    return pc_to_arcsec(pc0, distance_modulus(distance1))
end

#### Luminosity Utilities
L_from_MV(absmagv) = exp10(0.4 * (4.8 - absmagv))
MV_from_L(lum) = 4.8 - 2.5 * log10(lum)
""" Luminosity in watts. """
M_bol_from_L(lum) = 71.1974 - 2.5 * log10(lum)
""" Returns watts. """
L_from_M_bol( absbolmag ) = exp10((71.1974 - absbolmag)/2.5)
"""
    find_Mv_flat_mu(μ, area, dist_mod)
Given a constant surface brightness `μ`, an angular area `area`, and a distance modulus `dist_mod`, returns the magnitude of the feature. 
"""
function find_Mv_flat_mu(μ, area, dist_mod)
    L = L_from_MV(μ - dist_mod)
    L_total = L * area
    return MV_from_L(L_total)
end

#### Metallicity utilities
"""
    Y_from_Z(Z, Y_p=0.2485)
Calculates the helium mass fraction (Y) for a star given its metal mass fraction (Z) using the approximation `Y = Y_p + 1.78Z`, with `Y_p` being the primordial helium abundance equal to 0.2485 as assumed for [PARSEC](http://stev.oapd.inaf.it/cmd) isochrones. 
"""
Y_from_Z(Z, Y_p = 0.2485) = Y_p + 1.78Z
"""
    X_from_Z(Z)
Calculates the hydrogen mass fraction (X) for a star given its metal mass fraction (Z) via `X = 1 - (Z + Y)`, with the helium mass fraction `Y` approximated via [`StarFormationHistories.Y_from_Z`](@ref). 
"""
X_from_Z(Z) = 1 - (Y_from_Z(Z) + Z)
"""
    MH_from_Z(Z, solZ=0.01524)
Calculates [M/H] = log(Z/X) - log(Z/X)⊙. Given the provided solar metal mass fraction `solZ`, it calculates the hydrogen mass fraction X for both the Sun and the provided `Z` with [`StarFormationHistories.X_from_Z`](@ref).

The present-day solar Z is measured to be 0.01524 ([Caffau et al. 2011](https://ui.adsabs.harvard.edu/abs/2011SoPh..268..255C/abstract)), but for PARSEC isochrones an [M/H] of 0 corresponds to Z=0.01471. This is because of a difference between the Sun's initial and present helium content caused by diffusion. If you want to reproduce PARSEC's scaling, you should set `solZ=0.01471`.

This function is an approximation and may not be suitable for precision calculations.
"""
MH_from_Z(Z, solZ=0.01524) = log10(Z / X_from_Z(Z)) - log10(solZ / X_from_Z(solZ))
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
Martin2016_complete(m,A,m50,ρ) = A / (1 + exp((m-m50) / ρ))

"""
    exp_photerr(m, a, b, c, d)

Exponential model for photometric errors of the form

```math
\\sigma(m) = a^{b \\times \\left( m-c \\right)} + d
```

Reported values for some HST data were `a=1.05, b=10.0, c=32.0, d=0.01`. 
"""
exp_photerr(m, a, b, c, d) = a^(b * (m-c)) + d

################################################
#### Interpret arguments for generate_mock_stars
"""
    new_mags = ingest_mags(mini_vec::AbstractVector, mags::AbstractVector{T}) where {S <: Number, T <: AbstractVector{S}}
    new_mags = ingest_mags(mini_vec::AbstractVector, mags::AbstractMatrix{S}) where S <: Number

Reinterprets provided `mags` to be in the correct format for input to `Interpolations.interpolate`.

# Returns
 - `new_mags::Base.ReinterpretArray{StaticArrays.SVector}`: a `length(mini_vec)` vector of `StaticArrays.SVectors` containing the same data as `mags` but formatted for input to `Interpolations.interpolate`.
"""
function ingest_mags(mini_vec::AbstractVector, mags::AbstractMatrix{S}) where S <: Number
    if ndims(mags) != 2 # Check dimensionality of mags argument
        throw(ArgumentError("`generate_stars...` received a `mags::AbstractMatrix{<:Real}` with `ndims(mags) != 2`; when providing a `mags::AbstractMatrix{<:Real}`, it must always be 2-dimensional."))
    end
    nstars = length(mini_vec)
    shape = size(mags)
    if shape[1] == nstars
        return reinterpret(SVector{shape[2],S}, vec(permutedims(mags)))
    elseif shape[2] == nstars
        return reinterpret(SVector{shape[1],S}, vec(mags))
    else
        throw(ArgumentError("`generate_stars...` received a misshapen `mags` argument. When providing a `mags::AbstractMatrix{<:Real}`, then it should be 2-dimensional and have size of (N,M) or (M,N), where N is the number of elements in `mini_vec`, and M is the number of filters represented in the `mags` argument."))
    end
end
function ingest_mags(mini_vec::AbstractVector, mags::AbstractVector{T}) where {S <: Number, T <: AbstractVector{S}}
    if length(mags) != length(mini_vec)
        nstars = length(first(mags))
        if ~mapreduce(x->isequal(nstars,length(x)), &, mags)
            throw(ArgumentError("`generate_stars...` received a misshapen `mags` argument. When providing a `mags::AbstractVector{AbstractVector{<:Real}}` with `length(mags)!=length(mini_vec)`, then each element of `mags` should have length equal to `length(mini_vec)`."))
        else
            nfilters = length(mags)
            return reinterpret(SVector{nfilters,S}, vec(permutedims(hcat(mags...))))
        end
    else
        # Check that every vector in mags has equal length.
        nfilters = length(first(mags))
        if eltype(mags) <: SVector
            return mags
        elseif ~mapreduce(x->isequal(nfilters,length(x)), &, mags)
            throw(ArgumentError("`generate_stars...` received a misshapen `mags` argument. When providing a `mags::AbstractVector{AbstractVector{<:Real}}` with `length(mags)==length(mini_vec)`, each element of `mags` must have equal length, representing the number of filters being used."))
        else
            return reinterpret(SVector{nfilters,S}, vec(hcat(mags...)))
        end
    end
end
ingest_mags(mini_vec, mags) = throw(ArgumentError("There is no `ingest_mags` method for the provided types of `mini_vec` and `mags`. See the documentation for the public functions, (e.g., [generate_stars_mass](@ref)), for information on valid input types.")) # General fallback in case the types are not recognized.

"""
    (new_mini_vec, new_mags) = sort_ingested(mini_vec::AbstractVector, mags::AbstractVector)

Takes `mini_vec` and `mags` and ensures that `mini_vec` is sorted (sometimes in PARSEC isochrones they are not) and calls `Interpolations.deduplicate_knots!` on `mini_vec` to ensure there are no repeat entries. Arguments must satisfy `length(mini_vec) == length(mags)`. 
"""
function sort_ingested(mini_vec::AbstractVector, mags::AbstractVector)
    @assert axes(mini_vec) == axes(mags)
    idx = sortperm(mini_vec)
    if idx != eachindex(mini_vec)
        mini_vec = mini_vec[idx]
        mags = mags[idx]
    end
    deduplicate_knots!(mini_vec) # Interpolations.jl function. 
    return mini_vec, mags
end

"""
    (mmin, mmax) = mass_limits(mini_vec::AbstractVector{<:Number}, mags::AbstractVector{T},
                     mag_names::AbstractVector{String}, mag_lim::Number,
                     mag_lim_name::String) where T <: AbstractVector{<:Number}

Calculates initial mass limits that reflect a given faint-end magnitude limit.

# Arguments
 - `mini_vec::AbstractVector{<:Number}`: a length `nstars` vector containing initial stellar masses.
 - `mags::AbstractVector{<:AbstractVector{<:Number}}`: a length `nstars` vector, with each element being a length `nfilters` vector giving the magnitudes of each star in the filters `mag_names`.
 - `mag_names::AbstractVector{String}`: a vector giving the names of each filter as strings.
 - `mag_lim::Number`: the faint-end magnitude limit you wish to use.
 - `mag_lim_name::String`: the name of the filter in which `mag_lim` is to be applied. Must be contained in `mag_names`.

# Returns
 - `mmin::eltype(mini_vec)`: the initial mass corresponding to your requested `mag_lim` in the filter `mag_lim_name`. If all stars provided are brighter than your requested `mag_lim`, then this will be equal to `minimum(mini_vec)`.
 - `mmax::eltype(mini_vec)`: the maximum valid mass in `mini_vec`; simply `maximum(mini_vec)`.

# Examples
```julia
julia> mass_limits([0.05,0.1,0.2,0.3], [[4.0],[3.0],[2.0],[1.0]], ["F090W"], 2.5, "F090W")
(0.15, 0.3)

julia> mass_limits([0.05,0.1,0.2,0.3], [[4.0,3.0],[3.0,2.0],[2.0,1.0],[1.0,0.0]], ["F090W","F150W"], 2.5, "F090W")
(0.15, 0.3)
```
"""
function mass_limits(mini_vec::AbstractVector{<:Number}, mags::AbstractVector{T},
                     mag_names::AbstractVector{String}, mag_lim::Number,
                     mag_lim_name::String) where T <: AbstractVector{<:Number}
    @assert axes(mini_vec) == axes(mags) 
    mmin, mmax = extrema(mini_vec)
    # Update mmin respecing `mag_lim`, if provided.
    if !isfinite(mag_lim)
        return mmin, mmax
    else
        idxmag = findfirst(x->x==mag_lim_name, mag_names) # Find the index into mag_lim_names where == mag_lim_name.
        idxmag == nothing && throw(ArgumentError("Provided `mag_lim_name` is not contained in provided `mag_names` array."))
        if mag_lim < mags[findfirst(x->x==mmin, mini_vec)][idxmag]
            tmp_mags = [i[idxmag] for i in mags]
            mag_lim < minimum(tmp_mags) && throw(DomainError(mag_lim, "The provided `mag_lim` is brighter than all the stars in the input `mags` array assuming the input distance modulus `dist_mod`. Revise your arguments."))
            # Solve for stellar initial mass where mag == mag_lim by constructing interpolator and root-finding.
            itp = interpolate((mini_vec,), tmp_mags, Gridded(Linear()))
            mmin2 = find_zero(x -> itp(x) - mag_lim, (mmin, mmax))
            return mmin2, mmax
        else
            return mmin, mmax
        end
    end
end

##############################################################
#### Types and methods for non-interacting binary calculations
""" `StarFormationHistories.AbstractBinaryModel` is the abstract supertype for all types that are used to model multi-star systems in the package. All concrete subtypes must implement the [`StarFormationHistories.sample_system`](@ref) method and the `Base.length` method, which should return an integer indicating the number of stars per system that can be sampled by the model; this is equivalent to the length of the mass vector returned by `sample_system`. """
abstract type AbstractBinaryModel end
Base.Broadcast.broadcastable(m::AbstractBinaryModel) = Ref(m)
"""
    NoBinaries()
The `NoBinaries` type indicates that no binaries of any kind should be created. """
struct NoBinaries <: AbstractBinaryModel end
Base.length(::NoBinaries) = 1
"""
    RandomBinaryPairs(fraction::Real)
The `RandomBinaryPairs` type takes one argument `0 <= fraction::Real <= 1` that denotes the number fraction of binaries (e.g., 0.3 for 30% binary fraction) and will sample binaries as random pairs of two stars drawn from the same single-star IMF. This model will ONLY generate up to one additional star -- it will not generate any 3+ star systems. This model typically incurs a 10--20% speed penalty relative to `NoBinaries`. """
struct RandomBinaryPairs{T <: Real} <: AbstractBinaryModel
    fraction::T
    # Outer-only constructor to guarantee support
    # See https://docs.julialang.org/en/v1/manual/constructors/#Outer-only-constructors
    function RandomBinaryPairs(fraction::T) where T <: Real
        @assert (fraction >= zero(T)) && (fraction <= one(T))
        new{T}(fraction)
    end
end
Base.length(::RandomBinaryPairs) = 2
"""
    BinaryMassRatio(fraction::Real, qdist::Distributions.ContinuousUnivariateDistribution)
The `BinaryMassRatio` type takes two arguments; the binary fraction `0 <= fraction::Real <= 1` and a continuous univariate distribution `qdist` from which to sample binary mass ratios, defined as the ratio of the secondary mass to the primary mass: ``q = \\text{M}_S / \\text{M}_P``. The provided `qdist` must have the proper support of `(minimum(qdist) >= zero(V)) && (maximum(qdist) <= one(V))`; users may find the [`Distributions.truncated`](https://juliastats.org/Distributions.jl/stable/truncate/#Distributions.truncated) method useful for enforcing this support on more general distributions. 
"""
struct BinaryMassRatio{T <: Real, S <: Distribution{Univariate, Continuous}} <: AbstractBinaryModel
    fraction::T
    qdist::S
    # Outer-only constructor to guarantee support
    # See https://docs.julialang.org/en/v1/manual/constructors/#Outer-only-constructors
    function BinaryMassRatio(fraction::T, qdist::S=Uniform()) where {T <: Real, S <: Distribution{Univariate, Continuous}}
        V = eltype(qdist)
        @assert (fraction >= zero(T)) && (fraction <= one(T)) && (minimum(qdist) >= zero(V)) && (maximum(qdist) <= one(V))
        new{T,S}(fraction, qdist)
    end
end
Base.length(::BinaryMassRatio) = 2

#  - `itp`: a callable object that returns the magnitudes of a single star with mass `m` when called as `itp(m)`; may return an `AbstractVector` with each entry corresponding to a different photometric filter
# Returns
#  - `new_mags`: the effective magnitude of the multi-star system derived by summing the luminosity of the individual stars
"""
    (binary_mass, new_mags) = sample_binary(mass, mmin, mmax, mags, imf, itp, rng::AbstractRNG, binarymodel::StarFormationHistories.AbstractBinaryModel)

Simulates the effects of unresolved binaries on stellar photometry. Implementation depends on the choice of `binarymodel`.

# Arguments
 - `mass`: the initial mass of the single star
 - `mmin`: minimum mass to consider for stellar companions
 - `mmax`: maximum mass to consider for stellar companions
 - `mags`: a vector-like object giving the magnitudes of the single star in each filter
 - `imf`: an object implementing `rand(imf)` to draw a random single-star mass
 - `itp`: a callable object that returns the magnitudes of a star with mass `m` when called as `itp(m)`
 - `rng::AbstractRNG`: the random number generator to use when sampling stars
 - `binarymodel::StarFormationHistories.AbstractBinaryModel`: an instance of a binary model that determines which implementation will be used; currently provided options are [`NoBinaries`](@ref) and [`Binaries`](@ref)

# Returns
 - `binary_mass`: the total mass of the additional stellar companions
 - `new_mags`: the effective magnitude of the multi-star system
"""
@inline sample_binary(mass, mmin, mmax, mags, imf, itp, rng::AbstractRNG, binarymodel::NoBinaries) = zero(mass), mags
"""
    masses = sample_system(imf, rng::AbstractRNG, binarymodel::StarFormationHistories.AbstractBinaryModel)

Simulates the effects of non-interacting, unresolved stellar companions on stellar photometry. Implementation depends on the choice of `binarymodel`.

# Arguments
 - `imf`: an object implementing `rand(imf)` to draw a random mass for a single star or a stellar system (depends on choice of `binarymodel`)
 - `rng::AbstractRNG`: the random number generator to use when sampling stars
 - `binarymodel::StarFormationHistories.AbstractBinaryModel`: an instance of a binary model that determines which implementation will be used; currently provided options are [`NoBinaries`](@ref) and [`RandomBinaryPairs`](@ref).

# Returns
 - `masses::SVector{N,eltype(imf)}`: the masses of the individual stars sampled in the system in descending order
"""
@inline sample_system(imf, rng::AbstractRNG, binarymodel::NoBinaries) = SVector{1,eltype(imf)}(rand(rng, imf))

@inline function sample_binary(mass, mmin, mmax, mags, imf, itp, rng::AbstractRNG, binarymodel::RandomBinaryPairs)
    frac = binarymodel.fraction
    r = rand(rng) # Random uniform number
    if r <= frac  # Generate a binary star
        mass_new = rand(rng, imf)
        if (mass_new < mmin) | (mass_new > mmax) # Sampled mass is outside of valid range
            return mass_new, mags 
        end
        mags_new = itp(mass_new)
        result = MV_from_L.( L_from_MV.(mags) .+ L_from_MV.(mags_new) )
        return mass_new, result
    else
        return zero(mass), mags
    end
end

@inline function sample_system(imf, rng::AbstractRNG, binarymodel::RandomBinaryPairs)
    frac = binarymodel.fraction
    r = rand(rng) # Random uniform number
    mass1 = rand(rng, imf) # First star mass
    if r <= frac  # Generate a binary star
        mass2 = rand(rng, imf)
        return sort( SVector{2,eltype(imf)}(mass1, mass2); rev=true )
    else
        return SVector{2,eltype(imf)}(mass1, zero(eltype(imf)))
    end
end

@inline function sample_binary(mass, mmin, mmax, mags, imf, itp, rng::AbstractRNG, binarymodel::BinaryMassRatio)
    frac = binarymodel.fraction
    qdist = binarymodel.qdist
    T = eltype(qdist)

    r = rand(rng) # Random uniform number
    if r <= frac  # (Try to) Generate a binary companion
        # qmin = max( minimum(qdist), mmin / (mass - mmin) ) # Solve for minimum mass ratio
        # qmax = min( maximum(qdist), mmax / mass - 1 )      # Solve for maximum mass ratio
        # if (qmin > maximum(qdist)) || (qmax < minimum(qdist))
        #     return SVector{2,T}(mass, zero(T))
        # end
        q = rand(rng, qdist) # Sample binary mass ratio M_secondary / M_primary
        M_p = mass / (q + 1) # Primary mass
        M_s = mass - M_p
        if (M_p < mmin) & (mass >= mmin)
            # return SVector{2,T}(mass, zero(T)), mags # itp(mass)
            return zero(T), mags 
        elseif M_s < mmin # (M_s < mmin) | (M_p > mmax)
            # return SVector{2,T}(M_p, M_s), itp(M_p)
            return zero(T), itp(M_p) # mags 
        else # Compute mags and combine
            new_mags_p = itp(M_p)
            new_mags_s = itp(M_s)
            new_mags = MV_from_L.( L_from_MV.(new_mags_p) .+ L_from_MV.(new_mags_s) )
            # return SVector{2,T}(M_p, M_s), new_mags
            return zero(T), new_mags
        end
    else
        # return SVector{2,T}(mass, zero(T)), mags # itp(mass)
        return zero(T), mags 
    end
end

@inline function sample_system(imf, rng::AbstractRNG, binarymodel::BinaryMassRatio)
    frac = binarymodel.fraction
    qdist = binarymodel.qdist

    mass = rand(rng, imf) # Total system mass
    r = rand(rng) # Random uniform number
    if r <= frac  # Generate a binary companion
        q = rand(rng, qdist) # Sample binary mass ratio q = M_secondary / M_primary
        M_p = mass / (q + 1) # Primary mass
        M_s = mass - M_p     # Secondary mass
        return SVector{2,eltype(imf)}(M_p, M_s)
    else
        return SVector{2,eltype(imf)}(mass, zero(eltype(imf)))
    end
end

#########################################################
#### Functions to generate mock galaxy catalogs from SSPs

"""
    (sampled_masses, sampled_mags) = generate_stars_mass(mini_vec::AbstractVector{<:Number}, mags, mag_names::AbstractVector{String}, limit::Number, imf::Distributions.Sampleable{Distributions.Univariate, Distributions.Continuous}; dist_mod::Number=0, rng::Random.AbstractRNG=Random.default_rng(), mag_lim::Number=Inf, mag_lim_name::String="V", binary_model::StarFormationHistories.AbstractBinaryModel=StarFormationHistories.RandomBinaryPairs(0.3))

Generates a random sample of stars from an isochrone defined by the provided initial stellar masses `mini_vec`, absolute magnitudes `mags`, and filter names `mag_names` with total population birth stellar mass `limit` (e.g., 1e5 solar masses). Initial stellar masses are sampled from the provided `imf`. 

# Arguments
 - `mini_vec::AbstractVector{<:Number}` contains the initial masses (in solar masses) for the stars in the isochrone; must be mutable as we call `Interpolations.deduplicate_knots!(mini_vec)`.
 - `mags` contains the absolute magnitudes from the isochrone in the desired filters corresponding to the same stars as provided in `mini_vec`. `mags` is internally interpreted and converted into a standard format by [`StarFormationHistories.ingest_mags`](@ref). Valid inputs are:
    - `mags::AbstractVector{AbstractVector{<:Number}}`, in which case the length of the outer vector `length(mags)` can either be equal to `length(mini_vec)`, in which case the length of the inner vectors must all be equal to the number of filters you are providing, or the length of the outer vector can be equal to the number of filters you are providing, and the length of the inner vectors must all be equal to `length(mini_vec)`; this is the more common use-case.
    - `mags::AbstractMatrix{<:Number}`, in which case `mags` must be 2-dimensional. Valid shapes are `size(mags) == (length(mini_vec), nfilters)` or `size(mags) == (nfilters, length(mini_vec))`, with `nfilters` being the number of filters you are providing.
 - `mag_names::AbstractVector{String}` contains strings describing the filters you are providing in `mags`; an example might be `["B","V"]`. These are used when `mag_lim` is finite to determine what filter you want to use to limit the faintest stars you want returned.
 - `limit::Number` gives the total birth stellar mass of the population you want to sample. See the "Notes" section on population masses for more information.
 - `imf::Distributions.Sampleable{Distributions.Univariate, Distributions.Continuous}` is a sampleable continuous univariate distribution implementing a stellar initial mass function with a defined `rand(rng::Random.AbstractRNG, imf)` method to use for sampling masses. All instances of `Distributions.ContinuousUnivariateDistribution` are also valid. Implementations of commonly used IMFs are available in [InitialMassFunctions.jl](https://github.com/cgarling/InitialMassFunctions.jl).

# Keyword Arguments
 - `dist_mod::Number=0` is the distance modulus (see [`StarFormationHistories.distance_modulus`](@ref)) you wish to have added to the returned magnitudes to simulate a population at a particular distance.
 - `rng::Random.AbstractRNG=Random.default_rng()` is the rng instance that will be used to sample the stellar initial masses from `imf`.
 - `mag_lim::Number=Inf` gives the faintest apparent magnitude for stars you want to be returned in the output. Stars fainter than this magnitude will still be sampled and contribute properly to the total mass of the population, but they will not be returned.
 - `mag_lim_name::String="V"` gives the filter name (as contained in `mag_names`) to use when considering if a star is fainter than `mag_lim`. This is unused if `mag_lim` is infinite.
 - `binary_model::StarFormationHistories.AbstractBinaryModel=StarFormationHistories.RandomBinaryPairs(0.3)` is an instance of a model for treating binaries; currently provided options are [`NoBinaries`](@ref) and [`RandomBinaryPairs`](@ref).

# Returns
 - `sampled_masses::Vector{<:Number}`: a vector containing the initial stellar masses of the sampled stars. If you specified a `binary_model` that samples binary or multi-star systems, then these entries are the sum of the initial masses of all the stellar companions. 
 - `sampled_mags::Vector{SVector{N,<:Number}}`: a vector containing `StaticArrays.SVectors` with the multi-band magnitudes of the sampled stars. To get the magnitude of star `i` in band `j`, you would do `sampled_mags[i][j]`. This can be reinterpreted as a 2-dimensional `Matrix` with `reduce(hcat,sampled_mags)`. 

# Notes
## Population Masses
Given a particular isochrone with an initial mass vector `mini_vec`, it will never cover the full range of stellar birth masses because stars that die before present-day are not included in the isochrone. However, these stars *were* born, and so contribute to the total birth mass of the system. There are two ways to properly account for this lost mass when sampling:
 1. Set the upper limit for masses that can be sampled from the `imf` distribution to a physical value for the maximum birth mass of stars (e.g., 100 solar masses). In this case, these stars will be sampled from `imf`, and will contribute their masses to the system, but they will not be returned if their birth mass is greater than `maximum(mini_vec)`. This is typically easiest for the user and only results in ∼15% loss of efficiency for 10 Gyr isochrones.
 2. Set the upper limit for masses that can be sampled from the `imf` distribution to `maximum(mini_vec)` and adjust `limit` to respect the amount of initial stellar mass lost by not sampling higher mass stars. This can be calculated as `new_limit = limit * ( QuadGK.quadgk(x->x*pdf(imf,x), minimum(mini_vec), maximum(mini_vec))[1] / QuadGK.quadgk(x->x*pdf(imf,x), minimum(imf), maximum(imf))[1] )`, with the multiplicative factor being the fraction of the population stellar mass contained in stars with initial masses between `minimum(mini_vec)` and `maximum(mini_vec)`; this factor is the ratio
```math
\\frac{\\int_a^b \\ m \\times \\frac{dN(m)}{dm} \\ dm}{\\int_0^∞ \\ m \\times \\frac{dN(m)}{dm} \\ dm}
```
"""
generate_stars_mass(mini_vec::AbstractVector{<:Number}, mags, args...; kws...) =
    generate_stars_mass(mini_vec, ingest_mags(mini_vec, mags), args...; kws...)
function generate_stars_mass(mini_vec::AbstractVector{<:Number}, mags::AbstractVector{SVector{N,T}}, mag_names::AbstractVector{String}, limit::Number, imf::Sampleable{Univariate,Continuous}; dist_mod::Number=zero(T), rng::AbstractRNG=default_rng(), mag_lim::Number=Inf, mag_lim_name::String="V", binary_model::AbstractBinaryModel=RandomBinaryPairs(0.3)) where {N, T<:Number}
    mags = [ i .+ dist_mod for i in mags ] # Update mags with the provided distance modulus.
    mini_vec, mags = sort_ingested(mini_vec, mags) # Fix non-sorted mini_vec and deduplicate entries.
    # Construct the sampler object for the provided imf; for some distributions, this will return a Distributions.Sampleable for which rand(imf_sampler) is more efficient than rand(imf).
    imf_sampler = sampler(imf) 
    itp = interpolate((mini_vec,), mags, Gridded(Linear()))
    mmin1, mmax = extrema(mini_vec) # Need this to determine validity for mag interpolation.
    mmin2, _ = mass_limits(mini_vec, mags, mag_names, mag_lim, mag_lim_name) # Determine initial mass that corresponds to mag_lim, if provided.
    
    total = zero(eltype(imf))
    mass_vec = Vector{eltype(imf)}(undef,0)
    mag_vec = Vector{eltype(mags)}(undef,0)
    while total < limit
        mass_sample = rand(rng, imf_sampler) # Just sample one star.
        total += mass_sample # Add mass to total.
        # Continue loop if sampled mass is outside of isochrone range.
        if (mass_sample < mmin2) | (mass_sample > mmax)
            continue
        end
        mag_sample = itp(mass_sample) 
        # See if we sample any unresolved binary stars.
        binary_mass, mag_sample = sample_binary(mass_sample, mmin1, mmax, mag_sample, imf_sampler, itp, rng, binary_model)
        total += binary_mass
        push!(mass_vec, mass_sample + binary_mass)
        push!(mag_vec, mag_sample)
    end
    return mass_vec, mag_vec
end

generate_stars_mass2(mini_vec::AbstractVector{<:Number}, mags, args...; kws...) =
    generate_stars_mass2(mini_vec, ingest_mags(mini_vec, mags), args...; kws...)
function generate_stars_mass2(mini_vec::AbstractVector{<:Number}, mags::AbstractVector{SVector{N,T}}, mag_names::AbstractVector{String}, limit::Number, imf::Sampleable{Univariate,Continuous}; dist_mod::Number=zero(T), rng::AbstractRNG=default_rng(), mag_lim::Number=Inf, mag_lim_name::String="V", binary_model::AbstractBinaryModel=RandomBinaryPairs(0.3)) where {N, T<:Number}
    mags = [ i .+ dist_mod for i in mags ] # Update mags with the provided distance modulus.
    mini_vec, mags = sort_ingested(mini_vec, mags) # Fix non-sorted mini_vec and deduplicate entries.
    # Construct the sampler object for the provided imf; for some distributions, this will return a Distributions.Sampleable for which rand(imf_sampler) is more efficient than rand(imf).
    imf_sampler = sampler(imf) 
    itp = interpolate((mini_vec,), mags, Gridded(Linear()))
    mmin1, mmax = extrema(mini_vec) # Need this to determine validity for mag interpolation.
    mmin2, _ = mass_limits(mini_vec, mags, mag_names, mag_lim, mag_lim_name) # Determine initial mass that corresponds to mag_lim, if provided.
    
    total = zero(eltype(imf))
    mass_vec = Vector{SVector{length(binary_model),eltype(imf)}}(undef,0)
    mag_vec = Vector{eltype(mags)}(undef,0)
    while total < limit
        # mass_sample = rand(rng, imf_sampler) # Just sample one star.
        # total += mass_sample # Add mass to total.
        # # Continue loop if sampled mass is outside of isochrone range.
        # if (mass_sample < mmin2) | (mass_sample > mmax)
        #     continue
        # end
        # mag_sample = itp(mass_sample) 
        # # See if we sample any unresolved binary stars.
        # binary_mass, mag_sample = sample_binary(mass_sample, mmin1, mmax, mag_sample, imf_sampler, itp, rng, binary_model)
        # total += binary_mass
        # push!(mass_vec, mass_sample + binary_mass)
        # push!(mag_vec, mag_sample)

        masses = sample_system(imf_sampler, rng, binary_model) # Sample masses of stars in a single system
        total += sum(masses)                           # Add to accumulator
        # if (first(masses) < mmin2) | (first(masses) > mmax)
        if first(masses) < mmin2 # Primary by itself would be fainter than `mag_lim` so continue
            continue             
        end
        lum = zero(eltype(mags))
        for mass in masses
            if (mass > mmin1) & (mass < mmax) # Mass is in valid interpolation range
                lum += L_from_MV.( itp(mass) ) # Think this works for SVectors? not sure
            end
        end
        if sum(lum) > 0 # Possible that first(masses) > mmax and no valid companions either, meaning sum(lum) == 0
            push!(mass_vec, masses)
            push!(mag_vec, MV_from_L.(lum))
        end
    end
    return mass_vec, mag_vec
end

"""
    (sampled_masses, sampled_mags) =  generate_stars_mag(mini_vec::AbstractVector{<:Number}, mags, mag_names::AbstractVector{String}, absmag::Real, absmag_name::String, imf::Distributions.Sampleable{Distributions.Univariate,Distributions.Continuous}; dist_mod::Number=0, rng::AbstractRNG=default_rng(), mag_lim::Number=Inf, mag_lim_name::String="V", binary_model::StarFormationHistories.AbstractBinaryModel=RandomBinaryPairs(0.3))

Generates a mock stellar population from an isochrone defined by the provided initial stellar masses `mini_vec`, absolute magnitudes `mags`, and filter names `mag_names`. The population is sampled to a total absolute magnitude `absmag::Real` (e.g., -7 or -12) in the filter `absmag_name::String` (e.g., "V" or "F606W") which is contained in the provided `mag_names::AbstractVector{String}`. Other arguments are shared with [`generate_stars_mass`](@ref), which contains the main documentation.

# Notes
## Population Magnitudes
Unlike when sampling a population to a fixed initial birth mass, as is implemented in [`generate_stars_mag`](@ref), when generating a population up to a fixed absolute magnitude, only stars that survive to present-day contribute to the flux of the population. If you choose to limit the apparent magnitude of stars returned by passing the `mag_lim` and `mag_lim_name` keyword arguments, stars fainter than your chosen limit will still be sampled and will still contribute their luminosity to the total population, but they will not be contained in the returned output. 
"""
generate_stars_mag(mini_vec::AbstractVector{<:Number}, mags, args...; kws...) =
    generate_stars_mag(mini_vec, ingest_mags(mini_vec, mags), args...; kws...)
function generate_stars_mag(mini_vec::AbstractVector{<:Number}, mags::AbstractVector{SVector{N,T}}, mag_names::AbstractVector{String}, absmag::Number, absmag_name::String, imf::Sampleable{Univariate,Continuous}; dist_mod::Number=0, rng::AbstractRNG=default_rng(), mag_lim::Number=Inf, mag_lim_name::String="V", binary_model::AbstractBinaryModel=RandomBinaryPairs(0.3)) where {N, T<:Number}
    mags = [ i .+ dist_mod for i in mags ] # Update mags with the provided distance modulus.
    mini_vec, mags = sort_ingested(mini_vec, mags) # Fix non-sorted mini_vec and deduplicate entries.
    idxlim = findfirst(x->x==absmag_name, mag_names) # Get the index into `mags` and `mag_names` that equals `limit_name`.
    idxlim == nothing && throw(ArgumentError("Provided `absmag_name` is not contained in provided `mag_names` array.")) # Throw error if absmag_name not in mag_names.
    limit = L_from_MV(absmag) # Convert the provided `limit` from magnitudes into luminosity.
    # Construct the sampler object for the provided imf; for some distributions, this will return a
    # Distributions.Sampleable for which rand(imf_sampler) is more efficient than rand(imf).
    imf_sampler = sampler(imf)
    itp = interpolate((mini_vec,), mags, Gridded(Linear()))
    mmin1, mmax = extrema(mini_vec) # Need this to determine validity for mag interpolation.
    mmin2, _ = mass_limits(mini_vec, mags, mag_names, mag_lim, mag_lim_name) # Determine initial mass that corresponds to mag_lim, if provided.
    
    total = zero(T)
    mass_vec = Vector{eltype(imf)}(undef,0)
    mag_vec = Vector{eltype(mags)}(undef,0)
    while total < limit
        mass_sample = rand(rng, imf_sampler)  # Just sample one star.
        # Continue loop if sampled mass is outside of isochrone range.
        if (mass_sample < mmin1) | (mass_sample > mmax)
            continue
        end
        mag_sample = itp(mass_sample) # Interpolate the sampled mass.
        # See if we sample any binary stars.
        binary_mass, mag_sample = sample_binary(mass_sample, mmin1, mmax, mag_sample, imf_sampler, itp, rng, binary_model)
        mass_sample += binary_mass
        total += L_from_MV(mag_sample[idxlim] - dist_mod) # Add luminosity to total, subtracting the distance modulus.
        # Only push to the output vectors if the sampled mass is in the valid range. 
        if mmin2 <= mass_sample 
            push!(mass_vec, mass_sample)
            push!(mag_vec, mag_sample)
        end
    end
    # println(MV_from_L(total)) # Print the sampled absolute magnitude. 
    return mass_vec, mag_vec
end

generate_stars_mag2(mini_vec::AbstractVector{<:Number}, mags, args...; kws...) =
    generate_stars_mag(mini_vec, ingest_mags(mini_vec, mags), args...; kws...)
function generate_stars_mag2(mini_vec::AbstractVector{<:Number}, mags::AbstractVector{SVector{N,T}}, mag_names::AbstractVector{String}, absmag::Number, absmag_name::String, imf::Sampleable{Univariate,Continuous}; dist_mod::Number=0, rng::AbstractRNG=default_rng(), mag_lim::Number=Inf, mag_lim_name::String="V", binary_model::AbstractBinaryModel=RandomBinaryPairs(0.3)) where {N, T<:Number}
    # Setup and input ingestion
    mags = [ i .+ dist_mod for i in mags ]         # Update mags with the provided distance modulus.
    mini_vec, mags = sort_ingested(mini_vec, mags) # Fix non-sorted mini_vec and deduplicate entries.
    limit = L_from_MV(absmag) # Convert the provided `limit` from magnitudes into luminosity.
    # Construct the sampler object for the provided imf; for some distributions, this will return a
    # Distributions.Sampleable for which rand(imf_sampler) is more efficient than rand(imf).
    imf_sampler = sampler(imf)
    itp = interpolate((mini_vec,), mags, Gridded(Linear()))
    mmin1, mmax = extrema(mini_vec) # Need this to determine validity for mag interpolation.

    # Find indices into `mags` and `mag_names` that correspdong to `limit_name` and `mag_lim_name`.
    idxlim = findfirst(x->x==absmag_name, mag_names) 
    # Throw error if absmag_name not in mag_names.
    if idxlim == nothing
        throw(ArgumentError("Provided `absmag_name` is not contained in provided `mag_names` array."))
    end
    sourceidx = findfirst(x->x==mag_lim_name, mag_names) # Get the index into `mags` and `mag_names` that equals `mag_lim_name`
    if (!isinfinite(mag_lim) & (sourceidx == nothing))
        throw(ArgumentError("Provided `mag_lim_name` is not contained in provided `mag_names` array."))
    end

    # Set up accumulators and loop
    total = zero(T)
    mass_vec = Vector{SVector{length(binary_model),eltype(imf)}}(undef,0)
    mag_vec = Vector{eltype(mags)}(undef,0)
    while total < limit
        masses = sample_system(imf_sampler, rng, binary_model) # Sample masses of stars in a single system
        lum = zero(eltype(mags)) # Accumulator to hold per-filter luminosities
        for mass in masses
            # Continue loop if sampled mass is outside of valid isochrone range
            if (mass < mmin1) | (mass > mmax)
                continue
            end
            lum += L_from_MV.( itp(mass) )
        end
        total += lum[idxlim] # Add the luminosity in the correct filter to the `total` accumulator
        if (sum(lum) > 0) & (MV_from_L(lum[sourceidx]) > mag_lim) # If the source is bright enough, add it to output
            push!(mass_vec, masses)
            push!(mag_vec, MV_from_L.(lum))
        end
    end
    return mass_vec, mag_vec
end

############################################################################
#### Functions to generate composite mock galaxy catalogs from multiple SSPs
"""
    (sampled_masses, sampled_mags) = generate_stars_mass_composite(mini_vec::AbstractVector{<:AbstractVector{<:Number}}, mags::AbstractVector, mag_names::AbstractVector{String}, limit::Number, massfrac::AbstractVector{<:Number}, imf::Sampleable{Univariate,Continuous}; kws...)

Generates a random sample of stars with a complex star formation history using multiple isochrones. Very similar to [`generate_stars_mass`](@ref) except the isochrone-related arguments `mini_vec` and `mags` should now be vectors of vectors containing the relevant data for the full set of isochrones to be considered. The total birth stellar mass of the sampled population is given by `limit`. The proportion of this mass allotted to each of the individual isochrones is given by the entries of the `massfrac` vector. This basically just proportions `limit` according to `massfrac` and calls [`generate_stars_mass`](@ref) for each of the individual stellar populations; as such it is set up to multi-thread across the multiple stellar populations. 

# Arguments
 - `mini_vec::AbstractVector{<:AbstractVector{<:Number}}` contains the initial masses (in solar masses) for the stars in each isochrone; the internal vectors must be mutable as we will call `Interpolations.deduplicate_knots!` on each. The length of `mini_vec` should be equal to the number of isochrones. 
 - `mags` contains the absolute magnitudes from the isochrones in the desired filters corresponding to the same stars as provided in `mini_vec`. The length of `mags` should be equal to the number of isochrones. The individual elements of `mags` are each internally interpreted and converted into a standard format by [`StarFormationHistories.ingest_mags`](@ref). The valid formats for the individual elements of `mags` are:
    - `AbstractVector{AbstractVector{<:Number}}`, in which case the length of the vector `length(mags[i])` can either be equal to `length(mini_vec[i])`, in which case the length of the inner vectors must all be equal to the number of filters you are providing, or the length of the outer vector can be equal to the number of filters you are providing, and the length of the inner vectors must all be equal to `length(mini_vec[i])`; this is the more common use-case.
    - `AbstractMatrix{<:Number}`, in which case `mags[i]` must be 2-dimensional. Valid shapes are `size(mags[i]) == (length(mini_vec[i]), nfilters)` or `size(mags[i]) == (nfilters, length(mini_vec[i]))`, with `nfilters` being the number of filters you are providing.
 - `mag_names::AbstractVector{String}` contains strings describing the filters you are providing in `mags`; an example might be `["B","V"]`. These are used when `mag_lim` is finite to determine what filter you want to use to limit the faintest stars you want returned. These are assumed to be the same for all isochrones.
 - `limit::Number` gives the total birth stellar mass of the population you want to sample. 
 - `massfrac::AbstractVector{<:Number}` is vector giving the relative fraction of mass allotted to each individual stellar population; length must be equal to the length of `mini_vec` and `mags`. 
 - `imf::Distributions.Sampleable{Distributions.Univariate, Distributions.Continuous}` is a sampleable continuous univariate distribution implementing a stellar initial mass function with a defined `rand(rng::Random.AbstractRNG, imf)` method to use for sampling masses. All instances of `Distributions.ContinuousUnivariateDistribution` are also valid. Implementations of commonly used IMFs are available in [InitialMassFunctions.jl](https://github.com/cgarling/InitialMassFunctions.jl).

# Keyword Arguments
All keyword arguments `kws...` are passed to [`generate_stars_mass`](@ref); you should refer to that method's documentation for more information. 

# Returns
 - `sampled_masses::Vector{Vector{<:Number}}` is a vector of vectors containing the initial stellar masses of the sampled stars. The outer vectors are separated by the isochrone the stars were generated from; i.e., all of `sampled_masses[1]` were sampled from `mini_vec[1]` and so on. These can be concatenated into a single vector with `reduce(vcat,sampled_masses)`. If you specified a `binary_model` that samples binary or multi-star systems, then these entries are the sum of the initial masses of all the stellar companions. 
 - `sampled_mags::Vector{Vector{SVector{N,<:Number}}}` is a vector of vectors containing `StaticArrays.SVectors` with the multi-band magnitudes of the sampled stars. The outer vectors are separated by the isochrone the stars were generated from; i.e. all of `sampled_mags[1]` were sampled from `mags[1]` and so on. To get the magnitude of star `i` in band `j` sampled from isochrone `k`, you would do `sampled_mags[k][i][j]`. This can be concatenated into a `Vector{SVector}` with `reduce(vcat,sampled_mags)` and a 2-D `Matrix` with `reduce(hcat,reduce(vcat,sampled_mags))`. 
"""
function generate_stars_mass_composite(mini_vec::AbstractVector{T}, mags::AbstractVector, mag_names::AbstractVector{String}, limit::Number, massfrac::AbstractVector{<:Number}, imf::Sampleable{Univariate,Continuous}; kws...) where T <: AbstractVector{<:Number} 
    !(axes(mini_vec,1) == axes(mags,1) == axes(massfrac,1)) && throw(ArgumentError("The arguments `mini_vec`, `mags`, and `massfrac` to `generate_stars_mass_composite` must all have equal length and identical indexing."))
    ncomposite = length(mini_vec) # Number of stellar populations provided.
    massfrac = massfrac ./ sum(massfrac) # Ensure massfrac is normalized to sum to 1.
    # Allocate output vectors.
    massvec = [ Vector{eltype(imf)}(undef,0) for i in 1:ncomposite ]
    # Need to ingest here so we know what type of SVector we're going to be putting into mag_vec. 
    mags = [ ingest_mags(mini_vec[i], mags[i]) for i in eachindex( mini_vec, mags ) ]
    mag_vec = [ Vector{eltype(i)}(undef,0) for i in mags ]
    # Loop over each component, calling generate_stars_mass. Threading works with good scaling.
    Threads.@threads for i in eachindex(mini_vec, mags, massfrac)
        result = generate_stars_mass(mini_vec[i], mags[i], mag_names, limit * massfrac[i], imf; kws...)
        massvec[i] = result[1]
        mag_vec[i] = result[2]
    end
    return massvec, mag_vec
end

function generate_stars_mass_composite2(mini_vec::AbstractVector{T}, mags::AbstractVector, mag_names::AbstractVector{String}, limit::Number, massfrac::AbstractVector{<:Number}, imf::Sampleable{Univariate,Continuous}; binary_model::AbstractBinaryModel=RandomBinaryPairs(0.3), kws...) where T <: AbstractVector{<:Number} 
    if !(axes(mini_vec,1) == axes(mags,1) == axes(massfrac,1))
        throw(ArgumentError("The arguments `mini_vec`, `mags`, and `massfrac` to `generate_stars_mass_composite` must all have equal length and identical indexing."))
    end
    ncomposite = length(mini_vec) # Number of stellar populations provided.
    massfrac = massfrac ./ sum(massfrac) # Ensure massfrac is normalized to sum to 1.
    # Allocate output vectors.
    massvec = [ Vector{SVector{length(binary_model),eltype(imf)}}(undef,0) for i in 1:ncomposite ]
    # Need to ingest here so we know what type of SVector we're going to be putting into mag_vec. 
    mags = [ ingest_mags(mini_vec[i], mags[i]) for i in eachindex( mini_vec, mags ) ]
    mag_vec = [ Vector{eltype(i)}(undef,0) for i in mags ]
    # Loop over each component, calling generate_stars_mass. Threading works with good scaling.
    Threads.@threads for i in eachindex(mini_vec, mags, massfrac)
        result = generate_stars_mass2(mini_vec[i], mags[i], mag_names, limit * massfrac[i], imf;
                                      binary_model = binary_model, kws...)
        massvec[i] = result[1]
        mag_vec[i] = result[2]
    end
    return massvec, mag_vec
end

"""
    (sampled_masses, sampled_mags) = generate_stars_mag_composite(mini_vec::AbstractVector{<:AbstractVector{<:Number}}, mags::AbstractVector, mag_names::AbstractVector{String}, absmag::Number, absmag_name::String, fracs::AbstractVector{<:Number}, imf::Sampleable{Univariate,Continuous}; frac_type::String="lum", kws...)

Generates a random sample of stars with a complex star formation history using multiple isochrones. Very similar to [`generate_stars_mag`](@ref) except the isochrone-related arguments `mini_vec` and `mags` should now be vectors of vectors containing the relevant data for the full set of isochrones to be considered. The total absolute magnitude of the sampled population is given by `absmag`. The proportion of the luminosity allotted to each of the individual isochrones is given by the entries of the `frac` vector. This basically just proportions the luminosity according to `frac` and calls [`generate_stars_mass`](@ref) for each of the individual stellar populations; as such it is set up to multi-thread across the multiple stellar populations. 

# Arguments
 - `mini_vec::AbstractVector{<:AbstractVector{<:Number}}` contains the initial masses (in solar masses) for the stars in each isochrone; the internal vectors must be mutable as we will call `Interpolations.deduplicate_knots!` on each. The length of `mini_vec` should be equal to the number of isochrones. 
 - `mags` contains the absolute magnitudes from the isochrones in the desired filters corresponding to the same stars as provided in `mini_vec`. The length of `mags` should be equal to the number of isochrones. The individual elements of `mags` are each internally interpreted and converted into a standard format by [`StarFormationHistories.ingest_mags`](@ref). The valid formats for the individual elements of `mags` are:
    - `AbstractVector{AbstractVector{<:Number}}`, in which case the length of the vector `length(mags[i])` can either be equal to `length(mini_vec[i])`, in which case the length of the inner vectors must all be equal to the number of filters you are providing, or the length of the outer vector can be equal to the number of filters you are providing, and the length of the inner vectors must all be equal to `length(mini_vec[i])`; this is the more common use-case.
    - `AbstractMatrix{<:Number}`, in which case `mags[i]` must be 2-dimensional. Valid shapes are `size(mags[i]) == (length(mini_vec[i]), nfilters)` or `size(mags[i]) == (nfilters, length(mini_vec[i]))`, with `nfilters` being the number of filters you are providing.
 - `mag_names::AbstractVector{String}` contains strings describing the filters you are providing in `mags`; an example might be `["B","V"]`. These are used when `mag_lim` is finite to determine what filter you want to use to limit the faintest stars you want returned. These are assumed to be the same for all isochrones.
 - `absmag::Number` gives the total absolute magnitude of the complex population to be sampled. 
 - `fracs::AbstractVector{<:Number}` is a vector giving the relative fraction of luminosity or mass (determined by the `frac_type` keyword argument) allotted to each individual stellar population; length must be equal to the length of `mini_vec` and `mags`. 
 - `imf::Distributions.Sampleable{Distributions.Univariate, Distributions.Continuous}` is a sampleable continuous univariate distribution implementing a stellar initial mass function with a defined `rand(rng::Random.AbstractRNG, imf)` method to use for sampling masses. All instances of `Distributions.ContinuousUnivariateDistribution` are also valid. Implementations of commonly used IMFs are available in [InitialMassFunctions.jl](https://github.com/cgarling/InitialMassFunctions.jl).

# Keyword Arguments
 - `frac_type::String` either "lum", in which case `fracs` is assumed to contain the relative luminosity fractions for each individual isochrone, or "mass", in which case it is assumed that `fracs` contains mass fractions ("mass" is not yet implemented). 
All other keyword arguments `kws...` are passed to [`generate_stars_mag`](@ref); you should refer to that method's documentation for more information. 

# Returns
 - `sampled_masses::Vector{Vector{<:Number}}` is a vector of vectors containing the initial stellar masses of the sampled stars. The outer vectors are separated by the isochrone the stars were generated from; i.e., all of `sampled_masses[1]` were sampled from `mini_vec[1]` and so on. These can be concatenated into a single vector with `reduce(vcat,sampled_masses)`. If you specified a `binary_model` that samples binary or multi-star systems, then these entries are the sum of the initial masses of all the stellar companions. 
 - `sampled_mags::Vector{Vector{SVector{N,<:Number}}}` is a vector of vectors containing `StaticArrays.SVectors` with the multi-band magnitudes of the sampled stars. The outer vectors are separated by the isochrone the stars were generated from; i.e. all of `sampled_mags[1]` were sampled from `mags[1]` and so on. To get the magnitude of star `i` in band `j` sampled from isochrone `k`, you would do `sampled_mags[k][i][j]`. This can be concatenated into a `Vector{SVector}` with `reduce(vcat,sampled_mags)` and a 2-D `Matrix` with `reduce(hcat,reduce(vcat,sampled_mags))`. 
"""
function generate_stars_mag_composite(mini_vec::AbstractVector{T}, mags::AbstractVector, mag_names::AbstractVector{String}, absmag::Number, absmag_name::String, fracs::AbstractVector{<:Number}, imf::Sampleable{Univariate,Continuous}; frac_type::String="lum", kws...) where T <: AbstractVector{<:Number}
    !(axes(mini_vec,1) == axes(mags,1) == axes(fracs,1)) && throw(ArgumentError("The arguments `mini_vec`, `mags`, and `fracs` to `generate_stars_mag_composite` must all have equal length and identical indexing."))
    ncomposite = length(mini_vec) # Number of stellar populations provided.
    fracs = fracs ./ sum(fracs) # Ensure fracs is normalized to sum to 1.
    # Interpret whether user requests `fracs` represent luminosity or mass fractions.
    if frac_type == "lum"
        limit = L_from_MV(absmag)   # Convert the provided `limit` from magnitudes into luminosity.
        fracs = MV_from_L.( fracs .* limit )
    elseif frac_type == "mass"
        throw(ArgumentError("`frac_type == mass` not yet implemented."))
    else
        throw(ArgumentError("Supported `frac_type` arguments for generate_stars_mag_composite are \"lum\" or \"mass\"."))
    end
    # Allocate output vectors.
    massvec = [ Vector{eltype(imf)}(undef,0) for i in 1:ncomposite ]
    # Need to ingest here so we know what type of SVector we're going to be putting into mag_vec. 
    mags = [ ingest_mags(mini_vec[i], mags[i]) for i in eachindex( mini_vec, mags ) ]
    mag_vec = [ Vector{eltype(i)}(undef,0) for i in mags ]
    # Loop over each component, calling generate_stars_mass. Threading works with good scaling.
    Threads.@threads for i in eachindex(mini_vec, mags, fracs)
        result = generate_stars_mag(mini_vec[i], mags[i], mag_names, fracs[i], absmag_name, imf; kws...)
        massvec[i] = result[1]
        mag_vec[i] = result[2]
    end
    return massvec, mag_vec
end

###############################################
#### Functions for modelling observational effects
"""
    new_mags = model_cmd(mags::AbstractVector{<:AbstractVector{<:Number}}, errfuncs, completefuncs; rng::Random.AbstractRNG=Random.default_rng())

Simple method for modelling photometric error and incompleteness to "mock observe" a pure catalog of stellar photometry, such as those produced by [`generate_stars_mass`](@ref) and [`generate_stars_mag`](@ref). This method assumes Gaussian photometric errors and that the photometric error and completeness functions are separable by filter. 

# Arguments
 - `mags::AbstractVector{<:AbstractVector{<:Number}}`: a vector of vectors giving the magnitudes of each star to be modelled. The first index is the per-star index and the second index is the per-filter index (so `mags[10][2]` would give the magnitude of the tenth star in the second filter). This is the same format as the magnitudes returned by [`generate_stars_mass`](@ref) and [`generate_stars_mag`](@ref); to use output from the composite versions, you must first `reduce(vcat,mags)` before passing to this function.
 - `errfuncs`: an iterable (typically a vector or tuple) of callables (typically functions or interpolators) with length equal to the number of filters contained in the elements of `mags`. This iterable must contain callables that, when called with the associated magnitudes from `mags`, will return the expected 1-σ photometric error at that magnitude. The organization is such that the photometric error for star `i` in band `j` is `σ_ij = errfuncs[j](mags[i][j])`. 
 - `completefuncs`: an iterable (typically a vector or tuple) of callables (typically functions or interpolators) with length equal to the number of filters contained in the elements of `mags`. This iterable must contain callables that, when called with the associated magnitudes from `mags`, will return the probability that a star with that magnitude in that band will be found in your color-magnitude diagram (this should include the original detection probability and any post-detection quality, morphology, or other cuts). The organization is such that the detection probability for star `i` in band `j` is `c_ij = completefuncs[j](mags[i][j])`.

# Keyword Arguments
 - `rng::AbstractRNG=Random.default_rng()`: The object to use for random number generation.

# Returns
 - `new_mags`: an object similar to `mags` (i.e., a `Vector{Vector{<:Number}}`, `Vector{SVector{N,<:Number}}`, etc.) containing the magnitudes of the mock-observed stars. This will be shorter than the provided `mags` vector as we are modelling photometric incompleteness, and the magnitudes will also have random photometric errors added to them. This can be reinterpreted as a 2-dimensional `Matrix` with `reduce(hcat,new_mags)`. 

# Notes
 - This is a simple implementation that seeks to show a simple example of how one can post-process catalogs of "pure" stars from methods like [`generate_stars_mass`](@ref) and [`generate_stars_mag`](@ref) to include observational effects. This method assumes Gaussian photometric errors, which may not, in general, be accurate. It also assumes that the total detection probability can be modelled as the product of the single-filter detection probabilities as computed by `completefuncs` (i.e., that the completeness functions are separable across filters). This can be a reasonable assumption when you have separate photometric catalogs derived for each filter and you only collate them afterwards, but it is generally not a good assumption for detection algorithms that operate on simultaneously on multi-band photometry -- the completeness functions for these types of algorithms are generally not separable.
"""
function model_cmd(mags::AbstractVector{T}, errfuncs, completefuncs; rng::AbstractRNG=default_rng()) where T <: AbstractVector{<:Number}
    nstars = length(mags)
    nfilters = length(first(mags))
    !(axes(first(mags),1) == axes(errfuncs,1) == axes(completefuncs,1)) && throw(ArgumentError("Arguments to `StarFormationHistories.model_cmd` must satisfy `axes(first(mags),1) == axes(errfuncs,1) == axes(completefuncs,1)`."))
    randsamp = rand(rng, nstars) # Draw nstars random uniform variates for completeness testing.
    completeness = ones(axes(mags,1))
    # Estimate the overall completeness as the product of the single-band completeness values.
    for i in eachindex(mags)
        for j in eachindex(completefuncs)
            completeness[i] *= completefuncs[j](mags[i][j])
        end
    end
    # Pick out the entries where the random number is less than the product of the single-band completeness values 
    good = findall( map(<=, randsamp, completeness) ) # findall( randsamp .<= completeness )
    ret_mags = mags[good] # Get the good mags to be returned.
    for i in eachindex(ret_mags)
        for j in eachindex(errfuncs)
            ret_mags[i][j] += ( randn(rng) * errfuncs[j](ret_mags[i][j]) ) # Add error. 
        end
    end
    return ret_mags
end
# This is slower than the above implementation but I don't care to optimize it at the moment.
function model_cmd(mags::AbstractVector{SVector{N,T}}, errfuncs, completefuncs; rng::AbstractRNG=default_rng()) where {N, T <: Number}
    nstars = length(mags)
    nfilters = length(first(mags))
    !(axes(first(mags),1) == axes(errfuncs,1) == axes(completefuncs,1)) && throw(ArgumentError("Arguments to `StarFormationHistories.model_cmd` must satisfy `axes(first(mags),1) == axes(errfuncs,1) == axes(completefuncs,1)`."))
    randsamp = rand(rng, nstars) # Draw nstars random uniform variates for completeness testing.
    completeness = ones(axes(mags,1))
    # Estimate the overall completeness as the product of the single-band completeness values.
    for i in eachindex(mags)
        for j in eachindex(completefuncs)
            completeness[i] *= completefuncs[j](mags[i][j])
        end
    end
    # Pick out the entries where the random number is less than the product of the single-band completeness values 
    good = findall( map(<=, randsamp, completeness) ) # findall( randsamp .<= completeness )
    good_mags = mags[good] # Get the good mags to be returned.
    ret_mags = similar(good_mags) 
    for i in eachindex(good_mags)
        err_scale = sacollect(SVector{N,T}, errfuncs[j](good_mags[i][j]) for j in eachindex(errfuncs))
        ret_mags[i] = good_mags[i] .+ ( randn(rng,SVector{N,T}) .* err_scale)
    end
    return ret_mags
end
# In Julia 1.9 we should just be able to do stack(v). 
# eltocols(v::Vector{SVector{dim, T}}) where {dim, T} = reshape(reinterpret(T, v), dim, :)
# eltorows(v::Vector{SVector{dim, T}}) where {dim, T} = reshape(reinterpret(T, v), :, dim)
# eltocols(v::Vector{Vector{SVector{dim, T}}}) where {dim, T} = mapreduce(eltocols, hcat, v)
# eltocols(v::Vector{Vector{SVector{dim, T}}}) where {dim, T} = reduce(hcat, eltocols(i) for i in v)
# eltocols(v::Vector{Vector{SVector{dim, T}}}) where {dim, T} = reduce(hcat, eltocols.(v))
# eltocols(v::Vector{Vector{SVector{dim, T}}}) where {dim, T} = hcat(eltocols.(v)...)
