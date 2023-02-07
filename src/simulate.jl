#### Distance Utilities
"""
    distance_modulus(distance)
Finds distance modulus for distance in parsecs. 
"""
distance_modulus(distance) =  5 * log10(distance/10)
"""
    distance_modulus_to_distance(dist_mod)
Converts distance modulus to distance in parsecs.
"""
distance_modulus_to_distance(dist_mod) = exp10(dist_mod/5 + 1)
"""
    arcsec_to_pc(arcsec,dist_mod)
Converts angle in arcseconds to physical separation based on distance modulus; near-field only.
"""
arcsec_to_pc(arcsec, dist_mod) = exp10(dist_mod/5 + 1) * atan( deg2rad(arcsec/3600) )
"""
    pc_to_arcsec(pc, dist_mod)
Inverse of `arcsec_to_pc`.
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
    Y_from_Z(Z)
Calculates the helium mass fraction (Y) for a star given its metal mass fraction (Z) using the approximation `Y = 0.2485 + 1.78Z` as assumed for [PARSEC](http://stev.oapd.inaf.it/cgi-bin/cmd_3.7) isochrones. 
"""
Y_from_Z(Z) = 0.2485 + 1.78Z
"""
    X_from_Z(Z)
Calculates the hydrogen mass fraction (X) for a star given its metal mass fraction (Z) via `X = 1 - (Z + Y)`, with the helium mass fraction `Y` approximated via [`SFH.Y_from_Z`](@ref). 
"""
X_from_Z(Z) = 1 - (Y_from_Z(Z) + Z)
"""
    MH_from_Z(Z, solZ=0.0152)
Calculates [M/H] = log(Z/X) - log(Z/X)⊙. Given the provided solar metal mass fraction `solZ`, it calculates the hydrogen mass fraction X for both the Sun and the provided `Z` with [`SFH.X_from_Z`](@ref). This is based on an approximation and may not be suitable for precision calculations.
"""
MH_from_Z(Z, solZ=0.0152) = log10(Z / X_from_Z(Z)) - log10(solZ / X_from_Z(solZ))
# PARSEC says that the solar Z is 0.0152 and Z/X = 0.0207, but they don't quite agree
# when assuming their provided Y=0.2485+1.78Z. We'll adopt their solZ here, but this
# should probably not be used for precision calculations.

################################################
#### Interpret arguments for generate_mock_stars
"""
    new_mags = ingest_mags(mini_vec::AbstractVector, mags::AbstractVector{T}) where {S <: Number, T <: AbstractVector{S}}
    new_mags = ingest_mags(mini_vec::AbstractVector, mags::AbstractMatrix{S}) where S <: Number

# Returns
 - `new_mags::Base.ReinterpretArray{SVector}`: a `length(mini_vec)` vector of SVectors containing the same data as `mags` but formatted for input to `Interpolations.interpolate`.
"""
function ingest_mags(mini_vec::AbstractVector, mags::AbstractMatrix{S}) where S <: Number
    if ndims(mags) != 2 # Check dimensionality of mags argument
        throw(ArgumentError("`generate_stars...` received a `mags::AbstractMatrix{<:Real}` with `ndims(mags) != 2`; when providing a `mags::AbstractMatrix{<:Real}`, it must always be 2-dimensional."))
    end
    nstars = length(mini_vec)
    shape = size(mags)
    if shape[1] == nstars
        return copy(reinterpret(SVector{shape[2],S}, vec(permutedims(mags))))
    elseif shape[2] == nstars
        return copy(reinterpret(SVector{shape[1],S}, vec(mags)))
    else
        throw(ArgumentError("`generate_stars...` received a misshapen `mags` argument. When providing a `mags::AbstractMatrix{<:Real}`, then it should be 2-dimensional and have size of (N,M) or (M,N), where N is the number of elements in `mini_vec`, and M is the number of filters represented in the `mags` argument."))
    end
end
function ingest_mags(mini_vec::AbstractVector, mags::AbstractVector{T}) where {S <: Number, T <: AbstractVector{S}}
    # Commonly `mini_vec` will be a vector of length `N`, but `mags` will be a length `M` vector of length `N` vectors. E.g., if length(mini_vec) == 100, and we have two filters, then `mags` will be a vector of 2 vectors, each with length 100. The interpolation routine requires `mags` to be a vector of 100 vectors, each with length 2.
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

function mass_limits(mini_vec::AbstractVector{<:Number}, mags::AbstractVector{T},
                     mag_names::AbstractVector{String}, mag_lim::Number,
                     mag_lim_name::String) where T <: AbstractVector{<:Number}
    @assert axes(mini_vec) == axes(mags) 
    mmin, mmax = extrema(mini_vec) 
    # Update mmin respecing `mag_lim`, if provided.
    if isfinite(mag_lim)
        idxmag = findfirst(x->x==mag_lim_name, mag_names) # Find the index into mag_lim_names where == mag_lim_name.
        idxmag == nothing && throw(ArgumentError("Provided `mag_lim_name` is not contained in provided `mag_names` array."))
        if mag_lim < mags[findfirst(x->x==mmin, mini_vec)][idxmag]
            tmp_mags = [i[idxmag] for i in mags]
            mag_lim < minimum(tmp_mags) && throw(DomainError(mag_lim, "The provided `mag_lim` is brighter than all the stars in the input `mags` array assuming the input distance modulus `dist_mod`. Revise your arguments."))
            # Solve for stellar initial mass where mag == mag_lim by constructing interpolator and root-finding.
            itp = interpolate((mini_vec,), tmp_mags, Gridded(Linear()))
            mmin = find_zero(x -> itp(x) - mag_lim, (mmin, mmax))
        end
    end
    return mmin, mmax
end

###############################################
#### Types and methods for non-interacting binary calculations
""" `SFH.AbstractBinaryModel` is the abstract supertype for all types that are used to evaluate multi-star systems in the SFH package. All concrete subtypes must implement the [`SFH.sample_binary!`](@ref) method. """
abstract type AbstractBinaryModel end
Base.Broadcast.broadcastable(m::AbstractBinaryModel) = Ref(m)
""" The `NoBinaries` type indicates that no binaries of any kind should be created. """
struct NoBinaries <: AbstractBinaryModel end
""" The `Binaries` type takes one argument `fraction` that denotes the number fraction of binaries (e.g., 0.3 for 30% binary fraction). This model will ONLY generate up to one additional star -- it will not generate any 3+ star systems. This model typically incurs a 10--20% speed reduction relative to `NoBinaries`. """
struct Binaries{T <: Real} <: AbstractBinaryModel
    fraction::T
end

"""
    binary_mass = sample_binary!(mass, mmin, mmax, mags, imf, itp, rng::AbstractRNG, binarymodel::T) where T <: SFH.AbstractBinaryModel

Mutates input `mags` array that represents a single star's per-filter magnitudes by adding luminosity from unresolved binaries (if any) and returns the sampled mass of the binary star(s). Returns `zero(mass)` if no binary is sampled, but will return a non-zero mass even if a binary is outside the valid initial mass range `(mmin, mmax)`. Other details may vary based on the input `binarymodel`. 
"""
@inline sample_binary!(mass, mmin, mmax, mags, imf, itp, rng::AbstractRNG, binarymodel::NoBinaries) = zero(mass)
@inline function sample_binary!(mass, mmin, mmax, mags, imf, itp, rng::AbstractRNG, binarymodel::Binaries)
    frac = binarymodel.fraction
    r = rand(rng) # Random uniform number
    if r <= frac  # Generate a binary star
        mass_new = rand(rng, imf)
        if (mass_new < mmin) | (mass_new > mmax) # Sampled mass is outside of valid range
            return mass_new # We'll return it so it can be incremented to the mass tracker
        end
        mags_new = itp(mass_new)
        for i in eachindex(mags)
            L = L_from_MV(mags[i])
            L += L_from_MV(mags_new[i])
            mags[i] = MV_from_L(L) # Mutate the existing mags array
        end
        return mass_new
    else
        return zero(mass)
    end
end
"""
    binary_mass, new_mags = sample_binary(mass, mmin, mmax, mags, imf, itp, rng::AbstractRNG, binarymodel::SFH.AbstractBinaryModel)

Simulates the effects of unresolved binaries on stellar photometry without mutation. Implementation depends on the choice of `binarymodel`.

# Arguments
 - `mass`; the initial mass of the single star
 - `mmin`; minimum mass to consider for stellar companions
 - `mmax`; maximum mass to consider for stellar companions
 - `mags`; a vector-like object giving the magnitudes of the single star in each filter
 - `imf`; an object implementing `rand(imf)` to draw a random single-star mass
 - `itp`; a callable object that returns the magnitudes of a star with mass `m` when called as `itp(m)`
 - `rng::AbstractRNG`; the random number generator to use when sampling stars
 - `binarymodel::SFH.AbstractBinaryModel`; an instance of a binary model that determines which implementation will be used. 

# Returns
 - `binary_mass`; the total mass of the additional stellar companions
 - `new_mags`; the effective magnitude of the multiple stellar system 
"""
@inline sample_binary(mass, mmin, mmax, mags, imf, itp, rng::AbstractRNG, binarymodel::NoBinaries) = zero(mass), mags
@inline function sample_binary(mass, mmin, mmax, mags, imf, itp, rng::AbstractRNG, binarymodel::Binaries)
    frac = binarymodel.fraction
    r = rand(rng) # Random uniform number
    if r <= frac  # Generate a binary star
        mass_new = rand(rng, imf)
        if (mass_new < mmin) | (mass_new > mmax) # Sampled mass is outside of valid range
            return mass_new, mags 
        end
        mags_new = itp(mass_new)
        # for i in eachindex(mags)
        #     L = L_from_MV(mags[i])
        #     L += L_from_MV(mags_new[i])
        #     mags[i] = MV_from_L(L) # Mutate the existing mags array
        # end
        # return mass_new
        result = MV_from_L.( L_from_MV.(mags) .+ L_from_MV.(mags_new) )
        return mass_new, result
    else
        return zero(mass), mags
    end
end


###############################################
#### Functions to generate mock galaxy catalogs from SSPs

"""
    (sampled_masses, sampled_mags) = generate_stars_mass(mini_vec::Vector{<:Real}, mags, mag_names::Vector{String}, limit::Real, imf::Distributions.UnivariateDistribution{Distributions.Continuous}; dist_mod::Real=0, rng::Random.AbstractRNG=default_rng(), mag_lim::Real=Inf, mag_lim_name::String="V", binary_model::SFH.AbstractBinaryModel=Binaries(0.3))

# Arguments
 - `mini_vec::Vector{<:Real}` contains the initial masses (in solar masses) for the stars in the isochrone.
 - `mags` contains the absolute magnitudes from the isochrone in the desired filters corresponding to the same stars as provided in `mini_vec`. `mags` is internally interpreted and converted into a standard format by [`SFH.ingest_mags`](@ref). Valid inputs are:
    - `mags::Vector{Vector{<:Real}}`, in which case the length of the outer vector `length(mags)` can either be equal to `length(mini_vec)`, in which case the length of the inner vectors must all be equal to the number of filters you are providing, or the length of the outer vector can be equal to the number of filters you are providing, and the length of the inner vectors must all be equal to `length(mini_vec)`; this is the more common use-case.
    - `mags::Matrix{<:Real}`, in which case `mags` must be 2-dimensional. Valid shapes are `size(mags) == (length(mini_vec), nfilters)` or `size(mags) == (nfilters, length(mini_vec))`, with `nfilters` being the number of filters you are providing.
 - `mag_names::Vector{String}` contains strings describing the filters you are providing in `mags`; an example might be `["B","V"]`. These are used when `mag_lim` is finite to determine what filter you want to use to limit the faintest stars you want returned.
 - `limit::Real` gives the total birth stellar mass of the population you want to sample. See the "Notes" section on population masses for more information.
 - `imf::Distributions.UnivariateDistribution{Distributions.Continuous}` is a continuous univariate distribution implementing a stellar initial mass function with a defined `rand(rng::Random.AbstractRNG, imf)` method to use for sampling masses. Implementations of commonly used IMFs are available in [InitialMassFunctions.jl](https://github.com/cgarling/InitialMassFunctions.jl).

# Keyword Arguments
 - `dist_mod::Real=0` is the distance modulus (see [`SFH.distance_modulus`](@ref)) you wish to have added to the returned magnitudes to simulate a population at a particular distance.
 - `rng::Random.AbstractRNG=Random.default_rng()` is the rng instance that will be used to sample the stellar initial masses from `imf`.
 - `mag_lim::Real=Inf` gives the faintest apparent magnitude for stars you want to be returned in the output. Stars fainter than this magnitude will still be sampled and contribute properly to the total mass of the population, but they will not be returned.
 - `mag_lim_name::String="V"` gives the filter name (as contained in `mag_names`) to use when considering if a star is fainter than `mag_lim`. This is unused if `mag_lim` is infinite.
 - `binary_model::SFH.AbstractBinaryModel=Binaries(0.3)` is an instance of a model for treating binaries; options are [`NoBinaries`](@ref) and [`Binaries`](@ref). 

# Notes
## Population Masses
Given a particular isochrone with an initial mass vector `mini_vec`, it will never cover the full range of stellar birth masses because stars that die before present-day are not included in the isochrone. However, these stars *were* born, and so contribute to the total birth mass of the system. There are two ways to properly account for this lost mass when sampling:
 1. Set the upper limit for masses that can be sampled from the `imf` distribution to a physical value for the maximum birth mass of stars (e.g., 100 solar masses). In this case, these stars will be sampled from `imf`, and will contribute their masses to the system, but they will not be returned if their birth mass is greater than `maximum(mini_vec)`. This is typically easiest for the user and only results in ∼15% loss of efficiency for 10 Gyr isochrones.
 2. Set the upper limit for masses that can be sampled from the `imf` distribution to `maximum(mini_vec)` and adjust `limit` to respect the amount of initial stellar mass lost by not sampling higher mass stars. This can be calculated as `new_limit = limit * ( QuadGK.quadgk(x->x*pdf(imf,x), minimum(mini_vec), maximum(mini_vec))[1] / QuadGK.quadgk(x->x*pdf(imf,x), minimum(imf), maximum(imf))[1] )`, with the multiplicative factor being the fraction of the population stellar mass contained in stars with initial masses between `minimum(mini_vec)` and `maximum(mini_vec)`; this factor is the ratio
```math
\\frac{\\int_a^b \\ m \\times \\frac{dN}{dm} \\ dm}{\\int_0^∞ \\ m \\times \\frac{dN}{dm} \\ dm}
```
"""
function generate_stars_mass(mini_vec::AbstractVector{<:Number}, mags, mag_names::AbstractVector{String}, limit::Number, imf::UnivariateDistribution{Continuous}; dist_mod::Number=0, rng::AbstractRNG=default_rng(), mag_lim::Number=Inf, mag_lim_name::String="V", binary_model::AbstractBinaryModel=Binaries(0.3))
    # Interpret and reshape the `mags` argument into a (length(mini_vec), nfilters) vector of vectors.
    mags = ingest_mags(mini_vec, mags)
    mags = [ i .+ dist_mod for i in mags ] # Update mags with the provided distance modulus.
    mini_vec, mags = sort_ingested(mini_vec, mags) # Fix non-sorted mini_vec and deduplicate entries.
    # Construct the sampler object for the provided imf; for some distributions, this will return a
    # Distributions.Sampleable for which rand(imf_sampler) is more efficient than rand(imf).
    imf_sampler = sampler(imf) 
    itp = interpolate((mini_vec,), mags, Gridded(Linear()))
    mmin1, mmax = extrema(mini_vec) # Need this to determine validity for mag interpolation.
    mmin2, _ = mass_limits(mini_vec, mags, mag_names, mag_lim, mag_lim_name) # Determine initial mass that corresponds to mag_lim, if provided.
    # We might be able to gain some efficiency by creating a new truncated IMF with lower bound mmin,
    # when mag_lim is not infinite. Looks like a factor of 2 improvement but without renormalizing `limit`,
    # it will not sample the correct amount of stellar mass. 
    # mmin1 > minimum(mini_vec) && (imf = truncated(imf; lower=mmin1))
    # Setup for iteration. 
    total = zero(eltype(imf))
    mass_vec = Vector{eltype(imf)}(undef,0)
    # mag_vec = Vector{Vector{eltype(imf)}}(undef,0)
    mag_vec = Vector{eltype(mags)}(undef,0)
    while total < limit
        mass_sample = rand(rng, imf_sampler) # Just sample one star.
        total += mass_sample         # Add mass to total.
        # Continue loop if sampled mass is outside of isochrone range.
        if (mass_sample < mmin2) | (mass_sample > mmax)
            continue
        end
        mag_sample = itp(mass_sample) # Roughly 70 ns for 2 filters on 12600k. No speedup for bulk queries.
        # See if we sample any binary stars
        # binary_mass = sample_binary!(mass_sample, mmin1, mmax, mag_sample, imf_sampler, itp, rng, binary_model)
        binary_mass, mag_sample = sample_binary(mass_sample, mmin1, mmax, mag_sample, imf_sampler, itp, rng, binary_model)
        total += binary_mass
        push!(mass_vec, mass_sample + binary_mass) # scipy.interpolate.interp1d is ~74 ns per evaluation for batched 10k queries.
        push!(mag_vec, mag_sample)
    end
    return mass_vec, mag_vec
end

"""
    (sampled_masses, sampled_mags) =  generate_stars_mag(mini_vec::Vector{<:Real}, mags, mag_names::Vector{String}, absmag::Real, absmag_name::String, imf::UnivariateDistribution{Continuous}; dist_mod::Real=0, rng::AbstractRNG=default_rng(), mag_lim::Real=Inf, mag_lim_name::String="V", binary_model::SFH.AbstractBinaryModel=Binaries(0.3))

Generates a mock stellar population with absolute magnitude `absmag::Real` (e.g., -7 or -12) in the filter `absmag_name::String` (e.g., "V" or "F606W") which is contained in the provided `mag_names::Vector{String}`. Other arguments are shared with [`generate_stars_mass`](@ref), which contains the main documentation.

# Notes
## Population Magnitudes
Unlike when sampling a population to a fixed initial birth mass, as is implemented in [`generate_stars_mag`](@ref), when generating a population up to a fixed absolute magnitude, only stars that survive to present-day contribute to the flux of the population. If you choose to limit the apparent magnitude of stars returned by passing the `mag_lim` and `mag_lim_name` keyword arguments, stars fainter than your chosen limit will still be sampled and will still contribute their luminosity to the total population, but they will not be contained in the returned output. 
"""
function generate_stars_mag(mini_vec::Vector{<:Real}, mags, mag_names::Vector{String}, absmag::Real, absmag_name::String, imf::UnivariateDistribution{Continuous}; dist_mod::Real=0, rng::AbstractRNG=default_rng(), mag_lim::Real=Inf, mag_lim_name::String="V", binary_model::AbstractBinaryModel=Binaries(0.3))
    # Interpret and reshape the `mags` argument into a (length(mini_vec), nfilters) vector of vectors.
    mags = ingest_mags(mini_vec, mags)
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
    
    total = zero(eltype(imf))
    mass_vec = Vector{eltype(imf)}(undef,0)
    mag_vec = Vector{Vector{eltype(imf)}}(undef,0)
    while total < limit
        mass_sample = rand(rng, imf_sampler)  # Just sample one star.
        # Continue loop if sampled mass is outside of isochrone range.
        if (mass_sample < mmin1) | (mass_sample > mmax)
            continue
        end
        mag_sample = itp(mass_sample) # Interpolate the sampled mass.
        # See if we sample any binary stars
        binary_mass = sample_binary!(mass_sample, mmin1, mmax, mag_sample, imf_sampler, itp, rng, binary_model)
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

###############################################
#### Functions to generate composite mock galaxy catalogs from multiple SSPs

function generate_stars_mass_composite(mini_vec::Vector{T}, mags::Vector, mag_names::Vector{String}, limit::Real, massfrac::Vector{<:Real}, imf::UnivariateDistribution{Continuous}; kws...) where T<:Vector{<:Real} # dist_mod::Real=0, rng::AbstractRNG=default_rng(), mag_lim::Real=Inf, mag_lim_name::String="V") 
    !(length(mini_vec) == length(mags) == length(massfrac)) && throw(ArgumentError("The arguments `mini_vec`, `mags`, and `massfrac` to `generate_stars_mass_composite` must all be vectors of equal length."))
    ncomposite = length(mini_vec) # Number of stellar populations provided.
    massfrac ./= sum(massfrac) # Renormalize massfrac to sum to 1.
    # Allocate output vectors.
    massvec = [ Vector{eltype(imf)}(undef,0) for i in 1:ncomposite ]
    mag_vec = [ Vector{Vector{eltype(imf)}}(undef,0) for i in 1:ncomposite ]
    # Loop over each component, calling generate_stars_mass. Threading works with good scaling.
    Threads.@threads for i in eachindex(mini_vec, mags, massfrac)
        result = generate_stars_mass(mini_vec[i], mags[i], mag_names, limit * massfrac[i], imf; kws...)
        massvec[i] = result[1]
        mag_vec[i] = result[2]
    end
    return massvec, mag_vec
end

# For generate_stars_mags_composite we probably want to support both luminosity and initial mass fractions.
function generate_stars_mag_composite(mini_vec::Vector{T}, mags::Vector, mag_names::Vector{String}, absmag::Real, absmag_name::String, fracs::Vector{<:Real}, imf::UnivariateDistribution{Continuous}; frac_type::String="lum", kws...) where T<:Vector{<:Real}
    !(length(mini_vec) == length(mags) == length(fracs)) && throw(ArgumentError("The arguments `mini_vec`, `mags`, and `fracs` to `generate_stars_mag_composite` must all be vectors of equal length."))
    ncomposite = length(mini_vec) # Number of stellar populations provided.
    fracs ./= sum(fracs) # Renormalize fracs to sum to 1.
    # Interpret whether user requests `fracs` represent luminosity or mass fractions.
    if frac_type == "lum"
        limit = L_from_MV(absmag)   # Convert the provided `limit` from magnitudes into luminosity.
        fracs = fracs .* limit      # Portion the total luminosity across the different input models.
        @. fracs = MV_from_L(fracs) # Convert back to magnitudes.
    elseif frac_type == "mass"
        throw(ArgumentError("`frac_type == mass` not yet implemented."))
    else
        throw(ArgumentError("Supported `frac_type` arguments for generate_stars_mag_composite are \"lum\" or \"mass\"."))
    end
    # Allocate output vectors.
    massvec = [ Vector{eltype(imf)}(undef,0) for i in 1:ncomposite ]
    mag_vec = [ Vector{Vector{eltype(imf)}}(undef,0) for i in 1:ncomposite ]
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

function model_cmd(mags::Vector{T}, errfuncs::Vector, completefuncs::Vector; rng::AbstractRNG=default_rng()) where T<:Vector{<:Real}
    nstars = length(mags)
    nfilters = length(first(mags))
    !(nfilters == length(errfuncs) == length(completefuncs)) && throw(ArgumentError("The `errfuncs` and `completefuncs` arguments to `model_cmd` must have length equal to the elements of `mags` representing the number of filters that you are providing magnitudes for."))
    randsamp = rand(rng, nstars) # Draw nstars random uniform variates for completeness testing.
    # magmat = hcat(mags...)
    # completeness = hcat([ completefuncs[i].(view(magmat,i,:)) for i in 1:nfilters ]...)
    # completeness = map(x->reduce(*,x), eachrow(completeness))
    # return completeness
    completeness = ones(nstars) # Vector{eltype(mags[1])}(undef, nstars)
    cvals = Vector{eltype(completeness)}(undef, nfilters)
    # Estimate the overall completeness as the product of the single-band completeness values.
    for i in eachindex(mags)
        cvals .= mags[i]
        for j in eachindex(completefuncs)
            completeness[i] *= completefuncs[j](cvals[j])
        end
    end
    # Pick out the entries where the random number is less than the product of the single-band completeness values 
    good = findall( map(<=, randsamp, completeness) ) # findall( randsamp .<= completeness )
    ret_mags = mags[good] # Get the good mags to be returned.
    for i in eachindex(ret_mags)
        cvals .= ret_mags[i]
        for j in eachindex(errfuncs)
            ret_mags[i][j] += ( randn(rng) * errfuncs[j](cvals[j]) ) # Add error. 
        end
    end
    return ret_mags
end
# model_cmd(mags::Matrix, args...; kws...) = model_cmd(
