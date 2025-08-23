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
        nstars = length(mini_vec) 
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
    @argcheck axes(mini_vec) == axes(mags)
    idx = sortperm(mini_vec)
    if idx != eachindex(mini_vec)
        mini_vec = mini_vec[idx]
        mags = mags[idx]
    end
    mini_vec = deduplicate_knots!(mini_vec; move_knots=true) # Interpolations.jl function. 
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
    @argcheck axes(mini_vec) == axes(mags) 
    mmin, mmax = extrema(mini_vec)
    # Update mmin respecing `mag_lim`, if provided.
    if !isfinite(mag_lim)
        return mmin, mmax
    else
        idxmag = findfirst(x->x==mag_lim_name, mag_names) # Find the index into mag_lim_names where == mag_lim_name.
        idxmag == nothing && throw(ArgumentError("Provided `mag_lim_name` is not contained in provided `mag_names` array."))
        # If mag_lim is brighter than the faintest star in the isochrone, then we solve
        if mag_lim < mags[findfirst(x->x==mmin, mini_vec)][idxmag]
            tmp_mags = [i[idxmag] for i in mags]
            if mag_lim < minimum(tmp_mags)
                throw(DomainError(mag_lim, "The provided `mag_lim` is brighter than all the stars in the input `mags` array assuming the input distance modulus `dist_mod`. Revise your arguments."))
            end
            # Solve for stellar initial mass where mag == mag_lim by constructing interpolator and root-finding.
            # If the isochrone includes post-RGB evolution, then luminosity may not be a
            # monotonic function of initial birth mass. To bracket our root-finding, we will
            # find the brightest star in the isochrone and use the initial mass of that star
            # as the upper limit.
            brightest = findmin(tmp_mags)
            itp = interpolate((mini_vec,), tmp_mags, Gridded(Linear()))
            mmin2 = find_zero(x -> itp(x) - mag_lim, (mmin, mini_vec[brightest[2]])) # (mmin, mmax))
            return mmin2, mmax
        else
            return mmin, mmax
        end
    end
end

##############################################################
#### Types and methods for non-interacting binary calculations
""" `StarFormationHistories.AbstractBinaryModel` is the abstract supertype for all types that are used to model multi-star systems in the package. All concrete subtypes should implement the following methods to support all features:
 - [`StarFormationHistories.sample_system`](@ref)
 - [`StarFormationHistories.binary_system_fraction`](@ref)
 - [`StarFormationHistories.binary_mass_fraction`](@ref)
 - `Base.length`, which should return an integer indicating the number of stars per system that can be sampled by \
   the model; this is equivalent to the length of the mass vector returned by `sample_system`.

Note that all quantities relating to binary populations (e.g., `binary_system_fraction`) should be defined for the population *at birth*. As the stars in a binary system evolve, the more massive star may die before the system is observed at present-day. Of course, the stars in single-star systems can also die. If the rate at which binary systems become single-star systems is not equal to the rate at which single-star systems die, then there can be net transfer between these populations over time. Therefore the observed, present-day binary system fraction of an evolved population is not necessarily equal to the fraction at birth, which is the more fundamental quantity."""
abstract type AbstractBinaryModel end
Base.Broadcast.broadcastable(m::AbstractBinaryModel) = Ref(m)
"""
    binary_system_fraction(model::T) where T <: AbstractBinaryModel
Returns the number fraction of *stellar systems* that are binaries for the given concrete subtype `T <: AbstractBinaryModel`. Has a default implementation of `binary_system_fraction(model::AbstractBinaryModel) = model.fraction`."""
binary_system_fraction(model::AbstractBinaryModel) = model.fraction
"""
    binary_number_fraction(model::T) where T <: AbstractBinaryModel
Returns the number fraction of *stars* that in binary pairs for the given concrete subtype `T <: AbstractBinaryModel`. Has a default implementation of `2b / (1+b)`, where `b` is the result of [`StarFormationHistories.binary_system_fraction`](@ref)."""
binary_number_fraction(model::AbstractBinaryModel) = (b=binary_system_fraction(model); return 2b / (1 + b))
"""
    binary_mass_fraction(model::T, imf) where T <: AbstractBinaryModel
Returns the fraction of stellar mass in binary systems for the given concrete subtype `T <: AbstractBinaryModel` and initial mass function `imf`. `imf` must be a properly normalized probability distribution such that the number fraction of stars/systems between mass `m1` and `m2` is given by the integral of `dispatch_imf(imf, x)` from `m1` to `m2`. 
"""
binary_mass_fraction(model::T, imf) where T <: AbstractBinaryModel

"""
    NoBinaries()
The `NoBinaries` type indicates that no binaries of any kind should be created. """
struct NoBinaries <: AbstractBinaryModel end
Base.length(::NoBinaries) = 1
binary_system_fraction(::NoBinaries) = 0
binary_number_fraction(::NoBinaries) = 0
binary_mass_fraction(::NoBinaries, imf) = 0

"""
    RandomBinaryPairs(fraction::Real)
The `RandomBinaryPairs` type takes one argument `0 <= fraction::Real <= 1` that denotes the number fraction of stellar systems that are binaries (e.g., 0.3 for 30% binary fraction) and will sample binaries as random pairs of two stars drawn from the same single-star IMF. This model will ONLY generate up to one additional star -- it will not generate any 3+ star systems. This model typically incurs a 10--20% speed penalty relative to `NoBinaries`. """
struct RandomBinaryPairs{T <: Real} <: AbstractBinaryModel
    fraction::T
    # Outer-only constructor to guarantee support
    # See https://docs.julialang.org/en/v1/manual/constructors/#Outer-only-constructors
    function RandomBinaryPairs(fraction::T) where T <: Real
        @argcheck (fraction >= zero(T)) && (fraction <= one(T))
        new{T}(fraction)
    end
end
Base.length(::RandomBinaryPairs) = 2
# ```math
# \\int_{\\text{M}_\\text{min}}^{\\text{M}_\\text{max}} \\text{M} \\frac{d\\text{N} \\left( \\text{M} \\right)}{d\\text{M}}  d\\text{M}   =  \\int_{\\text{M}_\\text{min}}^{\\text{M}_\\text{max}} \\int_{\\text{M}_\\text{min}}^{\\text{M}_P} \\left( \\text{M}_P + \\text{M}_S \\right) \\frac{d\\text{N} \\left( \\text{M}_S \\right)}{d\\text{M}} \\frac{d\\text{N} \\left( \\text{M}_P \\right)}{d\\text{M}} d\\text{M}_S \\, d\\text{M}_P
# ```
# quadgk(Mp->quadgk(Ms->(Ms+Mp)*pdf(imf,Ms) * pdf(imf,Mp), minimum(imf), Mp)[1], extrema(imf)...) is equal to mean(imf)
# quadgk(Mp->quadgk(Ms->(Ms+Mp)*pdf(imf,Ms) * pdf(imf,Mp), extrema(imf)...)[1], extrema(imf)...) is equal to mean(imf)
"""
    binary_mass_fraction(m::RandomBinaryPairs, imf)
The `RandomBinaryPairs` model uses a single-star `imf`. If a system is chosen to be a binary pair, two stars are drawn from the single-star `imf` and the more massive star is made the primary. Given this model, it can be shown that the expectation value for the mass of a binary system is twice the expectation value for single star systems:

```math
2\\int_{\\text{M}_\\text{min}}^{\\text{M}_\\text{max}} \\text{M} \\frac{d\\text{N} \\left( \\text{M} \\right)}{d\\text{M}}  d\\text{M}  =  \\int_{\\text{M}_\\text{min}}^{\\text{M}_\\text{max}} \\int_{\\text{M}_\\text{min}}^{\\text{M}_\\text{max}} \\left( \\text{M}_P + \\text{M}_S \\right) \\frac{d\\text{N} \\left( \\text{M}_S \\right)}{d\\text{M}} \\frac{d\\text{N} \\left( \\text{M}_P \\right)}{d\\text{M}} d\\text{M}_S \\, d\\text{M}_P
```

for primary mass ``\\text{M}_P``, secondary mass ``\\text{M}_S``, and single-star IMF ``d\\text{N} / d\\text{M}``. As such, the fraction of total stellar mass in binaries is equal to the number fraction of all *stars* in binary pairs, which is given by [`StarFormationHistories.binary_number_fraction`](@ref).
"""
binary_mass_fraction(m::RandomBinaryPairs, imf) = binary_number_fraction(m)

"""
    BinaryMassRatio(fraction::Real,
                    qdist::Distributions.ContinuousUnivariateDistribution =
                        Distributions.Uniform(0.1, 1.0))
The `BinaryMassRatio` type takes two arguments; the number fraction of stellar systems that are binaries `0 <= fraction::Real <= 1` and a continuous univariate distribution `qdist` from which to sample binary mass ratios, defined as the ratio of the secondary mass to the primary mass: ``q = \\text{M}_S / \\text{M}_P``. The provided `qdist` must have the proper support of `(minimum(qdist) >= 0) & (maximum(qdist) <= 1)`. Users may find the [`Distributions.truncated`](https://juliastats.org/Distributions.jl/stable/truncate/#Distributions.truncated) method useful for enforcing this support on more general distributions. The default `qdist` is a uniform distribution from 0.1 to 1, which appears to give reasonably good agreement to observations (see, e.g., [Goodwin 2013](https://ui.adsabs.harvard.edu/abs/2013MNRAS.430L...6G)).
"""
struct BinaryMassRatio{T <: Real, S <: Distribution{Univariate, Continuous}} <: AbstractBinaryModel
    fraction::T
    qdist::S
    # Outer-only constructor to guarantee support
    # See https://docs.julialang.org/en/v1/manual/constructors/#Outer-only-constructors
    function BinaryMassRatio(fraction::T, qdist::S=Uniform(0.1, 1.0)) where {T <: Real, S <: Distribution{Univariate, Continuous}}
        V = eltype(qdist)
        @argcheck (fraction >= zero(T)) && (fraction <= one(T))
        @argcheck (minimum(qdist) >= zero(V)) && (maximum(qdist) <= one(V))
        new{T,S}(fraction, qdist)
    end
end
Base.length(::BinaryMassRatio) = 2
# quadgk(M -> quadgk(q -> M * pdf(qdist,q) * pdf(imf, M), extrema(qdist)...)[1], extrema(imf)...)
"""
    binary_mass_fraction(m::BinaryMassRatio, imf)
This binary model requires an `imf` that is defined by stellar system mass. If a system with a randomly sampled mass ``M`` is is a binary, the primary and secondary mass are determined based on a binary mass ratio ``q`` sampled from a user-defined distribution. By definition, the expectation value for the total mass of a binary system is equal to the expectation value for single-star systems. In this case the binary mass fraction is equal the binary system number fraction as given by [`StarFormationHistories.binary_system_fraction`](@ref).
"""
binary_mass_fraction(m::BinaryMassRatio, imf) = binary_system_fraction(m)

"""
    masses = sample_system(imf, rng::AbstractRNG, binarymodel::StarFormationHistories.AbstractBinaryModel)

Simulates the effects of non-interacting, unresolved stellar companions on stellar photometry. Implementation depends on the choice of `binarymodel`.

# Arguments
 - `imf`: an object implementing `rand(imf)` to draw a random mass for a single star or a stellar system (depends on choice of `binarymodel`)
 - `rng::AbstractRNG`: the random number generator to use when sampling stars
 - `binarymodel::StarFormationHistories.AbstractBinaryModel`: an instance of a binary model that determines which implementation will be used; currently provided options are [`NoBinaries`](@ref), [`RandomBinaryPairs`](@ref), and [`BinaryMassRatio`](@ref).

# Returns
 - `masses::SVector{N,eltype(imf)}`: the masses of the individual stars sampled in the system in descending order where `N` is the maximum number of stars that can be sampled by the provided `binarymodel` as given by `Base.length(binarymodel)`. 
"""
@inline sample_system(imf, rng::AbstractRNG, binarymodel::NoBinaries) = SVector{1,eltype(imf)}(rand(rng, imf))

@inline function sample_system(imf, rng::AbstractRNG, binarymodel::RandomBinaryPairs)
    frac = binarymodel.fraction
    r = rand(rng) # Random uniform number
    mass1 = rand(rng, imf) # First star mass
    if r <= frac  # Generate a binary star
        mass2 = rand(rng, imf)
        return sort(SVector{2,eltype(imf)}(mass1, mass2); rev=true)
    else
        return SVector{2,eltype(imf)}(mass1, zero(eltype(imf)))
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
    generate_stars_mass(mini_vec::AbstractVector{<:Number},
                        mags, mag_names::AbstractVector{String},
                        limit::Number,
                        imf::Distributions.Sampleable{Distributions.Univariate, Distributions.Continuous};
                        dist_mod::Number=0,
                        rng::Random.AbstractRNG=Random.default_rng(),
                        mag_lim::Number = Inf,
                        mag_lim_name::String = "V",
                        binary_model::StarFormationHistories.AbstractBinaryModel =
                            StarFormationHistories.RandomBinaryPairs(0.3))

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
 - `binary_model::StarFormationHistories.AbstractBinaryModel=StarFormationHistories.RandomBinaryPairs(0.3)` is an instance of a model for treating binaries; currently provided options are [`NoBinaries`](@ref), [`RandomBinaryPairs`](@ref), and [`BinaryMassRatio`](@ref).

# Returns
`(sampled_masses, sampled_mags)` defined as
 - `sampled_masses::Vector{SVector{N,eltype(imf)}}` is a vector containing the initial stellar masses of the stars sampled by [`sample_system`](@ref); see that method's documentation for details on format. In short, each `StaticArrays.SVector` represents one stellar system and each entry in each `StaticArrays.SVector` is one star in that system. Entries will be 0 when companions could have been sampled but were not (i.e., when using a `binary_model` that supports multi-star systems). 
 - `sampled_mags::Vector{SVector{N,<:Number}}` is a vector containing `StaticArrays.SVectors` with the multi-band magnitudes of the sampled stars. To get the magnitude of star `i` in band `j`, you index as `sampled_mags[i][j]`. This can be reinterpreted as a 2-dimensional `Matrix` with `reduce(hcat,sampled_mags)`. 

# Notes
## Population Masses
Given a particular isochrone with an initial mass vector `mini_vec`, it will never cover the full range of stellar birth masses because stars that die before present-day are not included in the isochrone. However, these stars *were* born, and so contribute to the total birth mass of the system. There are two ways to properly account for this lost mass when sampling:
 1. Set the upper limit for masses that can be sampled from the `imf` distribution to a physical value for the maximum birth mass of stars (e.g., 100 solar masses). In this case, these stars will be sampled from `imf`, and will contribute their masses to the system, but they will not be returned if their birth mass is greater than `maximum(mini_vec)`. This is typically easiest for the user and only results in ∼15% loss of efficiency for 10 Gyr isochrones. *This approach is preferred when sampling with binaries.*
 2. Set the upper limit for masses that can be sampled from the `imf` distribution to `maximum(mini_vec)` and adjust `limit` to respect the amount of initial stellar mass lost by not sampling higher mass stars. This can be calculated as `new_limit = limit * ( QuadGK.quadgk(x->x*pdf(imf,x), minimum(mini_vec), maximum(mini_vec))[1] / QuadGK.quadgk(x->x*pdf(imf,x), minimum(imf), maximum(imf))[1] )`, with the multiplicative factor being the fraction of the population stellar mass contained in stars with initial masses between `minimum(mini_vec)` and `maximum(mini_vec)`; this factor is the ratio
```math
\\frac{\\int_a^b \\ m \\times \\frac{dN(m)}{dm} \\ dm}{\\int_0^∞ \\ m \\times \\frac{dN(m)}{dm} \\ dm}.
```
Note that, if binaries are included, this approach only forms binary pairs between stars whose masses are less than `maximum(mini_vec)`. This is probably not desired, so we recommend the previous approach when including binaries.
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
    mass_vec = Vector{SVector{length(binary_model),eltype(imf)}}(undef,0)
    mag_vec = Vector{eltype(mags)}(undef,0)
    while total < limit
        masses = sample_system(imf_sampler, rng, binary_model) # Sample masses of stars in a single system
        total += sum(masses)                           # Add to accumulator
        # if (first(masses) < mmin2) | (first(masses) > mmax)
        if first(masses) < mmin2 # Primary by itself would be fainter than `mag_lim` so continue
            continue
        end
        flux = zero(eltype(mags))
        for mass in masses
            if (mass > mmin1) & (mass < mmax) # Mass is in valid interpolation range
                flux += mag2flux.( itp(mass) ) # Think this works for SVectors? not sure
            end
        end
        if sum(flux) > 0 # Possible that first(masses) > mmax and no valid companions either, meaning sum(lum) == 0
            push!(mass_vec, masses)
            push!(mag_vec, flux2mag.(flux))
        end
    end
    return mass_vec, mag_vec
end

"""
    (sampled_masses, sampled_mags) =  generate_stars_mag(mini_vec::AbstractVector{<:Number}, mags, mag_names::AbstractVector{String}, absmag::Real, absmag_name::String, imf::Distributions.Sampleable{Distributions.Univariate,Distributions.Continuous}; dist_mod::Number=0, rng::AbstractRNG=default_rng(), mag_lim::Number=Inf, mag_lim_name::String="V", binary_model::StarFormationHistories.AbstractBinaryModel=RandomBinaryPairs(0.3))

Generates a mock stellar population from an isochrone defined by the provided initial stellar masses `mini_vec`, absolute magnitudes `mags`, and filter names `mag_names`. The population is sampled to a total absolute magnitude `absmag::Real` (e.g., -7 or -12) in the filter `absmag_name::String` (e.g., "V" or "F606W") which is contained in the provided `mag_names::AbstractVector{String}`. Other arguments are shared with [`generate_stars_mass`](@ref), which contains the main documentation.

# Notes
## Population Magnitudes
Unlike when sampling a population to a fixed initial birth mass, as is implemented in [`generate_stars_mass`](@ref), when generating a population up to a fixed absolute magnitude, only stars that survive to present-day contribute to the flux of the population. If you choose to limit the apparent magnitude of stars returned by passing the `mag_lim` and `mag_lim_name` keyword arguments, stars fainter than your chosen limit will still be sampled and will still contribute their luminosity to the total population, but they will not be contained in the returned output. 
"""
generate_stars_mag(mini_vec::AbstractVector{<:Number}, mags, args...; kws...) =
    generate_stars_mag(mini_vec, ingest_mags(mini_vec, mags), args...; kws...)
function generate_stars_mag(mini_vec::AbstractVector{<:Number}, mags::AbstractVector{SVector{N,T}}, mag_names::AbstractVector{String}, absmag::Number, absmag_name::String, imf::Sampleable{Univariate,Continuous}; dist_mod::Number=0, rng::AbstractRNG=default_rng(), mag_lim::Number=Inf, mag_lim_name::String="V", binary_model::AbstractBinaryModel=RandomBinaryPairs(0.3)) where {N, T<:Number}
    # Setup and input ingestion
    mags = [ i .+ dist_mod for i in mags ]         # Update mags with the provided distance modulus.
    mini_vec, mags = sort_ingested(mini_vec, mags) # Fix non-sorted mini_vec and deduplicate entries.
    # Convert the provided `limit` from absolute magnitudes to apparent magnitudes and then
    # into flux; the conversion to apparent mag here will make it easier to accumulate
    # the flux later in the loop, since `mags` has distance modulus added as well. 
    limit = mag2flux(absmag + dist_mod) 
    # Construct the sampler object for the provided imf; for some distributions, this will return a
    # Distributions.Sampleable for which rand(imf_sampler) is more efficient than rand(imf).
    imf_sampler = sampler(imf)
    itp = interpolate((mini_vec,), mags, Gridded(Linear()))
    mmin1, mmax = extrema(mini_vec) # Need this to determine validity for mag interpolation.

    # Find indices into `mags` and `mag_names` that correspdong to `limit_name` and `mag_lim_name`.
    idxlim = findfirst(x->x==absmag_name, mag_names)
    # Throw error if `absmag_name` not in `mag_names`.
    if idxlim == nothing
        throw(ArgumentError("Provided `absmag_name` is not contained in provided `mag_names` array."))
    end
    sourceidx = findfirst(x->x==mag_lim_name, mag_names) # Get the index into `mags` and `mag_names` that equals `mag_lim_name`
    # Throw error if `mag_lim` is finite (not `Inf`) and `mag_lim_name` not in `mag_names`.
    if (isfinite(mag_lim) & (sourceidx == nothing))
        throw(ArgumentError("Provided `mag_lim_name` is not contained in provided `mag_names` array."))
    end

    # Set up accumulators and loop
    total = zero(T)
    mass_vec = Vector{SVector{length(binary_model),eltype(imf)}}(undef,0)
    mag_vec = Vector{eltype(mags)}(undef,0)
    while total < limit
        masses = sample_system(imf_sampler, rng, binary_model) # Sample masses of stars in a single system
        flux = zero(eltype(mags)) # Accumulator to hold per-filter luminosities
        for mass in masses
            # Continue loop if sampled mass is outside of valid isochrone range
            if (mass < mmin1) | (mass > mmax)
                continue
            end
            flux += mag2flux.( itp(mass) )
        end
        total += flux[idxlim] # Add the luminosity in the correct filter to the `total` accumulator
        # If the source is bright enough, or `mag_lim` is infinite, add it to output
        if (sum(flux) > 0) & (!isfinite(mag_lim) || flux2mag(flux[sourceidx]) < mag_lim) 
            push!(mass_vec, masses)
            push!(mag_vec, flux2mag.(flux))
        end
    end
    # println(flux2mag(total) - dist_mod) # Print absolute magnitude sampled
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
 - `massfrac::AbstractVector{<:Number}` is a vector giving the relative fraction of mass allotted to each individual stellar population; length must be equal to the length of `mini_vec` and `mags`. 
 - `imf::Distributions.Sampleable{Distributions.Univariate, Distributions.Continuous}` is a sampleable continuous univariate distribution implementing a stellar initial mass function with a defined `rand(rng::Random.AbstractRNG, imf)` method to use for sampling masses. All instances of `Distributions.ContinuousUnivariateDistribution` are also valid. Implementations of commonly used IMFs are available in [InitialMassFunctions.jl](https://github.com/cgarling/InitialMassFunctions.jl).

# Keyword Arguments
All keyword arguments `kws...` are passed to [`generate_stars_mass`](@ref); you should refer to that method's documentation for more information. 

# Returns
 - `sampled_masses::Vector{Vector{SVector{N,eltype(imf)}}}` is a vector of vectors containing the initial stellar masses of the sampled stars. The outer vectors are separated by the isochrone the stars were generated from; i.e., all of `sampled_masses[1]` were sampled from `mini_vec[1]` and so on. These can be concatenated into a single vector with `reduce(vcat,sampled_masses)`. The format of the contained `StaticArrays.SVector`s are as output from [`sample_system`](@ref); see that method's documentation for more details. 
 - `sampled_mags::Vector{Vector{SVector{N,<:Number}}}` is a vector of vectors containing `StaticArrays.SVectors` with the multi-band magnitudes of the sampled stars. The outer vectors are separated by the isochrone the stars were generated from; i.e. all of `sampled_mags[1]` were sampled from `mags[1]` and so on. To get the magnitude of star `i` in band `j` sampled from isochrone `k`, you would do `sampled_mags[k][i][j]`. This can be concatenated into a `Vector{SVector}` with `reduce(vcat,sampled_mags)` and a 2-D `Matrix` with `reduce(hcat,reduce(vcat,sampled_mags))`. 
"""
function generate_stars_mass_composite(mini_vec::AbstractVector{T}, mags::AbstractVector, mag_names::AbstractVector{String}, limit::Number, massfrac::AbstractVector{<:Number}, imf::Sampleable{Univariate,Continuous}; binary_model::AbstractBinaryModel=RandomBinaryPairs(0.3), kws...) where T <: AbstractVector{<:Number} 
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
        result = generate_stars_mass(mini_vec[i], mags[i], mag_names, limit * massfrac[i], imf;
                                      binary_model = binary_model, kws...)
        massvec[i] = result[1]
        mag_vec[i] = result[2]
    end
    return massvec, mag_vec
end

"""
    (sampled_masses, sampled_mags) = generate_stars_mag_composite(mini_vec::AbstractVector{<:AbstractVector{<:Number}}, mags::AbstractVector, mag_names::AbstractVector{String}, absmag::Number, absmag_name::String, fracs::AbstractVector{<:Number}, imf::Sampleable{Univariate,Continuous}; frac_type::Symbol=:lum, kws...)

Generates a random sample of stars with a complex star formation history using multiple isochrones. Very similar to [`generate_stars_mag`](@ref) except the isochrone-related arguments `mini_vec` and `mags` should now be vectors of vectors containing the relevant data for the full set of isochrones to be considered. The total absolute magnitude of the sampled population is given by `absmag`. The proportion of the luminosity allotted to each of the individual isochrones is given by the entries of the `frac` vector. This basically just proportions the luminosity according to `frac` and calls [`generate_stars_mag`](@ref) for each of the individual stellar populations; as such it is set up to multi-thread across the multiple stellar populations. 

# Arguments
 - `mini_vec::AbstractVector{<:AbstractVector{<:Number}}` contains the initial masses (in solar masses) for the stars in each isochrone; the internal vectors must be mutable as we will call `Interpolations.deduplicate_knots!` on each. The length of `mini_vec` should be equal to the number of isochrones. 
 - `mags` contains the absolute magnitudes from the isochrones in the desired filters corresponding to the same stars as provided in `mini_vec`. The length of `mags` should be equal to the number of isochrones. The individual elements of `mags` are each internally interpreted and converted into a standard format by [`StarFormationHistories.ingest_mags`](@ref). The valid formats for the individual elements of `mags` are:
    - `AbstractVector{AbstractVector{<:Number}}`, in which case the length of the vector `length(mags[i])` can either be equal to `length(mini_vec[i])`, in which case the length of the inner vectors must all be equal to the number of filters you are providing, or the length of the outer vector can be equal to the number of filters you are providing, and the length of the inner vectors must all be equal to `length(mini_vec[i])`; this is the more common use-case.
    - `AbstractMatrix{<:Number}`, in which case `mags[i]` must be 2-dimensional. Valid shapes are `size(mags[i]) == (length(mini_vec[i]), nfilters)` or `size(mags[i]) == (nfilters, length(mini_vec[i]))`, with `nfilters` being the number of filters you are providing.
 - `mag_names::AbstractVector{String}` contains strings describing the filters you are providing in `mags`; an example might be `["B","V"]`. These are used when `mag_lim` is finite to determine what filter you want to use to limit the faintest stars you want returned. These are assumed to be the same for all isochrones.
 - `absmag::Number` gives the total absolute magnitude of the complex population to be sampled.
 - `absmag_name::String` is the name of the filter for which the desired absolute magnitude is `absmag`; must be contained in `mag_names`.
 - `fracs::AbstractVector{<:Number}` is a vector giving the relative fraction of luminosity or mass (determined by the `frac_type` keyword argument) allotted to each individual stellar population; length must be equal to the length of `mini_vec` and `mags`. 
 - `imf::Distributions.Sampleable{Distributions.Univariate, Distributions.Continuous}` is a sampleable continuous univariate distribution implementing a stellar initial mass function with a defined `rand(rng::Random.AbstractRNG, imf)` method to use for sampling masses. All instances of `Distributions.ContinuousUnivariateDistribution` are also valid. Implementations of commonly used IMFs are available in [InitialMassFunctions.jl](https://github.com/cgarling/InitialMassFunctions.jl).

# Keyword Arguments
 - `frac_type::Symbol` either `:lum`, in which case `fracs` is assumed to contain the relative luminosity fractions for each individual isochrone, or `:mass`, in which case it is assumed that `fracs` contains mass fractions (`:mass` is not yet implemented). 
All other keyword arguments `kws...` are passed to [`generate_stars_mag`](@ref); you should refer to that method's documentation for more information. 

# Returns
 - `sampled_masses::Vector{Vector{SVector{N,eltype(imf)}}}` is a vector of vectors containing the initial stellar masses of the sampled stars. The outer vectors are separated by the isochrone the stars were generated from; i.e., all of `sampled_masses[1]` were sampled from `mini_vec[1]` and so on. These can be concatenated into a single vector with `reduce(vcat,sampled_masses)`. The format of the contained `StaticArrays.SVector`s are as output from [`sample_system`](@ref); see that method's documentation for more details. 
 - `sampled_mags::Vector{Vector{SVector{N,<:Number}}}` is a vector of vectors containing `StaticArrays.SVectors` with the multi-band magnitudes of the sampled stars. The outer vectors are separated by the isochrone the stars were generated from; i.e. all of `sampled_mags[1]` were sampled from `mags[1]` and so on. To get the magnitude of star `i` in band `j` sampled from isochrone `k`, you would do `sampled_mags[k][i][j]`. This can be concatenated into a `Vector{SVector}` with `reduce(vcat,sampled_mags)` and a 2-D `Matrix` with `reduce(hcat,reduce(vcat,sampled_mags))`. 
"""
function generate_stars_mag_composite(mini_vec::AbstractVector{T}, mags::AbstractVector, mag_names::AbstractVector{String}, absmag::Number, absmag_name::String, fracs::AbstractVector{<:Number}, imf::Sampleable{Univariate,Continuous}; frac_type::Symbol=:lum, binary_model::AbstractBinaryModel=RandomBinaryPairs(0.3), kws...) where T <: AbstractVector{<:Number}
    !(axes(mini_vec,1) == axes(mags,1) == axes(fracs,1)) && throw(ArgumentError("The arguments `mini_vec`, `mags`, and `fracs` to `generate_stars_mag_composite` must all have equal length and identical indexing."))
    ncomposite = length(mini_vec) # Number of stellar populations provided.
    fracs = fracs ./ sum(fracs) # Ensure fracs is normalized to sum to 1.
    # Interpret whether user requests `fracs` represent luminosity or mass fractions.
    if frac_type == :lum
        limit = mag2flux(absmag)   # Convert the provided `limit` from magnitudes into flux.
        fracs = flux2mag.( fracs .* limit )
    elseif frac_type == :mass
        throw(ArgumentError("`frac_type == :mass` not yet implemented."))
    else
        throw(ArgumentError("Supported `frac_type` arguments for generate_stars_mag_composite are `:lum` or `:mass`."))
    end
    # Allocate output vectors.
    massvec = [ Vector{SVector{length(binary_model),eltype(imf)}}(undef,0) for i in 1:ncomposite ]
    # Need to ingest here so we know what type of SVector we're going to be putting into mag_vec. 
    mags = [ ingest_mags(mini_vec[i], mags[i]) for i in eachindex( mini_vec, mags ) ]
    mag_vec = [ Vector{eltype(i)}(undef,0) for i in mags ]
    # Loop over each component, calling generate_stars_mag. Threading works with good scaling.
    for i in eachindex(mini_vec, mags, fracs)
        result = generate_stars_mag(mini_vec[i], mags[i], mag_names, fracs[i], absmag_name, imf;
                                     binary_model = binary_model, kws...)
        massvec[i] = result[1]
        mag_vec[i] = result[2]
    end
    return massvec, mag_vec
end

###############################################
#### Functions for modelling observational effects
"""
    new_mags [, good_idxs] = model_cmd(mags::AbstractVector{<:AbstractVector{<:Number}}, 
                                       errfuncs, 
                                       completefuncs,
                                       biasfuncs = [zero for i in completefuncs]; 
                                       rng::Random.AbstractRNG = Random.default_rng(), 
                                       ret_idxs::Bool = false)

Simple method for modeling photometric error and incompleteness to "mock observe" a pure catalog of stellar photometry, such as those produced by [`generate_stars_mass`](@ref) and [`generate_stars_mag`](@ref). This method assumes Gaussian photometric errors and that the photometric error and completeness functions are separable by filter. 

# Arguments
 - `mags::AbstractVector{<:AbstractVector{<:Number}}`: a vector of vectors giving the magnitudes of each star to be modeled. The first index is the per-star index and the second index is the per-filter index (so `mags[10][2]` would give the magnitude of the tenth star in the second filter). This is the same format as the magnitudes returned by [`generate_stars_mass`](@ref) and [`generate_stars_mag`](@ref); to use output from the composite versions, you must first `reduce(vcat,mags)` before passing to this function.
 - `errfuncs`: an iterable (typically a vector or tuple) of callables (typically functions or interpolators) with length equal to the number of filters contained in the elements of `mags`. This iterable must contain callables that, when called with the associated magnitudes from `mags`, will return the expected 1-σ photometric error at that magnitude. The organization is such that the photometric error for star `i` in band `j` is `σ_ij = errfuncs[j](mags[i][j])`. 
 - `completefuncs`: an iterable (typically a vector or tuple) of callables (typically functions or interpolators) with length equal to the number of filters contained in the elements of `mags`. This iterable must contain callables that, when called with the associated magnitudes from `mags`, will return the probability that a star with that magnitude in that band will be found in your color-magnitude diagram (this should include the original detection probability and any post-detection quality, morphology, or other cuts). The organization is such that the detection probability for star `i` in band `j` is `c_ij = completefuncs[j](mags[i][j])`.
 - `biasfuncs = [zero for i in completefuncs]`: an iterable (typically a vector or tuple) of callables (typically functions or interpolators) with length equal to the number of filters contained in the elements of `mags`. This iterable must contain callables that, when called with the associated magnitudes from `mags`, will return the expectation value of the photometric bias at that magnitude. We define the photometric bias as the typical difference between measured and input magnitudes; i.e., `⟨measured - input⟩`. This definition matches that of the bias functions returned by [`process_ASTs`](@ref)). By default this method assumes no photometric bias.

# Keyword Arguments
 - `rng::AbstractRNG = Random.default_rng()`: The object to use for random number generation.
 - `ret_idxs::Bool`: whether to return the indices of the input `mags` for the stars that were successfully "observed" and are represented in the output `new_mags`.

# Returns
 - `new_mags`: an object similar to `mags` (i.e., a `Vector{Vector{<:Number}}`, `Vector{SVector{N,<:Number}}`, etc.) containing the magnitudes of the mock-observed stars. This will be shorter than the provided `mags` vector as we are modeling photometric incompleteness, and the magnitudes will also have random photometric errors added to them. This can be reinterpreted as a 2-dimensional `Matrix` with `reduce(hcat,new_mags)`.
 - `good_idxs`: if `ret_idxs` is `true`, the vector of indices into the input `mags` for the stars that were successfully "observed" and are represented in the output `new_mags`.

# Notes
 - This is a simple implementation that seeks to show a simple example of how one can post-process catalogs of "pure" stars from methods like [`generate_stars_mass`](@ref) and [`generate_stars_mag`](@ref) to include observational effects. This method assumes Gaussian photometric errors, which may not, in general, be accurate. It also assumes that the total detection probability can be modeled as the product of the single-filter detection probabilities as computed by `completefuncs` (i.e., that the completeness functions are separable across filters). This can be a reasonable assumption when you have separate photometric catalogs derived for each filter and you only collate them afterwards, but it is generally not a good assumption for detection algorithms that operate simultaneously on multi-band photometry -- the completeness functions for these types of algorithms are generally not separable.
"""
function model_cmd(mags::AbstractVector{T}, errfuncs, completefuncs, biasfuncs=[zero for i in completefuncs];
                   rng::AbstractRNG=default_rng(), ret_idxs::Bool=false) where T <: AbstractVector{<:Number}
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
    good = findall(map(<=, randsamp, completeness)) # findall( randsamp .<= completeness )
    ret_mags = mags[good] # Get the good mags to be returned.
    # Calculate and add photometric errors, see below
    ret_mags = _model_cmd_loop(rng, ret_mags, errfuncs, biasfuncs)
    if ret_idxs
        return ret_mags, good
    else
        return ret_mags
    end
end
"""
    _model_cmd_loop(rng, ret_mags::AbstractVector{T}, errfuncs, biasfuncs) where {T <: AbstractVector{<:Number}}
    _model_cmd_loop(rng, ret_mags::AbstractVector{SVector{N, T}}, errfuncs, biasfuncs) where {N, T <: Number}

Given a vector of vectors `ret_mags` that contains the magnitudes of stars that pass the completeness criterion,
sample photometric errors from `errfuncs` and photometric biases from `biasfuncs` and return a vector of vectors
of the same size with photometric errors and biases applied. This requires a different implementation for 
`StaticArrays.SVector` and regular `Vector`s, which is why we remove this process from `model_cmd`.
"""
function _model_cmd_loop(rng, ret_mags::AbstractVector{T}, errfuncs, biasfuncs) where {T <: AbstractVector{<:Number}}
    # Since these are nested vectors, mutating the vectors contained in 
    # ret_mags will mutate the original `mags` argument to `model_cmd`, which may be undesirable.
    # We need to use a deepcopy to prevent this, though halves the performance.
    ret_mags = deepcopy(ret_mags)
    for i in eachindex(ret_mags)
        for j in eachindex(errfuncs)
            ret_mags[i][j] += (randn(rng) * errfuncs[j](ret_mags[i][j])) + biasfuncs[j](ret_mags[i][j])
        end
    end
    return ret_mags
end
function _model_cmd_loop(rng, ret_mags::AbstractVector{SVector{N, T}}, errfuncs, biasfuncs) where {N, T <: Number}
    for i in eachindex(ret_mags)
        err_scale = sacollect(SVector{N, T}, errfuncs[j](ret_mags[i][j]) for j in eachindex(errfuncs))
        bias = sacollect(SVector{N, T}, biasfuncs[j](ret_mags[i][j]) for j in eachindex(biasfuncs))
        ret_mags[i] = ret_mags[i] .+ (randn(rng, SVector{N, T}) .* err_scale) .+ bias
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
# Fast way to sum over vector of SVectors
# @benchmark reduce(+,reduce(+,x)) setup=(x=rand(SVector{2,Float64}, 3))
