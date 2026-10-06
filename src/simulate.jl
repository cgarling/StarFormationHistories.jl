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
    (new_mini_vec, new_mags) = ingest_isochrone(mini_vec::AbstractVector, mags::AbstractVector)

Prepares the initial masses `mini_vec` and magnitudes `mags` (one element per isochrone point, either a number or a vector of magnitudes in several filters) of an isochrone for interpolation in initial mass. The arguments are not mutated.

Isochrone points should be provided in evolutionary order (e.g., in order of equivalent evolutionary points), as given by isochrone libraries. Initial mass then increases along the isochrone, but interpolated isochrones can have small decreases in initial mass between evolutionary phases (e.g., from the tip of the red giant branch to the horizontal branch). Sorting by initial mass would interleave the points of the two phases, so instead each decrease is removed by shifting the initial masses of all later points up by the size of the decrease. This preserves the differences in initial mass within each phase, which set the relative numbers of stars in each phase. Repeated initial masses are then separated with `Interpolations.deduplicate_knots!`.

Points in evolutionary order trace the isochrone, while points out of order jump across it. If the path through the magnitudes of consecutive points is more than twice as long in the given order as in order of initial mass, the points are assumed not to be in evolutionary order; a warning is issued and they are sorted by initial mass instead.
"""
function ingest_isochrone(mini_vec::AbstractVector, mags::AbstractVector)
    @argcheck axes(mini_vec) == axes(mags)
    
    # Length of the path through consecutive points `v`, each a number or a vector of magnitudes
    path_length(v::AbstractVector) = sum(sqrt(sum(abs2, v[i+1] - v[i])) for i in firstindex(v):lastindex(v)-1; init=0.0)

    idx = sortperm(mini_vec)
    if idx != eachindex(mini_vec) && path_length(mags) > 2 * path_length(view(mags, idx))
        @warn "The isochrone points do not appear to be in evolutionary order, so they will be sorted by initial mass. Isochrone points should be provided in evolutionary order, as given by isochrone libraries."
        mini_vec, mags = mini_vec[idx], mags[idx]
    end
    m = collect(float(eltype(mini_vec)), mini_vec) # Copy so the caller's vector is not mutated
    offset = zero(eltype(m))
    for i in eachindex(m)[begin+1:end]
        m[i] += offset
        if m[i] < m[i-1]
            offset += m[i-1] - m[i]
            m[i] = m[i-1]
        end
    end
    return deduplicate_knots!(m; move_knots=true), mags
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
    mini_vec, mags = ingest_isochrone(mini_vec, mags)
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
 - [`StarFormationHistories.binary_systems`](@ref), used to construct smooth Hess diagram templates
 - [`StarFormationHistories.with_fraction`](@ref), used to construct Hess diagram templates with a variable binary \
   fraction (e.g., when fitting the binary fraction with [`SSPModel`](@ref StarFormationHistories.SSPModel))
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
    with_fraction(model::AbstractBinaryModel, fraction::Real)
Returns a copy of `model` with its binary system fraction (see [`StarFormationHistories.binary_system_fraction`](@ref)) set to `fraction`, keeping any other properties of `model`."""
function with_fraction end

"""
    NoBinaries()
The `NoBinaries` type indicates that no binaries of any kind should be created. """
struct NoBinaries <: AbstractBinaryModel end
Base.length(::NoBinaries) = 1
binary_system_fraction(::NoBinaries) = 0
binary_number_fraction(::NoBinaries) = 0
binary_mass_fraction(::NoBinaries, imf) = 0
with_fraction(::NoBinaries, fraction::Real) = (@argcheck iszero(fraction) "`NoBinaries` cannot have a nonzero binary fraction."; NoBinaries())

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
with_fraction(::RandomBinaryPairs, fraction::Real) = RandomBinaryPairs(fraction)

"""
    BinaryMassRatio(fraction::Real,
                    qdist::Distributions.ContinuousUnivariateDistribution =
                        Distributions.Uniform(0.1, 1.0))
The `BinaryMassRatio` type takes two arguments; the number fraction of stellar systems that are binaries `0 <= fraction::Real <= 1` and a continuous univariate distribution `qdist` from which to sample binary mass ratios, defined as the ratio of the secondary mass to the primary mass: ``q = \\text{M}_S / \\text{M}_P``. The provided `qdist` must have the proper support of `(minimum(qdist) >= 0) & (maximum(qdist) <= 1)`. Users may find the [`Distributions.truncated`](https://juliastats.org/Distributions.jl/stable/truncate/#Distributions.truncated) method useful for enforcing this support on more general distributions. The default `qdist` is a uniform distribution from 0.1 to 1, which appears to give reasonably good agreement to observations (see, e.g., [Goodwin2013](@cite)).
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
with_fraction(m::BinaryMassRatio, fraction::Real) = BinaryMassRatio(fraction, m.qdist)

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

"""
    (m1, m2, w) = binary_systems(model::StarFormationHistories.AbstractBinaryModel, m::AbstractVector{<:Number}, imf; mean_mass::Number=mean(imf))

Returns a quadrature over the binary systems of `model`, used to construct smooth Hess diagram templates with [`partial_cmd_smooth`](@ref); this is the deterministic counterpart of [`StarFormationHistories.sample_system`](@ref). `m` is a sorted vector of initial masses spanning the stars present in the isochrone. Node `k` is a binary system whose stars have initial masses `m1[k] >= m2[k]`; stars with initial masses outside `extrema(m)` are not in the isochrone (e.g., they have died) and contribute no light. `w[k]` is the expected number of binary systems represented by node `k` per unit stellar mass formed. The nodes cover every binary system with at least one star in the isochrone. `imf` and `mean_mass` are as in [`partial_cmd_smooth`](@ref).

Single-star systems are modeled separately, so concrete subtypes of `AbstractBinaryModel` only need to implement this method (along with [`StarFormationHistories.binary_mass_fraction`](@ref)) to support smooth Hess diagram templates.
"""
function binary_systems end
# Trapezoidal rule weights for integrating over the sorted nodes `m`
trapezoid_weights(m::AbstractVector) = (d = diff(m) ./ 2; [d; zero(eltype(d))] .+ [zero(eltype(d)); d])
function binary_systems(model::RandomBinaryPairs, m::AbstractVector{<:Number}, imf; mean_mass::Number=mean(imf))
    Base.require_one_based_indexing(m)
    a = trapezoid_weights(m) .* dispatch_imf.(imf, m) # IMF probability represented by each node
    b = binary_system_fraction(model)
    nb = b / ((1 + b) * mean_mass) # Number of binary systems per unit stellar mass formed
    n = length(m)
    # Both stars in the isochrone; unordered pairs i <= j, so distinct masses count both orderings
    m1 = [m[j] for i in 1:n for j in i:n]
    m2 = [m[i] for i in 1:n for j in i:n]
    w = [nb * a[i] * a[j] * (i == j ? 1 : 2) for i in 1:n for j in i:n]
    # One star in the isochrone and its companion outside it, in either order; max guards against
    # trapezoidal rule error when the support of `imf` matches the isochrone
    outside = max(0, 1 - sum(a))
    return [m1; fill(eltype(m1)(Inf), n)], [m2; m], [w; (2 * nb * outside) .* a]
end
function binary_systems(model::BinaryMassRatio, m::AbstractVector{<:Number}, imf; mean_mass::Number=mean(imf))
    Base.require_one_based_indexing(m)
    # Equal-width cells in mass ratio q over the support of qdist, with their probabilities from the cdf
    # ponytail: fixed number of q cells; tie to the Hess diagram bin width if templates need finer q sampling
    qedges = range(minimum(model.qdist), maximum(model.qdist); length=51)
    q, pq = midpoints(qedges), diff(cdf.(model.qdist, qedges))
    a = trapezoid_weights(m)
    nb = binary_system_fraction(model) / mean_mass # `imf` is the distribution of system masses
    mmax = maximum(m)
    T = float(eltype(m))
    m1, m2, w = T[], T[], T[]
    for i in eachindex(m, a), l in eachindex(q, pq)
        # Primary in the isochrone; system mass M = m1 (1 + q), so dM = (1 + q) dm1
        M = m[i] * (1 + q[l])
        push!(m1, m[i]); push!(m2, m[i] * q[l]); push!(w, nb * a[i] * pq[l] * (1 + q[l]) * dispatch_imf(imf, M))
        # Primary outside the isochrone and secondary in it; M = m2 (1 + q) / q, so dM = (1 + q) / q dm2
        if m[i] / q[l] > mmax
            push!(m1, m[i] / q[l]); push!(m2, m[i]); push!(w, nb * a[i] * pq[l] * (1 + q[l]) / q[l] * dispatch_imf(imf, M / q[l]))
        end
    end
    return m1, m2, w
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
 - `mini_vec::AbstractVector{<:Number}` contains the initial masses (in solar masses) for the stars in the isochrone. The isochrone points must be in evolutionary order (e.g., in order of equivalent evolutionary points), as given by isochrone libraries; see [`StarFormationHistories.ingest_isochrone`](@ref).
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
    mini_vec, mags = ingest_isochrone(mini_vec, mags) # Make initial masses increase along the isochrone
    # Construct the sampler object for the provided imf; for some distributions, this will return a Distributions.Sampleable for which rand(imf_sampler) is more efficient than rand(imf).
    imf_sampler = sampler(imf) 
    itp = interpolate((mini_vec,), mags, Gridded(Linear()))
    mmin1, mmax = extrema(mini_vec) # Need this to determine validity for mag interpolation.
    # Initial mass below which a system of length(binary_model) stars, none more massive than the primary,
    # must be fainter than `mag_lim`; systems above it are checked against `mag_lim` with their combined flux below.
    # The first call only validates the arguments so errors report the user's `mag_lim`; the second computes the shifted cutoff.
    mass_limits(mini_vec, mags, mag_names, mag_lim, mag_lim_name)
    mmin2, _ = mass_limits(mini_vec, mags, mag_names, mag_lim + 2.5 * log10(length(binary_model)), mag_lim_name)
    sourceidx = findfirst(==(mag_lim_name), mag_names)

    total = zero(eltype(imf))
    mass_vec = Vector{SVector{length(binary_model),eltype(imf)}}(undef,0)
    mag_vec = Vector{eltype(mags)}(undef,0)
    while total < limit
        masses = sample_system(imf_sampler, rng, binary_model) # Sample masses of stars in a single system
        m = sum(masses)
        # Keep the system that crosses `limit` with probability (limit - total) / m so that the
        # expected total mass equals `limit`; early `break` will discard this sample. 
        total + m > limit && rand(rng) * m > limit - total && break
        total += m
        if first(masses) < mmin2 # System would be fainter than `mag_lim` so continue
            continue
        end
        flux = zero(eltype(mags))
        for mass in masses
            if (mass > mmin1) & (mass < mmax) # Mass is in valid interpolation range
                flux += mag2flux.( itp(mass) ) # Think this works for SVectors? not sure
            end
        end
        # Possible that first(masses) > mmax and no valid companions either, meaning sum(flux) == 0
        if (sum(flux) > 0) & (!isfinite(mag_lim) || flux2mag(flux[sourceidx]) < mag_lim)
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
    mini_vec, mags = ingest_isochrone(mini_vec, mags) # Make initial masses increase along the isochrone
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
        # Keep the system that crosses `limit` with probability (limit - total) / m so that the
        # expected total mass equals `limit`; early `break` will discard this sample. 
        total + flux[idxlim] > limit && rand(rng) * flux[idxlim] > limit - total && break
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
    (sampled_masses, sampled_mags) = generate_stars_mass_composite(mini_vec::AbstractVector{<:AbstractVector{<:Number}}, mags::AbstractVector, mag_names::AbstractVector{String}, limit::Number, massfrac::AbstractVector{<:Number}, imf::Sampleable{Univariate,Continuous}; rng::Random.AbstractRNG=Random.default_rng(), kws...)

Generates a random sample of stars with a complex star formation history using multiple isochrones. Very similar to [`generate_stars_mass`](@ref) except the isochrone-related arguments `mini_vec` and `mags` should now be vectors of vectors containing the relevant data for the full set of isochrones to be considered. The total birth stellar mass of the sampled population is given by `limit`. The proportion of this mass allotted to each of the individual isochrones is given by the entries of the `massfrac` vector. This basically just proportions `limit` according to `massfrac` and calls [`generate_stars_mass`](@ref) for each of the individual stellar populations; as such it is set up to multi-thread across the multiple stellar populations. 

# Arguments
 - `mini_vec::AbstractVector{<:AbstractVector{<:Number}}` contains the initial masses (in solar masses) for the stars in each isochrone. The isochrone points must be in evolutionary order (e.g., in order of equivalent evolutionary points), as given by isochrone libraries; see [`StarFormationHistories.ingest_isochrone`](@ref). The length of `mini_vec` should be equal to the number of isochrones. 
 - `mags` contains the absolute magnitudes from the isochrones in the desired filters corresponding to the same stars as provided in `mini_vec`. The length of `mags` should be equal to the number of isochrones. The individual elements of `mags` are each internally interpreted and converted into a standard format by [`StarFormationHistories.ingest_mags`](@ref). The valid formats for the individual elements of `mags` are:
    - `AbstractVector{AbstractVector{<:Number}}`, in which case the length of the vector `length(mags[i])` can either be equal to `length(mini_vec[i])`, in which case the length of the inner vectors must all be equal to the number of filters you are providing, or the length of the outer vector can be equal to the number of filters you are providing, and the length of the inner vectors must all be equal to `length(mini_vec[i])`; this is the more common use-case.
    - `AbstractMatrix{<:Number}`, in which case `mags[i]` must be 2-dimensional. Valid shapes are `size(mags[i]) == (length(mini_vec[i]), nfilters)` or `size(mags[i]) == (nfilters, length(mini_vec[i]))`, with `nfilters` being the number of filters you are providing.
 - `mag_names::AbstractVector{String}` contains strings describing the filters you are providing in `mags`; an example might be `["B","V"]`. These are used when `mag_lim` is finite to determine what filter you want to use to limit the faintest stars you want returned. These are assumed to be the same for all isochrones.
 - `limit::Number` gives the total birth stellar mass of the population you want to sample. 
 - `massfrac::AbstractVector{<:Number}` is a vector giving the relative fraction of mass allotted to each individual stellar population; length must be equal to the length of `mini_vec` and `mags`. 
 - `imf::Distributions.Sampleable{Distributions.Univariate, Distributions.Continuous}` is a sampleable continuous univariate distribution implementing a stellar initial mass function with a defined `rand(rng::Random.AbstractRNG, imf)` method to use for sampling masses. All instances of `Distributions.ContinuousUnivariateDistribution` are also valid. Implementations of commonly used IMFs are available in [InitialMassFunctions.jl](https://github.com/cgarling/InitialMassFunctions.jl).

# Keyword Arguments
 - `rng::Random.AbstractRNG` is the random number generator. Because the populations are sampled in parallel, `rng` is not used to sample stars directly; instead, one seed per population is drawn from `rng`, and each population is sampled with a copy of `rng` seeded with `Random.seed!(copy(rng), seed)`. Results for a given `rng` state are therefore reproducible regardless of the number of threads. `rng` must support `copy` and seeding from an integer, which is true of `Random.default_rng()`, `Random.Xoshiro`, `Random.MersenneTwister`, and `StableRNGs.StableRNG`, among others.
All other keyword arguments `kws...` are passed to [`generate_stars_mass`](@ref); you should refer to that method's documentation for more information.

# Returns
 - `sampled_masses::Vector{Vector{SVector{N,eltype(imf)}}}` is a vector of vectors containing the initial stellar masses of the sampled stars. The outer vectors are separated by the isochrone the stars were generated from; i.e., all of `sampled_masses[1]` were sampled from `mini_vec[1]` and so on. These can be concatenated into a single vector with `reduce(vcat,sampled_masses)`. The format of the contained `StaticArrays.SVector`s are as output from [`sample_system`](@ref); see that method's documentation for more details.
 - `sampled_mags::Vector{Vector{SVector{N,<:Number}}}` is a vector of vectors containing `StaticArrays.SVectors` with the multi-band magnitudes of the sampled stars. The outer vectors are separated by the isochrone the stars were generated from; i.e. all of `sampled_mags[1]` were sampled from `mags[1]` and so on. To get the magnitude of star `i` in band `j` sampled from isochrone `k`, you would do `sampled_mags[k][i][j]`. This can be concatenated into a `Vector{SVector}` with `reduce(vcat,sampled_mags)` and a 2-D `Matrix` with `reduce(hcat,reduce(vcat,sampled_mags))`.
"""
function generate_stars_mass_composite(mini_vec::AbstractVector{T}, mags::AbstractVector, mag_names::AbstractVector{String}, limit::Number, massfrac::AbstractVector{<:Number}, imf::Sampleable{Univariate,Continuous}; binary_model::AbstractBinaryModel=RandomBinaryPairs(0.3), rng::AbstractRNG=default_rng(), kws...) where T <: AbstractVector{<:Number}
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
    # Most RNG objects are not thread-safe, so `rng` cannot be shared across the threaded loop.
    # Instead, draw one seed per population from `rng` serially and sample each population with
    # its own seeded copy of `rng`, which also makes results independent of the number of threads.
    seeds = rand(rng, UInt64, ncomposite)
    # One task per population for better load balancing between pops of different masses
    @sync for i in eachindex(mini_vec, mags, massfrac)
        Threads.@spawn massvec[i], mag_vec[i] = generate_stars_mass(mini_vec[i], mags[i], mag_names, limit * massfrac[i], imf; binary_model, rng=seed!(copy(rng), seeds[i]), kws...)
    end
    return massvec, mag_vec
end

"""
    (sampled_masses, sampled_mags) = generate_stars_mag_composite(mini_vec::AbstractVector{<:AbstractVector{<:Number}}, mags::AbstractVector, mag_names::AbstractVector{String}, absmag::Number, absmag_name::String, fracs::AbstractVector{<:Number}, imf::Sampleable{Univariate,Continuous}; frac_type::Symbol=:lum, rng::Random.AbstractRNG=Random.default_rng(), kws...)

Generates a random sample of stars with a complex star formation history using multiple isochrones. Very similar to [`generate_stars_mag`](@ref) except the isochrone-related arguments `mini_vec` and `mags` should now be vectors of vectors containing the relevant data for the full set of isochrones to be considered. The total absolute magnitude of the sampled population is given by `absmag`. The proportion of the luminosity allotted to each of the individual isochrones is given by the entries of the `frac` vector. This basically just proportions the luminosity according to `frac` and calls [`generate_stars_mag`](@ref) for each of the individual stellar populations; as such it is set up to multi-thread across the multiple stellar populations. 

# Arguments
 - `mini_vec::AbstractVector{<:AbstractVector{<:Number}}` contains the initial masses (in solar masses) for the stars in each isochrone. The isochrone points must be in evolutionary order (e.g., in order of equivalent evolutionary points), as given by isochrone libraries; see [`StarFormationHistories.ingest_isochrone`](@ref). The length of `mini_vec` should be equal to the number of isochrones. 
 - `mags` contains the absolute magnitudes from the isochrones in the desired filters corresponding to the same stars as provided in `mini_vec`. The length of `mags` should be equal to the number of isochrones. The individual elements of `mags` are each internally interpreted and converted into a standard format by [`StarFormationHistories.ingest_mags`](@ref). The valid formats for the individual elements of `mags` are:
    - `AbstractVector{AbstractVector{<:Number}}`, in which case the length of the vector `length(mags[i])` can either be equal to `length(mini_vec[i])`, in which case the length of the inner vectors must all be equal to the number of filters you are providing, or the length of the outer vector can be equal to the number of filters you are providing, and the length of the inner vectors must all be equal to `length(mini_vec[i])`; this is the more common use-case.
    - `AbstractMatrix{<:Number}`, in which case `mags[i]` must be 2-dimensional. Valid shapes are `size(mags[i]) == (length(mini_vec[i]), nfilters)` or `size(mags[i]) == (nfilters, length(mini_vec[i]))`, with `nfilters` being the number of filters you are providing.
 - `mag_names::AbstractVector{String}` contains strings describing the filters you are providing in `mags`; an example might be `["B","V"]`. These are used when `mag_lim` is finite to determine what filter you want to use to limit the faintest stars you want returned. These are assumed to be the same for all isochrones.
 - `absmag::Number` gives the total absolute magnitude of the complex population to be sampled.
 - `absmag_name::String` is the name of the filter for which the desired absolute magnitude is `absmag`; must be contained in `mag_names`.
 - `fracs::AbstractVector{<:Number}` is a vector giving the relative fraction of luminosity or mass (determined by the `frac_type` keyword argument) allotted to each individual stellar population; length must be equal to the length of `mini_vec` and `mags`. 
 - `imf::Distributions.Sampleable{Distributions.Univariate, Distributions.Continuous}` is a sampleable continuous univariate distribution implementing a stellar initial mass function with a defined `rand(rng::Random.AbstractRNG, imf)` method to use for sampling masses. All instances of `Distributions.ContinuousUnivariateDistribution` are also valid. Implementations of commonly used IMFs are available in [InitialMassFunctions.jl](https://github.com/cgarling/InitialMassFunctions.jl).

# Keyword Arguments
 - `frac_type::Symbol` either `:lum`, in which case `fracs` is assumed to contain the relative luminosity fractions for each individual isochrone, or `:mass`, in which case it is assumed that `fracs` contains birth stellar mass fractions. Mass fractions are converted to luminosity fractions in the `absmag_name` filter using [`StarFormationHistories.luminosity_per_mass`](@ref).
 - `rng::Random.AbstractRNG` is the random number generator. As in [`generate_stars_mass_composite`](@ref), one seed per population is drawn from `rng` and each population is sampled with a seeded copy of `rng`, so results for a given `rng` state are reproducible regardless of the number of threads. `rng` must support `copy` and seeding from an integer.
All other keyword arguments `kws...` are passed to [`generate_stars_mag`](@ref); you should refer to that method's documentation for more information.

# Returns
 - `sampled_masses::Vector{Vector{SVector{N,eltype(imf)}}}` is a vector of vectors containing the initial stellar masses of the sampled stars. The outer vectors are separated by the isochrone the stars were generated from; i.e., all of `sampled_masses[1]` were sampled from `mini_vec[1]` and so on. These can be concatenated into a single vector with `reduce(vcat,sampled_masses)`. The format of the contained `StaticArrays.SVector`s are as output from [`sample_system`](@ref); see that method's documentation for more details. 
 - `sampled_mags::Vector{Vector{SVector{N,<:Number}}}` is a vector of vectors containing `StaticArrays.SVectors` with the multi-band magnitudes of the sampled stars. The outer vectors are separated by the isochrone the stars were generated from; i.e. all of `sampled_mags[1]` were sampled from `mags[1]` and so on. To get the magnitude of star `i` in band `j` sampled from isochrone `k`, you would do `sampled_mags[k][i][j]`. This can be concatenated into a `Vector{SVector}` with `reduce(vcat,sampled_mags)` and a 2-D `Matrix` with `reduce(hcat,reduce(vcat,sampled_mags))`. 
"""
function generate_stars_mag_composite(mini_vec::AbstractVector{T}, mags::AbstractVector, mag_names::AbstractVector{String}, absmag::Number, absmag_name::String, fracs::AbstractVector{<:Number}, imf::Sampleable{Univariate,Continuous}; frac_type::Symbol=:lum, binary_model::AbstractBinaryModel=RandomBinaryPairs(0.3), rng::AbstractRNG=default_rng(), kws...) where T <: AbstractVector{<:Number}
    !(axes(mini_vec,1) == axes(mags,1) == axes(fracs,1)) && throw(ArgumentError("The arguments `mini_vec`, `mags`, and `fracs` to `generate_stars_mag_composite` must all have equal length and identical indexing."))
    ncomposite = length(mini_vec) # Number of stellar populations provided.
    # Validate before the threaded loop below, where errors would be wrapped in a CompositeException
    idx = findfirst(==(absmag_name), mag_names)
    isnothing(idx) && throw(ArgumentError("Provided `absmag_name` is not contained in provided `mag_names` array."))
    # Interpret whether user requests `fracs` represent luminosity or mass fractions.
    if frac_type == :mass
        # Convert mass fractions to luminosity fractions in the `absmag_name` filter. The 1 / mean_mass
        # `mean_mass` normalization is common to all populations and cancels when fracs is normalized below.
        fracs = fracs .* [luminosity_per_mass(mini_vec[i], mags[i], imf; mean_mass=1)[idx] for i in eachindex(mini_vec, mags)]
    elseif frac_type != :lum
        throw(ArgumentError("Supported `frac_type` arguments for generate_stars_mag_composite are `:lum` or `:mass`."))
    end
    fracs = fracs ./ sum(fracs) # Ensure fracs is normalized to sum to 1.
    limit = mag2flux(absmag)    # Convert the provided `limit` from magnitudes into flux.
    fracs = flux2mag.(fracs .* limit)
    # Allocate output vectors.
    massvec = [ Vector{SVector{length(binary_model),eltype(imf)}}(undef,0) for i in 1:ncomposite ]
    # Need to ingest here so we know what type of SVector we're going to be putting into mag_vec. 
    mags = [ ingest_mags(mini_vec[i], mags[i]) for i in eachindex( mini_vec, mags ) ]
    mag_vec = [ Vector{eltype(i)}(undef,0) for i in mags ]
    # Sample each population with its own seeded copy of `rng`; see generate_stars_mass_composite
    seeds = rand(rng, UInt64, ncomposite)
    # One task per population for better load balancing between pops of different magnitudes
    @sync for i in eachindex(mini_vec, mags, fracs)
        Threads.@spawn massvec[i], mag_vec[i] = generate_stars_mag(mini_vec[i], mags[i], mag_names, fracs[i], absmag_name, imf; binary_model, rng=seed!(copy(rng), seeds[i]), kws...)
    end
    return massvec, mag_vec
end

"""
    luminosity_per_mass(mini_vec::AbstractVector{<:Number}, mags, imf; mean_mass::Number=mean(imf))

Returns the expected present-day luminosity per unit birth stellar mass of the stellar population defined by the isochrone initial masses `mini_vec` and absolute magnitudes `mags` for the provided `imf`, as a `StaticArrays.SVector` with one entry per filter. `mags` accepts the same formats as [`generate_stars_mass`](@ref). As there, the isochrone points must be in evolutionary order.

```math
\\frac{L_i}{M_*} = \\frac{1}{\\langle m \\rangle} \\int_{m_\\text{min}}^{m_\\text{max}} \\phi(m) \\, 10^{-0.4 \\, M_i(m)} \\, dm
```

where ``\\phi(m)`` is the IMF probability density, ``M_i(m)`` is the absolute magnitude in filter ``i`` along the isochrone at initial mass ``m``, ``\\langle m \\rangle`` is `mean_mass`, and ``m_\\text{min}`` and ``m_\\text{max}`` are `extrema(mini_vec)`.

Luminosities are returned as `10^(-0.4 M)` for absolute magnitude `M`, i.e., in units of the luminosity of a source with absolute magnitude zero in the same filter and photometric system (the absolute-magnitude analog of maggies). These are the units returned by [`StarFormationHistories.mag2flux`](@ref) with its default zeropoint, so a population with total birth stellar mass `M_*` has expected integrated absolute magnitude `flux2mag(M_* * luminosity_per_mass(mini_vec, mags, imf)[i])` in filter `i`. To convert to solar luminosities, multiply by `10^(0.4 M_⊙)`, where `M_⊙` is the Sun's absolute magnitude in that filter.

As in [`partial_cmd_smooth`](@ref), `imf` can be a `Distributions.ContinuousUnivariateDistribution` or any callable returning the IMF probability density at a given mass (see `StarFormationHistories.dispatch_imf`). `mean_mass` is the mean initial stellar mass of `imf`; it is computed for you if `imf` is a `Distributions.Distribution`, but must be provided otherwise.

The integral is computed with the trapezoidal rule over the isochrone points, so only stars with initial masses in `extrema(mini_vec)` contribute; stars more massive than the end of the isochrone are assumed to be dead. For binary models that pair stars independently drawn from the IMF (e.g., [`RandomBinaryPairs`](@ref StarFormationHistories.RandomBinaryPairs)), the expected luminosity per unit mass is unchanged by the binary model. For [`BinaryMassRatio`](@ref StarFormationHistories.BinaryMassRatio) this is an approximation.
"""
function luminosity_per_mass(mini_vec::AbstractVector{<:Number}, mags, imf; mean_mass::Number=mean(imf))
    mini_vec, mags = ingest_isochrone(mini_vec, ingest_mags(mini_vec, mags))
    f = [dispatch_imf(imf, mini_vec[i]) .* mag2flux.(mags[i]) for i in eachindex(mini_vec, mags)]
    return sum((mini_vec[i+1] - mini_vec[i]) .* (f[i] .+ f[i+1]) ./ 2 for i in eachindex(mini_vec)[begin:end-1]) ./ mean_mass
end

"""
    mass_from_absmag(mini_vec::AbstractVector{<:AbstractVector{<:Number}}, mags::AbstractVector, mag_names::AbstractVector{String}, absmag::Number, absmag_name::String, massfrac::AbstractVector{<:Number}, imf; mean_mass::Number=mean(imf))

Returns the expected total birth stellar mass of a complex stellar population whose integrated absolute magnitude in the filter `absmag_name` is `absmag`. The population consists of the isochrones given by `mini_vec` and `mags` with relative birth stellar mass fractions `massfrac`, which need not be normalized. Arguments are as in [`generate_stars_mag_composite`](@ref) with `frac_type=:mass`, and `imf` and `mean_mass` are as in [`StarFormationHistories.luminosity_per_mass`](@ref), which is used to compute the luminosity per unit birth stellar mass of each isochrone.

```math
M_* = \\frac{\\langle m \\rangle \\, 10^{-0.4 \\, M} \\sum_j x_j}{\\sum_j x_j \\int_{m_{\\text{min},j}}^{m_{\\text{max},j}} \\phi(m) \\, 10^{-0.4 \\, M_j(m)} \\, dm}
```

where ``M`` is `absmag`, ``x_j`` is `massfrac[j]`, ``M_j(m)`` is the absolute magnitude in filter `absmag_name` along isochrone ``j`` at initial mass ``m``, ``m_{\\text{min},j}`` and ``m_{\\text{max},j}`` are `extrema(mini_vec[j])`, and ``\\phi(m)`` and ``\\langle m \\rangle`` are as in [`StarFormationHistories.luminosity_per_mass`](@ref). Equivalently, ``M_*`` is the mass for which ``M_* \\sum_j (x_j / \\sum_k x_k) (L/M_*)_j = 10^{-0.4 \\, M}``, where ``(L/M_*)_j`` is the luminosity per unit birth stellar mass of isochrone ``j``.

This is the expected mass; a population sampled to `absmag` with [`generate_stars_mag_composite`](@ref) will scatter around it due to stochastic sampling of the IMF.
"""
function mass_from_absmag(mini_vec::AbstractVector{<:AbstractVector{<:Number}}, mags::AbstractVector, mag_names::AbstractVector{String}, absmag::Number, absmag_name::String, massfrac::AbstractVector{<:Number}, imf; mean_mass::Number=mean(imf))
    @argcheck axes(mini_vec, 1) == axes(mags, 1) == axes(massfrac, 1) "The arguments `mini_vec`, `mags`, and `massfrac` must all have equal length and identical indexing."
    idx = findfirst(==(absmag_name), mag_names)
    isnothing(idx) && throw(ArgumentError("Provided `absmag_name` is not contained in provided `mag_names` array."))
    L = [luminosity_per_mass(mini_vec[i], mags[i], imf; mean_mass)[idx] for i in eachindex(mini_vec, mags)]
    # Total flux = M * Σ (x_i / Σx) L_i; solve for the total birth mass M
    return mag2flux(absmag) * sum(massfrac) / sum(massfrac .* L)
end

"""
    result::Vector{NamedTuple} = add_metadata(starcat, mag_symbols; kws...)
Processes output star catalogs from the composite `generate_stars...` methods that create complex stellar populations into `Vector{NamedTuple}` with added metadata which can then be used to construct tables (e.g., `TypedTables.Table` or `DataFrames.DataFrame`).

# Arguments
 - `starcat`: output from [`generate_stars_mass_composite`](@ref) or [`generate_stars_mag_composite`](@ref).
 - `mag_symbols`: the list of magnitude names to use when unpacking the magnitudes in `starcat`.

# Keyword Arguments
All keyword arguments must have length equal to `length(starcat[1])` and `length(starcat[2])`; that is, equal to the number of SSPs used when constructing the mock catalog. This metadata will then be added to the output result. For example, if the `log10(age [yr])` of the SSPs used are in a vector `ages` and the metallicities in a vector `MH`, you can call

```julia
julia> starcat = generate_stars_mass_composite(...)

julia> filters = (:F475W, :F814W) # Names of filters to associate with the magnitudes in `starcat[2]`

julia> add_metadata(starcat, filters, logAge = ages, MH = MH)
5866911-element Vector{@NamedTuple{mini::Float64, F475W::Float64, F814W::Float64, logAge::Float64, MH::Float64}}:
 (mini = 0.25293134503128467, F475W = 33.53553000158083, F814W = 31.415817763596586, logAge = 6.6, MH = -2.4)
[...]
```

# Initial Masses
The initial masses of each sampled system are split into scalar fields. `mini` is always the initial mass of the primary star. When `starcat` was sampled with a binary model (e.g., [`RandomBinaryPairs`](@ref StarFormationHistories.RandomBinaryPairs)), the companion initial mass is given in `mini2`, which is zero for systems without a companion, so `mini + mini2` is the initial mass of the system.
"""
function add_metadata(starcat, mag_symbols::NTuple{N, Symbol}; kws...) where N
    @argcheck length(starcat[1]) == length(starcat[2]) "Length of `starcat[1]` not equal to length of `starcat[2]`."
    mini, mags = starcat
    # Shapes come from the StaticArrays.SVector element types so that catalogs with no sampled stars work
    mini_T, mag_T = eltype(eltype(mini)), eltype(eltype(mags))
    @argcheck (mini_T <: SVector) && (mag_T <: SVector) "Dimensions of input `starcat` incorrect. `add_metadata` only works on `starcat` constructed with multiple SSPs."
    @argcheck N == length(mag_T) "Length of `mag_symbols` must match length of magnitude vectors (i.e., `length(mag_symbols) == length(starcat[2][1][1])`."
    nSSP = length(starcat[1]) # Number of SSP models
    for (k, v) in kws
        @argcheck (length(v) == nSSP) "Length of keyword argument $k is incorrect; must be equal to `length(starcat[1])` and `length(starcat[2])`."
    end

    # Pre-compute the NamedTuple type for performance
    # Each system's initial masses (primary first) are split into scalar columns mini, mini2, ..., miniK
    # so the result can be written by any table writer
    K = length(mini_T)
    mass_symbols = (:mini, ntuple(k -> Symbol(:mini, k + 1), K - 1)...)
    mass_types = ntuple(_ -> eltype(mini_T), K)
    mag_scalar_type = eltype(mag_T)
    kw_values = [values(kws)...]  # Convert to vector for easier indexing
    kw_types = isempty(kw_values) ? () : Tuple{(eltype(val) for val in kw_values)...} # eltype.(kw_values)

    # Build tuple type: (mini, mini2, ..., miniK, mag1, mag2, ..., magN, kw1, kw2, ...)
    mag_types = ntuple(_ -> mag_scalar_type, N)
    symbols = (mass_symbols..., mag_symbols...)
    T = Tuple{mass_types..., mag_types...}
    S = NamedTuple{symbols, T}
    result = _rows(S, mini, mags)
    if !isempty(kw_values)
        kw_T = NamedTuple{keys(kws), kw_types}
        kw_rows = _kwrows(kw_T, mini, mags, kw_values)
        result = merge.(result, kw_rows)
    end
    return result
end
# Support general mag_symbols
function add_metadata(starcat, mag_symbols; kws...)
    return add_metadata(starcat, tuple((Symbol(i) for i in mag_symbols)...); kws...)
end
# Helper function to create rows with pre-computed type
function _rows(::Type{NT}, mini, mags) where NT
    # nSSP = length(mini)
    total = sum(length, mini)
    result = Vector{NT}(undef, total)
    idx = 1
    @inbounds for i in eachindex(mini)
        ni = length(mini[i])
        for j in 1:ni
            vals = (mini[i][j]..., mags[i][j]...)
            result[idx] = NT(vals)
            idx += 1
        end
    end
    return result
end
function _kwrows(::Type{NT}, mini, mags, kw_values) where NT
    total = sum(length, mini)
    result = Vector{NT}(undef, total)
    idx = 1
    @inbounds for i in eachindex(mini)
        ni = length(mini[i])
        kw_tup = NT((kw[i] for kw in kw_values))
        for _ in 1:ni
            # result[idx] = NT((kw[i] for kw in kw_values))
            result[idx] = kw_tup
            idx += 1
        end
    end
    return result
end
# _kwrows(::Type{NT}, mini, mags, kw_values) where NT = [NT((kw[i] for kw in kw_values)) for i in eachindex(mini, mags) for j in eachindex(mini[i])]
# _rows(::Type{NT}, mini, mags, kw_values) where NT = [NT((mini[i][j], mags[i][j]..., (kw[i] for kw in kw_values)...)) for i in eachindex(mini, mags) for j in eachindex(mini[i])]


###############################################
#### Functions for modelling observational effects
"""
    new_mags [, good_idxs] = model_cmd(mags::AbstractVector{<:AbstractVector{<:Number}}, 
                                       errfuncs, 
                                       completefuncs,
                                       biasfuncs = [zero for i in first(mags)]; 
                                       rng::Random.AbstractRNG = Random.default_rng(), 
                                       ret_idxs::Bool = false)

Simple method for modeling photometric error and incompleteness to "mock observe" a pure catalog of stellar photometry, such as those produced by [`generate_stars_mass`](@ref) and [`generate_stars_mag`](@ref). This method assumes Gaussian photometric errors that are uncorrelated between filters. Each of `errfuncs`, `completefuncs`, and `biasfuncs` may be either an iterable of single-filter callables as described below, or a single joint callable taking one magnitude per filter, as described in the "Joint Photometric Models" section of [`partial_cmd_smooth`](@ref); for example, `completefuncs(mags[i]...)` then gives the detection probability of star `i`, and `errfuncs(mags[i]...)` gives its photometric error in each filter. 

# Arguments
 - `mags::AbstractVector{<:AbstractVector{<:Number}}`: a vector of vectors giving the magnitudes of each star to be modeled. The first index is the per-star index and the second index is the per-filter index (so `mags[10][2]` would give the magnitude of the tenth star in the second filter). This is the same format as the magnitudes returned by [`generate_stars_mass`](@ref) and [`generate_stars_mag`](@ref); to use output from the composite versions, you must first `reduce(vcat,mags)` before passing to this function.
 - `errfuncs`: an iterable (typically a vector or tuple) of callables (typically functions or interpolators) with length equal to the number of filters contained in the elements of `mags`. This iterable must contain callables that, when called with the associated magnitudes from `mags`, will return the expected 1-σ photometric error at that magnitude. The organization is such that the photometric error for star `i` in band `j` is `σ_ij = errfuncs[j](mags[i][j])`. 
 - `completefuncs`: an iterable (typically a vector or tuple) of callables (typically functions or interpolators) with length equal to the number of filters contained in the elements of `mags`. This iterable must contain callables that, when called with the associated magnitudes from `mags`, will return the probability that a star with that magnitude in that band will be found in your color-magnitude diagram (this should include the original detection probability and any post-detection quality, morphology, or other cuts). The organization is such that the detection probability for star `i` in band `j` is `c_ij = completefuncs[j](mags[i][j])`.
 - `biasfuncs = [zero for i in first(mags)]`: an iterable (typically a vector or tuple) of callables (typically functions or interpolators) with length equal to the number of filters contained in the elements of `mags`. This iterable must contain callables that, when called with the associated magnitudes from `mags`, will return the expectation value of the photometric bias at that magnitude. We define the photometric bias as the typical difference between measured and input magnitudes; i.e., `⟨measured - input⟩`. This definition matches that of the bias functions returned by [`process_ASTs`](@ref)). By default this method assumes no photometric bias.

# Keyword Arguments
 - `rng::AbstractRNG = Random.default_rng()`: The object to use for random number generation.
 - `ret_idxs::Bool`: whether to return the indices of the input `mags` for the stars that were successfully "observed" and are represented in the output `new_mags`.

# Returns
 - `new_mags`: an object similar to `mags` (i.e., a `Vector{Vector{<:Number}}`, `Vector{SVector{N,<:Number}}`, etc.) containing the magnitudes of the mock-observed stars. This will be shorter than the provided `mags` vector as we are modeling photometric incompleteness, and the magnitudes will also have random photometric errors added to them. This can be reinterpreted as a 2-dimensional `Matrix` with `reduce(hcat,new_mags)`.
 - `good_idxs`: if `ret_idxs` is `true`, the vector of indices into the input `mags` for the stars that were successfully "observed" and are represented in the output `new_mags`.

# Notes
 - This is a simple implementation that seeks to show a simple example of how one can post-process catalogs of "pure" stars from methods like [`generate_stars_mass`](@ref) and [`generate_stars_mag`](@ref) to include observational effects. This method assumes Gaussian photometric errors, which may not, in general, be accurate. When `completefuncs` contains single-filter callables, it also assumes that the total detection probability can be modeled as the product of the single-filter detection probabilities (i.e., that the completeness functions are separable across filters). This can be a reasonable assumption when you have separate photometric catalogs derived for each filter and you only collate them afterwards, but it is generally not a good assumption for detection algorithms that operate simultaneously on multi-band photometry or for catalogs with selection criteria applied in multiple filters -- the completeness functions for these cases are generally not separable and are better represented by a joint `completefuncs`.
"""
function model_cmd(mags::AbstractVector{T}, errfuncs, completefuncs, biasfuncs=[zero for i in first(mags)];
                   rng::AbstractRNG=default_rng(), ret_idxs::Bool=false) where T <: AbstractVector{<:Number}
    nstars = length(mags)
    nfilters = length(first(mags))
    all(length(f) == nfilters for f in (errfuncs, completefuncs, biasfuncs) if f isa Union{Tuple, AbstractVector}) || throw(ArgumentError("Per-filter callables passed to `StarFormationHistories.model_cmd` in `errfuncs`, `completefuncs`, or `biasfuncs` must have one callable per filter, i.e., length equal to `length(first(mags))`."))
    randsamp = rand(rng, nstars) # Draw nstars random uniform variates for completeness testing.
    cols = [getindex.(mags, j) for j in 1:nfilters] # One magnitude vector per filter
    # Detection probability; the product of the single-band completeness values for per-filter models
    completeness = completeness_values(completefuncs, cols)
    # Pick out the entries where the random number is less than the detection probability
    good = findall(map(<=, randsamp, completeness)) # findall( randsamp .<= completeness )
    ret_mags = mags[good] # Get the good mags to be returned.
    # Calculate and add photometric errors and biases, see below
    good_cols = [c[good] for c in cols]
    ret_mags = _model_cmd_loop(rng, ret_mags, filter_values(errfuncs, good_cols), filter_values(biasfuncs, good_cols))
    if ret_idxs
        return ret_mags, good
    else
        return ret_mags
    end
end
"""
    _model_cmd_loop(rng, ret_mags::AbstractVector{T}, err, bias) where {T <: AbstractVector{<:Number}}
    _model_cmd_loop(rng, ret_mags::AbstractVector{SVector{N, T}}, err, bias) where {N, T <: Number}

Given a vector of vectors `ret_mags` that contains the magnitudes of stars that pass the completeness criterion,
sample Gaussian photometric errors with standard deviations `err[j][i]` and add photometric biases `bias[j][i]`
for star `i` in filter `j`, and return a vector of vectors of the same size with photometric errors and biases applied. This requires a different implementation for 
`StaticArrays.SVector` and regular `Vector`s, which is why we remove this process from `model_cmd`.
"""
function _model_cmd_loop(rng, ret_mags::AbstractVector{T}, err, bias) where {T <: AbstractVector{<:Number}}
    # Since these are nested vectors, mutating the vectors contained in 
    # ret_mags will mutate the original `mags` argument to `model_cmd`, which may be undesirable.
    # We need to use a deepcopy to prevent this, though halves the performance.
    ret_mags = deepcopy(ret_mags)
    for i in eachindex(ret_mags)
        for j in eachindex(err)
            ret_mags[i][j] += (randn(rng) * err[j][i]) + bias[j][i]
        end
    end
    return ret_mags
end
function _model_cmd_loop(rng, ret_mags::AbstractVector{SVector{N, T}}, err, bias) where {N, T <: Number}
    for i in eachindex(ret_mags)
        err_scale = sacollect(SVector{N, T}, err[j][i] for j in 1:N)
        bias_i = sacollect(SVector{N, T}, bias[j][i] for j in 1:N)
        ret_mags[i] = ret_mags[i] .+ (randn(rng, SVector{N, T}) .* err_scale) .+ bias_i
    end
    return ret_mags
end

"""
    surviving_fraction(massvec::AbstractVector{<:AbstractVector}, coeffs::AbstractVector, imf)
Returns the fraction of stars that were born that are still undergoing fusion (i.e., have not yet ended their lives as white dwarfs, neutron stars, or black holes).

# Arguments
 - `massvec::AbstractVector{<:AbstractVector}`: a vector of vectors containing the initial masses (in solar masses) of the stars in each stellar population. *This must contain stellar masses for stars that have not yet died -- remnants must have been removed.* The length of `massvec` should be equal to the number of stellar populations.
 - `coeffs::AbstractVector`: a vector of coefficients giving the relative contribution of each stellar population to the total. The length of `coeffs` must be equal to the length of `massvec`.
 - `imf`: a `Distributions.Sampleable{Distributions.Univariate, Distributions.Continuous}` implementing a stellar initial mass function with a defined `cdf` methods. Implementations of commonly used IMFs are available in InitialMassFunctions.jl.
"""
function surviving_fraction(massvec::AbstractVector{<:AbstractVector}, coeffs::AbstractVector, imf)
    @argcheck length(massvec) == length(coeffs) "Length of `massvec` and `coeffs` must be equal."
    tot = sum(coeffs)
    surviving = zero(eltype(coeffs))
    for i in eachindex(massvec, coeffs)
        ext = extrema(massvec[i])
        frac = cdf(imf, ext[2]) - cdf(imf, ext[1])
        surviving += coeffs[i] * frac
    end
    return surviving / tot
end
"""
    recycling_fraction(massvec::AbstractVector{<:AbstractVector}, coeffs::AbstractVector, imf)
Returns the fraction of stars that were born that have ended fusion (i.e., have become white dwarfs, neutron stars, or black holes). Arguments are as in [`surviving_fraction`](@ref StarFormationHistories.surviving_fraction). By definition this is equal to `1 - surviving_fraction(args...)`.
"""
recycling_fraction(massvec::AbstractVector{<:AbstractVector}, coeffs::AbstractVector, imf) = 1 - surviving_fraction(massvec, coeffs, imf)
"""
    surviving_mass_fraction(massvec::AbstractVector{<:AbstractVector}, coeffs::AbstractVector, imf)
Returns the fraction of the birth stellar mass in stars that are still undergoing fusion (i.e., have not yet ended their lives as white dwarfs, neutron stars, or black holes). Arguments are as in [`surviving_fraction`](@ref StarFormationHistories.surviving_fraction).
"""
function surviving_mass_fraction(massvec::AbstractVector{<:AbstractVector}, coeffs::AbstractVector, imf)
    @argcheck length(massvec) == length(coeffs) "Length of `massvec` and `coeffs` must be equal."
    # Basically just compute mean mass of surviving stars and divide by mean birth mass of stars
    # to get mass fraction surviving. Multiply by coeffs[i] and sum over i and compare to sum(coeffs).
    return sum(trapz(massvec[i], massvec[i] .* pdf.(Ref(imf), massvec[i])) * coeffs[i] for i in eachindex(massvec, coeffs)) / sum(coeffs) / mean(imf)
end
"""
    recycling_mass_fraction(massvec::AbstractVector{<:AbstractVector}, coeffs::AbstractVector, imf)
Returns the fraction of the birth stellar mass that was in stars that have ended their lives (i.e., have become white dwarfs, neutron stars, or black holes). Arguments are as in [`surviving_fraction`](@ref StarFormationHistories.surviving_fraction). By definition this is equal to `1 - surviving_mass_fraction(args...)`.
"""
recycling_mass_fraction(massvec::AbstractVector{<:AbstractVector}, coeffs::AbstractVector, imf) = 1 - surviving_mass_fraction(massvec, coeffs, imf)
# In Julia 1.9 we should just be able to do stack(v). 
# eltocols(v::Vector{SVector{dim, T}}) where {dim, T} = reshape(reinterpret(T, v), dim, :)
# eltorows(v::Vector{SVector{dim, T}}) where {dim, T} = reshape(reinterpret(T, v), :, dim)
# eltocols(v::Vector{Vector{SVector{dim, T}}}) where {dim, T} = mapreduce(eltocols, hcat, v)
# eltocols(v::Vector{Vector{SVector{dim, T}}}) where {dim, T} = reduce(hcat, eltocols(i) for i in v)
# eltocols(v::Vector{Vector{SVector{dim, T}}}) where {dim, T} = reduce(hcat, eltocols.(v))
# eltocols(v::Vector{Vector{SVector{dim, T}}}) where {dim, T} = hcat(eltocols.(v)...)
# Fast way to sum over vector of SVectors
# @benchmark reduce(+,reduce(+,x)) setup=(x=rand(SVector{2,Float64}, 3))
