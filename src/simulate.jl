#### Distance Utilities
"""
    distance_modulus(distance)
Finds distance modulus for distance in parsecs. 
"""
distance_modulus(distance) =  5*log10(distance/10)
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
pc_to_arcsec(pc, dist_mod) = rad2deg( tan( pc/exp10(dist_mod/5 + 1)))*3600
"""
    angular_transformation_distance(angle, distance0, distance1)
Transforms an angular separation in arcseconds at distance `distance0` in parsecs to another distance `distance1` in parsecs. Uses the small angle approximation. 
"""
function angular_transformation_distance(angle, distance0, distance1)
    pc0 = arcsec_to_pc(angle,distance_modulus(distance0))
    return pc_to_arcsec(pc0,distance_modulus(distance1))
end

#### Luminosity Utilities
L_from_MV( absmagv ) = exp10(0.4 * (4.8 - absmagv))
MV_from_L(lum) = 4.8 - 2.5 * log10(lum)
""" Luminosity in watts. """
M_bol_from_L(lum) = 71.1974 - 2.5 * log10(lum)
""" Returns watts. """
L_from_M_bol( absbolmag ) = exp10((71.1974-absbolmag)/2.5)
"""
    find_Mv_flat_mu(μ, area, dist_mod)
Given a constant surface brightness `μ`, an angular area `area`, and a distance modulus `dist_mod`, returns the magnitude of the feature. 
"""
function find_Mv_flat_mu(μ, area, dist_mod)
    L = L_from_MV(μ-dist_mod)
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

function ingest_mags(mini_vec::Vector{<:Real}, mags::Matrix{<:Real})
    shape = size(mags)
    if shape[1] == length(mini_vec)
        return collect(eachrow(mags)) # eachrow returns views which are fast but annoying.
        # return [mags[i,:] for i in 1:length(mini_vec)]
    elseif shape[2] == length(mini_vec)
        return collect(eachcol(mags)) # eachrow returns views which are fast but annoying.
        # return [mags[:,i] for i in 1:length(mini_vec)]
    else
        throw(ArgumentError("generate_mock_stars received a misshapen `mags` argument. When providing a Matrix{<:Real}, then it should be 2-dimensional and have size of NxM or MxN, where N is the number of elements in mini_vec, and M is the number of filters represented in the `mags` argument."))
    end
end
# Should improve the shape checks on `mags` for this method.
function ingest_mags(mini_vec::Vector{<:Real}, mags::Vector{T}) where T <: Vector{<:Real}
    # Commonly `mini_vec` will be a vector of length `N`, but `mags` will be a length `M` vector of length `N` vectors. E.g., if length(mini_vec) == 100, and we have two filters, then `mags` will be a vector of 2 vectors, each with length 100. The interpolation routine requires `mags` to be a vector of 100 vectors, each with length 2.
    if length(mags) != length(mini_vec)
        if length(mags[1]) != length(mini_vec) # If the above is false, then this should be true; otherwise error. 
            throw(ArgumentError("generate_mock_stars received a misshapen `mags` argument. When providing a Vector{Vector{<:Real}}, then it should either have a shape of NxM or MxN, where N is the number of elements in mini_vec, and M is the number of filters represented in the `mags` argument."))
        else
            return collect(eachrow(hcat(mags...))) # eachrow returns views which are fast but annoying. 
            # mat = hcat(mags...)
            # return [mat[i,:] for i in 1:length(mini_vec)]
            # mags = [[mags[i][j]] for j=eachindex(mags[i]),i=eachindex(mags)]
            # new_mags = [ Vector{T}(undef,length(mags)) for i in 1:length(mini_vec) ]
            # @inbounds for i in eachindex(mags)
            #     @inbounds for j in eachindex(mags[i])
            #         new_mags[j][i] = mags[i][j]
            #     end
            # end
            # mags = new_mags
        end
    else
        return mags
    end
end
ingest_mags(mini_vec, mags) = throw(ArgumentError("There is no ingest_mags method for the provided mini_vec/mags types. See the documentation for the public functions, (e.g., [generate_mock_stars_mass](@ref)), for information on valid input types.")) # General fallback in case the types are not recognized. 

function mass_limits(mini_vec::Vector{<:Real}, mags::Vector{T},
                     mag_names::Vector{String}, mag_lim::Real,
                     mag_lim_name::String) where T <: Vector{<:Real}
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
#### Functions to generate mock galaxy catalogs from SSPs

"""
    (sampled_masses, sampled_mags) = generate_mock_stars_mass(mini_vec::Vector{<:Real}, mags, mag_names::Vector{String}, limit::Real, imf::Distributions.UnivariateDistribution{Distributions.Continuous}; dist_mod::Real=0, rng::Random.AbstractRNG=default_rng(), mag_lim::Real=Inf, mag_lim_name::String="V")

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

# Notes
## Population Masses
Given a particular isochrone with an initial mass vector `mini_vec`, it will never cover the full range of stellar birth masses because stars that die are not included in the isochrone. As such, these stars will be sampled by `generate_mock_stars` (assuming that the maximum mass of your `imf` type is set to a physical value like 100 solar masses), but will not end up in the returned catalog. As such, the sum of the initial mass vector you get out will be less than what you requested. The fraction of mass that ends up in the final returned catalog will depend on your IMF model and your isochrone. Generally, if your requested stellar mass is `limit` and the sum of the initial mass vector returned by `generate_mock_stars` is `x * limit` with `x < 1`, `x` can be identified as the surviving mass fraction, which should have an expectation value given by the integral `QuadGK.quadgk(x->x*pdf(imf,x), mmin, maximum(mini_vec))[1] / QuadGK.quadgk(x->x*pdf(imf,x), mmin, mmax)[1]`, with `mmin` and `mmax` being the minimum and maximum mass that can be sampled from your IMF model object `imf`. 
"""
function generate_mock_stars_mass(mini_vec::Vector{<:Real}, mags, mag_names::Vector{String}, limit::Real, imf::UnivariateDistribution{Continuous}; dist_mod::Real=0, rng::AbstractRNG=default_rng(), mag_lim::Real=Inf, mag_lim_name::String="V")
    # Interpret and reshape the `mags` argument into a (length(mini_vec), nfilters) vector of vectors.
    mags = ingest_mags(mini_vec, mags)
    mags = [ i .+ dist_mod for i in mags ] # Update mags with the provided distance modulus.
    # itp = extrapolate(interpolate((mini_vec,), mags, Gridded(Linear())), Throw())
    itp = interpolate((mini_vec,), mags, Gridded(Linear()))
    # Get minimum and maximum masses provided by the isochrone, respecting `mag_lim`, if provided.
    mmin, mmax = mass_limits(mini_vec, mags, mag_names, mag_lim, mag_lim_name)
    # We might be able to gain some efficiency by creating a new truncated IMF with lower bound mmin,
    # when mag_lim is not infinite. Looks like a factor of 2 improvement but without renormalizing `limit`,
    # it will not sample the correct amount of stellar mass. 
    # mmin > minimum(mini_vec) && (imf = truncated(imf; lower=mmin))
    # Setup for iteration. 
    total = zero(eltype(imf))
    mass_vec = Vector{eltype(imf)}(undef,0)
    mag_vec = Vector{Vector{eltype(imf)}}(undef,0)
    while total < limit
        mass_sample = rand(rng, imf) # Just sample one star.
        total += mass_sample         # Add mass to total.
        # Continue loop if sampled mass is outside of isochrone range.
        if (mass_sample < mmin) | (mass_sample > mmax)
            continue
        end
        mag_sample = itp(mass_sample) # Roughly 70 ns for 2 filters on 12600k. No speedup for bulk queries.
        push!(mass_vec, mass_sample)  # scipy.interpolate.interp1d is ~74 ns per evaluation for batched 10k queries.
        push!(mag_vec, mag_sample)
    end
    return mass_vec, mag_vec
end

"""
    (sampled_masses, sampled_mags) =  generate_mock_stars_mag(mini_vec::Vector{<:Real}, mags, mag_names::Vector{String}, absmag::Real, absmag_name::String, imf::UnivariateDistribution{Continuous}; dist_mod::Real=0, rng::AbstractRNG=default_rng(), mag_lim::Real=Inf, mag_lim_name::String="V")

Generates a mock stellar population with absolute magnitude `absmag::Real` (e.g., -7 or -12) in the filter `absmag_name::String` (e.g., "V" or "F606W") which is contained in the provided `mag_names::Vector{String}`. Other arguments are shared with [`generate_mock_stars_mass`](@ref), which contains the main documentation.
"""
function generate_mock_stars_mag(mini_vec::Vector{<:Real}, mags, mag_names::Vector{String}, absmag::Real, absmag_name::String, imf::UnivariateDistribution{Continuous}; dist_mod::Real=0, rng::AbstractRNG=default_rng(), mag_lim::Real=Inf, mag_lim_name::String="V")
    # Interpret and reshape the `mags` argument into a (length(mini_vec), nfilters) vector of vectors.
    mags = ingest_mags(mini_vec, mags)
    mags = [ i .+ dist_mod for i in mags ] # Update mags with the provided distance modulus.
    idxlim = findfirst(x->x==absmag_name, mag_names) # Get the index into `mags` and `mag_names` that equals `limit_name`.
    idxlim == nothing && throw(ArgumentError("Provided `absmag_name` is not contained in provided `mag_names` array.")) # Throw error if absmag_name not in mag_names.
    limit = L_from_MV(absmag) # Convert the provided `limit` from magnitudes into luminosity.
    itp = interpolate((mini_vec,), mags, Gridded(Linear()))
    mmin1, mmax1 = extrema(mini_vec) # Need this to determine validity for mag interpolation.
    mmin2, mmax2 = mass_limits(mini_vec, mags, mag_names, mag_lim, mag_lim_name) # Determine initial mass that corresponds to mag_lim, if provided.
    
    total = zero(eltype(imf))
    mass_vec = Vector{eltype(imf)}(undef,0)
    mag_vec = Vector{Vector{eltype(imf)}}(undef,0)
    while total < limit
        mass_sample = rand(rng, imf)  # Just sample one star.
        # Continue loop if sampled mass is outside of isochrone range.
        if (mass_sample < mmin1) | (mass_sample > mmax1)
            continue
        end
        mag_sample = itp(mass_sample) # Interpolate the sampled mass.
        total += L_from_MV(mag_sample[idxlim] - dist_mod) # Add luminosity to total, subtracting the distance modulus.
        # Only push to the output vectors if the sampled mass is in the valid range. 
        if mmin2 <= mass_sample # <= mmax2 
            push!(mass_vec, mass_sample)
            push!(mag_vec, mag_sample)
        end
    end
    # println(MV_from_L(total)) # Print the sampled absolute magnitude. 
    return mass_vec, mag_vec
end

###############################################
#### Functions to generate composite mock galaxy catalogs from multiple SSPs
