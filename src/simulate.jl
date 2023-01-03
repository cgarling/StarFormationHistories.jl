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

#### Functions to generate mock galaxy catalogs
function generate_mock_stars_mass(mini_vec::Vector{<:Real}, mags::Vector{Vector{T}}, mag_names::Vector{String}, limit::Real, imf::UnivariateDistribution{Continuous}; dist_mod::Number=0, rng::AbstractRNG=default_rng(), mag_lim::Number=Inf, mag_lim_name::String="V") where T<:Real
    mags = [ i .+ dmod for i in mags ] # Update mags with the provided distance modulus
    itp = extrapolate(interpolate((mini_vec,), mags, Gridded(Linear())), Throw())
    mmin = minimum(mini_vec) 
    mmax = maximum(mini_vec)
    # Update mmin respecing mag_lim, if provided
    if isfinite(mag_lim)
        idxmag = findfirst(x->x==mag_lim_name, mag_names)
        if mag_lim < mags[findfirst(x->x==mmin, mini_vec)][idxmag]
            # Solve for stellar initial mass where mag == mag_lim.
            itp_tmp = extrapolate(interpolate((mini_vec,), mags[idxmag], Gridded(Linear())), Throw())
            # roots find_zero or something

        end

    end
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
        mag_sample = itp(mass_sample) # .+ dist_mod
        # Test here if sampled magnitudes are inside or outside magnitude range.
        # Alternatively, just derive the initial masses range that gives you valid magnitudes
        # And alter mmin and mmax. 
        push!(mass_vec, mass_sample)
        push!(mag_vec, mag_sample)
        

    end
    return mass_vec, mag_vec


end
