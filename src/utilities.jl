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
Inverse of [`arcsec_to_pc`](@ref StarFormationHistories.arcsec_to_pc).

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
mag2flux(m::T, zpt::S=0) where {T<:Real,S<:Real} = exp10(4 * (zpt-m) / 10)
flux2mag(f::T, zpt::S=0) where {T<:Real,S<:Real} = zpt - 5 * log10(f) / 2

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

# Numerical utilities

# function estimate_mode(data::AbstractVector{<:Real})
#     bw = KernelDensity.default_bandwidth(data) / 10
#     KDE = KernelDensity.kde(data; bandwidth=bw)
#     return KDE.x[findmax(KDE.density)[2]]
# end
