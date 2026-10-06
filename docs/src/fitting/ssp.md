# [Single Stellar Populations](@id ssp)

Rather than fitting a star formation history, the Hess diagram of a population formed in a single burst, such as a star cluster, can be fit with a single simple stellar population (SSP) of one age and metallicity. Because only one SSP template is needed per evaluation of the likelihood, templates are computed on the fly from an isochrone at any age and metallicity, rather than on a fixed grid. This also allows the distance modulus, extinction, binary fraction, and the fraction of stars belonging to a background population (e.g., foreground and background field stars) to be fit simultaneously, each with a fixed value or a user-provided prior.

The model is defined by an [`SSPModel`](@ref StarFormationHistories.SSPModel). The maximum a posteriori (MAP) parameters are found with [`fit_ssp`](@ref), and the posterior is sampled with [`sample_ssp`](@ref), which also returns the stellar mass formed in the SSP. The isochrone is provided by a user function of age, metallicity, and extinction, so any stellar model library can be used; for example, with [StellarTracks.jl](https://github.com/cgarling/StellarTracks.jl) and [BolometricCorrections.jl](https://github.com/cgarling/BolometricCorrections.jl),

```julia
isofunc(logAge, MH, Av) = (iso = isochrone(tracklib, bclib, logAge, MH, Av); (iso.m_ini, [iso.F475W, iso.F814W]))
```

The binary fraction is fit without recomputing templates: the Hess diagram for any binary fraction is a linear combination of a single-star component and a binary component, as returned by [`ssp_components`](@ref StarFormationHistories.ssp_components).

```@docs
StarFormationHistories.SSPModel
StarFormationHistories.fit_ssp
StarFormationHistories.SSPFit
StarFormationHistories.sample_ssp
StarFormationHistories.ssp_hess
StarFormationHistories.ssp_mass
StarFormationHistories.ssp_components
```
