# [Overview](@id metal_evo_intro)

Why should the metallicity evolution be constrained? While the above methods work well for optimizing the per-template ``r_j`` as a means for fitting SFHs, these methods can produce metallicity evolutions that could be considered unphysical, with large changes in the mean metallicity over small changes in time. An example of this type of behavior is shown in the SFH fit below.

![Example of a SFH fit with variations in the metallicity evolution.](figures/mean_mh.png)

While some metallicity variation in the star-forming gas is to be expected, these variations in the SFH fit can end up being quite large depending on the data and isochrone grid adopted. A solution is to construct a more physically-motivated model.

We can do this using a hierarchical model with a parameterized metallicity evolution where the the ``r_j`` are not the parameters directly optimized. Rather, we can optimize one stellar mass (or star formation rate) parameter per age bin, and then a number of metallicity evolution parameters that determine how that stellar mass is split between models with different metallicities at fixed age.

In most star formation history analyses, the metallicities are constrained through *age-metallicity relations (AMRs)*, where the mean metallicity at time ``t`` is a function of time and a small set of metallicity evolution parameters. A popular AMR model is the linear age-metallicity relation ``\langle [\text{M}/\text{H}] \rangle (t) = \alpha \, \left( T_\text{max} - t \right) + \beta`` with a Gaussian distribution in metallicity at fixed age. ``T_\text{max}`` here is the earliest lookback time under consideration such that ``\langle [\text{M}/\text{H}] \rangle (T_\text{max}) = \beta``. This model is described in more detail [here](@ref linear_amr_section).

AMRs have historically been popular because they are generally capable of producing reasonable fits to observed data and it is relatively easy to derive the gradient of the objective function with respect to the AMR parameters analytically. However, in AMR models there is no direct link between the SFRs being fit and the metallicity evolution as a function of time, even though the two should in principle have some correlation as stellar processes are responsible for enriching the ISM.

A promising avenue of research involves fitting *mass-metallicity relations* (MZRs) rather than AMRs. In these models, the mean metallicity of stars forming at time ``t`` is a function of the total stellar mass of the population at that time -- therefore, the mean metallicity evolution changes self-consistently with the SFRs during the fitting process, resulting in a metallicity evolution that is meaningfully coupled to the star formation history. Additionally, AMRs can be difficult to compare between different galaxies because they do not reflect the different SFHs of the galaxies, whereas MZRs can be compared between galaxies much more easily. Our methods for MZR fitting are described in more detail [here](@ref MZR).

## Generic Methods

While there are some methods in this package that are unique to AMR or MZR models, we present a minimal unified interface that can be used to fit SFHs under both types of models. To support multiple dispatch, we define [`AbstractMetallicityModel`](@ref StarFormationHistories.AbstractMetallicityModel) as the abstract supertype of [`AbstractAMR`](@ref StarFormationHistories.AbstractAMR) and [`AbstractMZR`](@ref StarFormationHistories.AbstractMZR), which are each the supertypes for AMR and MZR types, respectively.

```@docs
StarFormationHistories.AbstractMetallicityModel
```

The generic methods that can be used for both AMRs and MZRs are described here. The main method for obtaining best-fit star formation histories is [`fit_sfh`](@ref).

```@docs
fit_sfh
```

This function returns an instance of [`CompositeBFGSResult`](@ref StarFormationHistories.CompositeBFGSResult).

```@docs
StarFormationHistories.CompositeBFGSResult
StarFormationHistories.BFGSResult
```

This can be used to obtain random samples under a multivariable Normal approximation to the posterior or used to initialize a Hamiltonian Monte Carlo (HMC) sampling process to obtain more accurate posterior samples with [`sample_sfh`](@ref) and its multi-threaded alternative [`tsample_sfh`](@ref StarFormationHistories.tsample_sfh).

```@docs
sample_sfh
StarFormationHistories.tsample_sfh
```

The per-SSP stellar mass coefficients (``r_{j,k}`` in the [derivation](@ref mzr_derivation)) can be derived from a metallicity model, a [metallicity dispersion model](@ref dispersion_models), the per-unique-log(age) stellar mass coefficients (``R_j`` in the [derivation](@ref mzr_derivation)), and the set of SSP logarithmic ages `logAge = log10(age [yr])` and metallicites using [`calculate_coeffs`](@ref StarFormationHistories.calculate_coeffs). Alternatively a [`CompositeBFGSResult`](@ref StarFormationHistories.CompositeBFGSResult) can be fed into this method and the first three arguments will be read from the result object.

```@docs
calculate_coeffs
```