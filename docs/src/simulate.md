# [Simulating Color-Magnitude Diagrams](@id simulate)

Modelling observations of resolved stellar populations (e.g., color-magnitude or Hess diagrams) with user-defined star formation histories can be useful for comparison to actual observations, but also enables a number of other scientific activities (e.g., making predictions to motivate observational proposals). To support these uses we offer methods for sampling stellar populations from isochrones using user-defined star formation histories, initial mass functions, and stellar binary models. These methods require data from user-provided isochrones (this package does not provide any), an initial mass function model (such as those provided in [InitialMassFunctions.jl](https://github.com/cgarling/InitialMassFunctions.jl)), and a model specifying how (or if) to sample [binary or multi-star systems](@ref binaries). 

The simplest methods only sample stars from a single stellar population. We provide a method that samples up to a provided stellar mass, [`generate_stars_mass`](@ref) (e.g., $10^7 \, \text{M}_\odot$) and a method that samples up to a provided absolute magnitude [`generate_stars_mag`](@ref) (e.g., -10). These are documented under the first subsection below. These methods are single-threaded.

We also offer methods for sampling populations with complex star formation histories; these are implicitly multi-threaded across the separate populations if you start Julia with multiple threads (e.g., with `julia -t 4` or similar). We provide [`generate_stars_mass_composite`](@ref) for sampling such populations up to a provided stellar mass and [`generate_stars_mag_composite`](@ref) for sampling such populations up to a provided absolute magnitude. These are documented under the second subsection below.

The output produced from the above methods are "pure" in the sense that they do not include any observational effects like photometric error or incompleteness. These effects should be implemented in a post-processing step. We provide a simple method [`model_cmd`](@ref) that accepts user-defined photometric error and completeness functions and applies them to the pure catalog, returning a Monte Carlo realization of a possible observed catalog.

## Simple Stellar Populations
```@docs
generate_stars_mass
generate_stars_mag
```

## Complex Stellar Populations
```@docs
generate_stars_mass_composite
generate_stars_mag_composite
```

## Observational Effects
```@docs
model_cmd
```