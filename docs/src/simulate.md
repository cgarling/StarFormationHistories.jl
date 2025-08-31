# [Simulating Color-Magnitude Diagrams](@id simulate)

Modelling observations of resolved stellar populations (e.g., color-magnitude or Hess diagrams) with user-defined star formation histories can be useful for comparison to actual observations, but also enables a number of other scientific activities (e.g., making predictions to motivate observational proposals). To support these uses we offer methods for sampling stellar populations from isochrones using user-defined star formation histories, initial mass functions, and stellar binary models. These methods require data from user-provided isochrones (this package does not provide any), an initial mass function model (such as those provided in [InitialMassFunctions.jl](https://github.com/cgarling/InitialMassFunctions.jl)), and a model specifying how (or if) to sample [binary or multi-star systems](@ref binaries). 

The simplest methods only sample stars from a single stellar population. We provide a method that samples up to a provided stellar mass, [`generate_stars_mass`](@ref) (e.g., $10^7 \, \text{M}_\odot$) and a method that samples up to a provided absolute magnitude [`generate_stars_mag`](@ref) (e.g., $M_V=-10$). These are documented under the first subsection below. These methods are single-threaded.

We also offer methods for sampling populations with complex star formation histories; these are implicitly multi-threaded across the separate populations if you start Julia with multiple threads (e.g., with `julia -t 4` or similar). We provide [`generate_stars_mass_composite`](@ref) for sampling such populations up to a provided stellar mass and [`generate_stars_mag_composite`](@ref) for sampling such populations up to a provided absolute magnitude. These are documented under the second subsection below.

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

We provide the convenience function [`StarFormationHistories.add_metadata`](@ref) that takes output from [`generate_stars_mass_composite`](@ref) or [`generate_stars_mag_composite`](@ref) and associates additional metadata (like SSP ages and metallicities) with the stars, returning a `Vector{NamedTuple}` that can be used to construct tables like `TypedTables.Table` and `DataFrames.DataFrame`.

```@docs
StarFormationHistories.add_metadata
```

## Observational Effects

The output produced from the above methods are clean in the sense that they do not include any observational effects like photometric error or incompleteness. These effects should be implemented in a post-processing step. We provide a simple method [`model_cmd`](@ref) that accepts user-defined photometric error and completeness functions and applies them to the initial catalog, returning a Monte Carlo realization of a possible observed catalog. This method assumes Gaussian photometric errors and that the photometric error and completeness functions are separable by filter -- these assumptions are not applicable for all types of data, but the source code for the method is exceedingly simple (~20 lines) and should provide an example for how you could write a similar method that more accurately reflects your data.

```@docs
model_cmd
```

## Developer Internals
```@docs
StarFormationHistories.ingest_mags
StarFormationHistories.sort_ingested
StarFormationHistories.mass_limits
```