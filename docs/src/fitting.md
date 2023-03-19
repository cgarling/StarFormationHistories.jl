# [Deriving Star Formation Histories from Color-Magnitude Diagrams](@id fitting)

## [Constructing Templates](@id templates)

![Comparison of smooth CMD model from `partial_cmd` and a Monte Carlo model made with `generate_stars_mass`.](figures/model_cmd.png)
A comparison of a smooth Hess diagram model constructed with [`partial_cmd_smooth`](@ref) with a Monte Carlo realization created with [`generate_stars_mass`](@ref) and mock-observed with [`model_cmd`](@ref). These use a simple stellar population of age 10 Gyr and metallicity [M/H] of -2 from PARSEC with identical observational error and completeness models. For the provided stellar mass of $10^7 \, \text{M}_\odot$, the Monte Carlo model is fairly well-sampled but still noticably noisy in regions of the Hess diagram that are less well-populated. 

```@docs
partial_cmd_smooth
```

## High-Level Methods for Fitting

## Low-Level Building Blocks