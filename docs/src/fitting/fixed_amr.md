# Fixed Age-Metallicity Relations

It is often the case that one may want to fit for star formation rates under a fixed age-metallicity relation or other metallicity evolution model with no degrees of freedom. Such functionality is provided by [`fixed_amr`](@ref StarFormationHistories.fixed_amr), which takes as input the relative weights (`relweights` in the function call, equivalently the ``r_{j,k}`` in the above derivation) on each template due to a predetermined metallicity model and fits only the per-age-bin coefficients ($R_j$ in the above derivation). 

```@docs
StarFormationHistories.fixed_amr
```
