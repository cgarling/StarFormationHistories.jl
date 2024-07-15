# [Binary Systems](@id binaries)
Here we review the API for including binary systems in our population models. Our [Monte Carlo sampling methods](@ref simulate) supports all three models, while our [smooth template modelling](@ref templates) procedure only supports `NoBinaries` and `RandomBinaryPairs`. A comparison between a Monte Carlo population and a smooth template model for a `RandomBinaryPairs` model with binary fraction of 70% is shown below. The redward shift of the lower main sequence typical of populations with high binary fractions is clearly evident and robustly modelled.

```@example
mv("../../examples/templates/template_compare_binaries.svg", "template_compare_binaries.svg") # hide
mv("../../examples/templates/sigma_distribution_binaries.svg", "sigma_distribution_binaries.svg") # hide
nothing # hide
```

![Comparison of CMD-sampled population with smooth Hess diagram template, with binaries.](template_compare_binaries.svg)

## Types
```@docs
StarFormationHistories.AbstractBinaryModel
StarFormationHistories.NoBinaries
StarFormationHistories.RandomBinaryPairs
StarFormationHistories.BinaryMassRatio
```

## Methods
```@docs
StarFormationHistories.binary_system_fraction
StarFormationHistories.binary_number_fraction
StarFormationHistories.binary_mass_fraction
StarFormationHistories.sample_system
```

## Notes

The trickiest part of including binaries in the smooth template modelling procedure is deriving the IMF weights. Let ``M_p`` be the sorted list of initial masses for primary stars and ``M_s`` be the sorted list of initial masses for secondary stars. Conceptually, the IMF weight for a binary system with primary mass ``M_{p,i}`` and secondary mass ``M_{s,j}`` should compute the number fraction of binary systems born with primary masses between ``M_{p,i}`` and ``M_{p,i+1}`` and secondary masses between ``M_{s,j}`` and ``M_{s,j+1}`` per unit solar mass formed. 

In the case of the `RandomBinaryPairs` model, the IMF weights are calculated as follows, with ``dN(M)/dM`` being the IMF for single stars, ``\langle M \rangle`` being the mean mass of single stars over the full range of possible initial masses, and the integral in the denominator being over the range of initial masses in the isochrone. The integral in the denominator accounts for losses due to stellar evolution.

```math
  w_{\text{IMF},i,j} = \frac{\int_{M_{p,i}}^{M_{p,i+1}} \int_{M_{s,j}}^{M_{s,j+1}} \frac{dN(M_p)}{dM} \frac{dN(M_s)}{dM} \ dM_p \ dM_s}{\langle M \rangle \ \int_{M_{\text{min}}}^{M_{\text{max}}} \frac{dN(M)}{dM} \ dM}
```