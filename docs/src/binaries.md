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