# [Binary Systems](@id binaries)
Here we review the API for including binary systems in our population models. Both our [Monte Carlo sampling methods](@ref simulate) and our [smooth template modelling](@ref templates) procedure support all three models. A comparison between a Monte Carlo population and a smooth template model for a `RandomBinaryPairs` model with binary fraction of 70% is shown below. The redward shift of the lower main sequence typical of populations with high binary fractions is clearly evident and robustly modelled.

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
StarFormationHistories.binary_systems
StarFormationHistories.with_fraction
```

## Notes

Smooth templates with binaries are constructed from a quadrature over the binary systems of the chosen model, as returned by [`StarFormationHistories.binary_systems`](@ref). Each node of the quadrature is a binary system with stars of initial masses ``M_1 \geq M_2`` and a weight ``w`` giving the expected number of such systems per unit stellar mass formed. Stars with initial masses outside the range of the isochrone (e.g., stars that have died) contribute no light, so binary systems in which only one star remains appear at the position of that star in the Hess diagram. The magnitudes of each system are computed from the summed flux of its stars, after which completeness and photometric bias are applied at the system magnitudes as for single stars. The systems are then aggregated in cells half the width of the Hess diagram bins, and each cell is represented by a single smoothing kernel placed at the weighted mean position of its systems, with photometric errors evaluated at the weighted mean intrinsic magnitudes of its systems. Single-star systems are modelled as for populations without binaries and contribute a fraction ``1 - f_M`` of the template, where ``f_M`` is given by [`StarFormationHistories.binary_mass_fraction`](@ref).

For the `RandomBinaryPairs` model with binary system fraction ``b``, both stars are drawn from the single-star IMF ``dN(M)/dM`` with mean mass ``\langle M \rangle``, so that the number of binary systems per unit stellar mass formed is ``b / \left[ (1 + b) \langle M \rangle \right]``. With ``a_i`` being the IMF probability represented by isochrone mass ``M_i`` under the trapezoidal rule, the system with masses ``M_i \geq M_j`` has weight

```math
  w_{i,j} = \frac{b}{(1 + b) \langle M \rangle} \, a_i \, a_j \times \begin{cases} 1 & i = j \\ 2 & i \neq j \end{cases}
```

where the factor of 2 counts both orderings of the two independently drawn stars. Systems in which one star has a mass outside the range of the isochrone are included with weight ``2 \, b \, a_i \left( 1 - \sum_j a_j \right) / \left[ (1 + b) \langle M \rangle \right]``.

For the `BinaryMassRatio` model, the IMF ``dN(M)/dM`` is the distribution of system masses ``M = M_1 + M_2``, and the mass ratio ``q = M_2 / M_1`` follows the distribution ``dN(q)/dq``. The number of binary systems per unit stellar mass formed is ``b / \langle M \rangle``, and the quadrature covers ``q`` with equal-width cells of probability ``p_l`` together with the isochrone masses. For systems whose primary is in the isochrone, the system with primary mass ``M_{1,i}`` and mass ratio ``q_l`` has weight

```math
  w_{i,l} = \frac{b}{\langle M \rangle} \, \Delta M_i \, p_l \, (1 + q_l) \, \frac{dN}{dM}\left( M_{1,i} (1 + q_l) \right)
```

with ``\Delta M_i`` the trapezoidal rule weight of ``M_{1,i}``, while systems whose primary is more massive than the isochrone and whose secondary is in it are included through the analogous change of variables in the secondary mass, ``M = M_2 (1 + q) / q``.
