# [Changelog](@id changelog)

## Version [v1.2.0] 2025-09-17
Added methods calculating the fraction of stars that have survived / perished from an SSP.

 - `surviving_fraction`
 - `recycling_fraction`

Added methods calculating the fraction of initial SSP mass that remains / has been returned to the ISM.

 - `surviving_mass_fraction`
 - `recycling_mass_fraction`

## Version [v1.1.0] -- 2025-07-08
Added treatment of photometric bias to template generation (`partial_cmd_smooth` and `binary_hess`) and CMD modeling (`model_cmd`); see [#66].

## Version [v1.0.2] -- 2025-05-25
Support for MCMCChains v7.

## Version [v1.0.1] -- 2025-05-16
Support for Interpolations v0.16.

## Version [v1.0.0] -- 2025-03-03
Initial release corresponding to code version used in the initial paper.