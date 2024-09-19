# [Helper Functions](@id helpers)

## Distances and Sizes

```@docs
StarFormationHistories.arcsec_to_pc
StarFormationHistories.pc_to_arcsec
StarFormationHistories.distance_modulus
StarFormationHistories.distance_modulus_to_distance
```

## Magnitudes and Luminosities

```@docs
StarFormationHistories.mag2flux
StarFormationHistories.flux2mag
StarFormationHistories.magerr
StarFormationHistories.fluxerr
StarFormationHistories.snr_magerr
StarFormationHistories.magerr_snr
```

## [Metallicities](@id metallicity_helpers)

```@docs
StarFormationHistories.Y_from_Z
StarFormationHistories.X_from_Z
StarFormationHistories.MH_from_Z
StarFormationHistories.dMH_dZ
StarFormationHistories.Z_from_MH
StarFormationHistories.mdf_amr
```

## [Photometric Error and Completeness Models](@id phot_helpers)

```@docs
StarFormationHistories.Martin2016_complete
StarFormationHistories.exp_photerr
StarFormationHistories.process_ASTs
```