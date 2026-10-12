# [Changelog](@id changelog)

## Version [v2.0.0] -- 2026-10-11

### Added
 - `cum_sfr_quantiles` returns quantiles of the total birth stellar mass of the sampled star formation histories (`mstar`) and, given per-population surviving mass fractions through the new `surviving_fractions` keyword (e.g., from `surviving_mass_fraction`), of their present-day stellar mass (`mstar_present`, otherwise `nothing`). These are quantiles of the total mass of each sampled history, which account for the correlations between time bins, unlike sums of the per-bin star formation rate quantiles.
 - Fitting of single stellar populations (e.g., star clusters): `SSPModel`, `fit_ssp`, `sample_ssp`, and `SSPFit`. Templates are computed on the fly from a user-provided isochrone function, so age, metallicity, distance modulus, and extinction are continuous parameters, each fixed or free with a Distributions.jl prior, as are the binary fraction and the fraction of stars in a uniform or user-provided background. The total amplitude is marginalized analytically. `fit_ssp` finds the maximum a posteriori parameters by optimizing in parallel from starting points found by a quasirandom search over age, metallicity, and distance, and `sample_ssp` samples the posterior, including every mode found by those optimizations, with a short ensemble-sampler pilot per mode followed by adaptive importance sampling. It returns an `SSPSamples` with equal-weight samples (including the stellar mass formed and the expected number of background stars), the probability and evidence of each mode, and convergence diagnostics, and warns when its settings were inadequate. `ssp_hess` and `ssp_mass` give the expected Hess diagram and stellar mass formed for given parameters. `SSPModel` implements the LogDensityProblems.jl interface for use with other samplers.
 - `ssp_components`, which returns the single-star and binary Hess diagrams of a population so that the Hess diagram for any binary fraction is a linear combination of the two, and `with_fraction(binary_model, fraction)` for each binary model.
 - `hess_mask(edges, gates)`, which marks the Hess diagram bins whose centers lie inside any of the given polygons, to exclude regions of the Hess diagram from fits (equivalent to gates in MATCH).
 - Joint observational models: the completeness, photometric error, and bias passed to `partial_cmd_smooth` and `model_cmd` may each be a single callable taking the magnitudes in all filters, which can represent selection that depends on several filters together, e.g., a signal-to-noise cut in every filter. A new 2-D method of `process_ASTs` measures these on a grid of input magnitudes in two filters, blending sparsely populated cells with the 1-D statistics of their rows and columns (keyword `n0`).
 - `binary_systems`, a deterministic quadrature over the binary systems of a binary model, which all binary models now use to build smooth templates. A new binary model only needs to implement `binary_systems` and `binary_mass_fraction` to be supported in templates.
 - Backgrounds in `fit_sfh`: the new `background` keyword takes a `HessBackground`, a fixed Hess diagram shape (e.g., uniform, or a control field) whose expected number of stars is fit (starting from 1% of the observed stars unless given), or fixed, alongside the star formation history and metallicity parameters. Contaminants such as background galaxies and foreground stars otherwise bias the fit toward more old stars and broader metallicity dispersions. The default `NoBackground()` reproduces previous results. The background parameter is the last element of the fitted parameters, and `BFGSResult` has a new `bg_model` field holding the best-fit background; `sample_sfh`, `tsample_sfh`, `cum_sfr_quantiles`, and `calculate_coeffs` account for it. `renormalize_x0` has a new `background` keyword.
 - `generate_stars_mag_composite` accepts birth stellar mass fractions with `frac_type=:mass`, which are converted to luminosity fractions with the new `luminosity_per_mass`. `mass_from_absmag` returns the birth stellar mass of a composite population with a given integrated absolute magnitude.
 - Templates with differential extinction: a new method of `partial_cmd_smooth` (and of `ssp_components`) takes the isochrone magnitudes at several extinctions (`knots`, e.g., the extinction nodes of the bolometric correction grid, between which magnitudes are interpolated linearly) and a continuous distribution of V-band extinctions. Each isochrone point is deposited along its reddening path in steps across which it moves by at most `max_shift` (default 0.5) times the larger of its photometric error and the bin width, each as a Gaussian kernel broadened by the spread of extinction across the step, with errors, completeness, and bias evaluated at the step's mean extinction. Binary systems and joint observational models are supported. `match_extinction` gives the MATCH model of a uniform spread `dAv` for all stars plus an additional spread `dAvy` for young stars, using the new `UniformSum` distribution, and `prepare_extinction` truncates unbounded distributions at their ±3σ quantiles and at zero. `ExponentialDiskExtinction` models stars and dust in exponential disks with different scale heights and an offset between their midplanes, of which the MATCH model is the case of equal scale heights and no offset, optionally with a log-normal spread of the total extinction of the dust across the disk (as in the `-diskav` model of MATCH), and `match_extinction` also adds the additional spread for young stars to any bounded distribution (`UniformConvolution`). `GaussianKernel` now accepts any covariance term, not only -1, 0, and 1.
 - `surviving_mass_fraction(mmax, imf, binary_model=NoBinaries(); mean_mass=mean(imf))`, the fraction of the birth stellar mass of a simple stellar population in stars that are still alive given the initial mass `mmax` of its most massive living star. For `BinaryMassRatio`, it accounts for binary systems whose primary has died while their secondary is alive. `surviving_mass_fraction(massvec, coeffs, imf)` and `recycling_mass_fraction` accept the same `binary_model` argument and `mean_mass` keyword.
 - `surviving_fraction(mmax, imf, binary_model=NoBinaries())`, the number fraction of the stars born in a simple stellar population that are still alive, counting each star in a binary system separately. `surviving_fraction(massvec, coeffs, imf)` and `recycling_fraction` accept the same `binary_model` argument.


### Breaking changes
 - Julia 1.10 (the current LTS) or later is now required; previously Julia 1.7 was supported.
 - Isochrone ingestion no longer sorts the initial mass vector. Interpolated isochrones can have small decreases in initial mass along the isochrone, e.g., from the tip of the red giant branch to the horizontal branch, and sorting interleaved the points of the two phases. The new `ingest_isochrone` (which replaces `sort_ingested`) keeps the points in evolutionary order and removes each decrease by shifting the initial masses of all later points up by its size, preserving the mass differences within each phase. Points that are clearly not in evolutionary order are still sorted by initial mass, with a warning. Templates and simulated catalogs change for isochrones with such decreases. `ingest_isochrone` also no longer mutates its input.
 - `process_ASTs` now returns the photometric error as the standard deviation of a Gaussian with the same median absolute deviation as the residuals (1.4826 times the median absolute deviation), so it can be used directly as the error in templates and `model_cmd`. Returned errors are about 1.48 times larger than before. The `statistic` keyword was removed; bias and error always use the median.
 - Isochrone points used to build smooth templates (`partial_cmd_smooth`) are now spaced by at most half the photometric error (with errors below 0.01 mag treated as 0.01 mag), in addition to at most the bin width. Templates are smoother along the isochrone, particularly at bright magnitudes where the errors are small, and change slightly. `mini_spacing` accepts a vector of per-point maximum spacings for this.
 - The Gaussian kernels that add isochrone points to Hess diagrams were consolidated into a single `GaussianKernel`, replacing the internal `GaussianPSFAsymmetric` and `GaussianPSFCovariant`. When the magnitude of the Hess diagram also appears in its color, each pixel row is now oversampled along the magnitude axis, so the covariance between color and magnitude is integrated accurately even when the errors are small compared to the pixels; see the kernels documentation. Templates change slightly. The `VectorizationBase` dependency was removed.
 - `calculate_cum_sfr` now gives bins without star formation the mean metallicity of the nearest older bin with star formation, or of the nearest younger one for bins older than all star formation. Previously such bins took the value of the next younger bin, and the youngest bin took the unweighted mean of the metallicity grid.
 - `generate_stars_mass` and `generate_stars_mag` now keep the last sampled system, which crosses the requested total, with a probability set by how far it overshoots, so the mean total over many samples equals the requested total; previously the last system was always kept, biasing the mean high. Samples for a given random seed differ from previous versions.
 - `fit_sfh` limits each of its MAP and MLE optimizations to 500 iterations by default, set by the new `iterations` keyword; previously the limit was Optim.jl's default of 1000. Converged fits typically take 100-200 iterations.
 - `fit_sfh` now fits the stellar mass coefficients (and background) with the metallicity and dispersion parameters held at their initial values to obtain an initial solution and starts the full BFGS optimization from that solution. From a poor initial guess, the first steps could otherwise push the mean metallicities beyond the metallicity grid, where the likelihood does not depend on them and the fit could not recover; results of fits that converged before change only within their tolerances.
 - `generate_stars_mass_composite` and `generate_stars_mag_composite` sample their populations in parallel (now with `Threads.@spawn`), each with its own seed drawn from the new `rng` keyword, so results are reproducible for a given `rng` regardless of the number of threads, but differ from previous versions for a given seed.
 - `generate_stars_mass`, `generate_stars_mag`, and their composite versions now record stars that have died (initial masses above the isochrone) with zero mass in the returned systems, so summing the returned masses gives the birth mass of the living stars; previously a binary system whose primary had died but whose companion was alive kept the primary's initial mass. Stars keep their positions in each system, so a primary that has died is 0 while its companion is not, and `add_metadata` then gives `mini = 0`. Stars with initial masses exactly at either end of the isochrone are now treated as in the isochrone by both functions; previously `generate_stars_mass` excluded them.
 - `surviving_mass_fraction(massvec, coeffs, imf)` is now the average of the new single-population method (see Added) over the populations, so it counts the living stars less massive than each isochrone and integrates the IMF exactly instead of with the trapezoidal rule over the isochrone masses. Results increase for isochrones that start above the minimum mass of the IMF and change by the error of the trapezoidal rule otherwise; `recycling_mass_fraction` changes accordingly. The Trapz.jl dependency was removed, and InitialMassFunctions.jl (v0.1.7 or later) is now a dependency, whose `partial_mean` computes the integral in closed form for its IMFs.
 - `surviving_fraction(massvec, coeffs, imf)` is now the average of the new single-population method (see Added) over the populations, so it counts the living stars less massive than each isochrone, which it previously excluded; results increase for isochrones that start above the minimum mass of the IMF (e.g., by 0.12 for `Kroupa2001(0.08, Inf)` and an isochrone starting at 0.1 solar masses). `recycling_fraction` changes accordingly.

### Fixed
 - The default mass ratio distribution of `BinaryMassRatio` takes the floating-point type of `fraction`, e.g., `BinaryMassRatio(0.3f0)` samples Float32 mass ratios; previously it was always Float64, which promoted Float32 computations.
 - Gradients of the hierarchical age-metallicity and mass-metallicity relation fits no longer become `NaN` when the model in an age bin underflows to zero, and metallicity dispersion weights no longer become `NaN` when the mean metallicity is far outside the metallicity grid; all weight is then placed on the nearest grid metallicity.
 - `model_cmd` accepts a `Tuple` of per-filter error and bias models, as documented.
 - Gradients of the hierarchical fits are accurate when the mean metallicity of an age bin is far outside the metallicity grid, e.g., for the oldest bins of steep mass-metallicity relations. The relative metallicity weights underflowed there, which made the gradient of the mass-metallicity relation fits wrong (by orders of magnitude) and could make it infinite, so that some fits stalled for many minutes or failed in the line search. Dispersion models have two new methods, `logweight` and `∇logweight`, with defaults for existing user-defined models; the weights are now computed in logarithmic form relative to the largest weight in each age bin.
 - The Poisson log likelihood (`loglikelihood`) and its gradients add `eps` of the element type to the model in every bin instead of clamping the model to at least `eps`. With the clamp, the log likelihood did not change with the model in bins where the model was below `eps` but the gradient did, so observed stars in such bins (e.g., contaminants without a background model) made the gradient inconsistent with the objective. `loglikelihood` also no longer returns `-typemax` when the log likelihood is exactly 0, which is its maximum, reached when the model equals the data.
 - Metallicity and dispersion model parameters constrained to be negative (`transforms` returning `-1`) are now supported properly in `fit_sfh` and `sample_sfh`; these were never used before, and in the placeholder implementation the log density took the logarithm of the negative parameter, which threw an error, and its gradient had the wrong sign. This was corrected and negatively-constrained parameters now work properly. The Jacobian corrections of all transformed variables use the transformed variable directly rather than the logarithm of the parameter, which is `-Inf` when the parameter underflows to 0.
 - The standard errors that `fit_sfh` reports for log-transformed metallicity and dispersion model parameters were scaled by the MAP value rather than the MLE value in `result.mle.σ`, and, for parameters following a fixed parameter, by the value of a different parameter.

## Version [v1.3.1] -- 2026-06-01
Restricted LineSearches.jl to versions 7.2 through 7.6.0, because changes to its Hager-Zhang line search in later versions made fits worse.

## Version [v1.3.0] -- 2026-05-20
### Changed
 - The Poisson likelihood of the template fits now includes the bins with no observed stars, where the model's expected number of stars `c` contributes `-c` to the log likelihood. Previously these bins were ignored, so models predicting stars in empty regions of the Hess diagram were not penalized. Fit results change, particularly for models that overpredict stars outside the observed populations.
 - The initial guess `x0` is now rescaled before optimization so that the total number of stars in the initial model matches the data (`renormalize_x0`), in `fit_templates`, `fit_templates_lbfgsb`, `fixed_amr`, and the hierarchical `fit_sfh`. For the hierarchical fits, if the initial metallicity model gives invalid coefficients, `x0` is first spread across the metallicity bins, with a warning.
 - Support for Roots.jl v3.

### Added
 - `tau` and `tau_interp`, which give the lookback time at which a given fraction of the total stellar mass formed (e.g., ``\tau_{90}``), optionally with uncertainties.
 - Citations in the documentation through DocumenterCitations.jl, and an extended benchmark suite.

### Fixed
 - `add_metadata` for catalogs with a number of filters other than two, and for catalogs whose first population has no stars.

## Version [v1.2.1] -- 2026-01-10
### Changed
 - Support for Optim.jl v2.
 - `process_ASTs` infers its output types from the input table.
 - `model_cmd` accepts `Tuple`s of error, completeness, and bias functions.

### Fixed
 - `add_metadata` for catalogs in which some simple stellar populations have no star formation.

### Added
 - An example notebook modeling a color-magnitude diagram from a star formation history (`examples/cmd_from_sfr.ipynb`), and package benchmarks.

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