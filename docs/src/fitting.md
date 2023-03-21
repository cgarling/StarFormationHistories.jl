# [Deriving Star Formation Histories from Color-Magnitude Diagrams](@id fitting)

## Background

In the classic formulation of star formation history fitting from resolved-star photometry [(Dolphin 2002)](http://adsabs.harvard.edu/abs/2002MNRAS.332...91D), an observed color-magnitude diagram (CMD) is binned into a 2-D histogram known as a Hess diagram. Such a CMD and Hess diagram pair is shown below.

![Comparison of CMD and a Hess diagram generated from the same observational data.](figures/cmd_hess.png)

The representation of the observations as a Hess diagram allows one to apply Poisson statistics, specifically the Poisson likelihood ratio (Equations 7--10 in Dolphin 2002), to model the observations. As the CMD of a complex stellar population is simply the sum of the CMDs of its sub-populations, one need only prepare a number of **templates** for each simple stellar population (SSP) which may make up the complex population in question and model the observed Hess diagram as a linear combination of these templates. Keeping the same notation as Dolphin 2002, the complex model Hess diagram is simply

```math
m_i = \sum_j \, r_j \, c_{i,j}
```

where ``m_i`` is the value of the complex model in bin ``i``, ``c_{i,j}`` is the value of simple template ``j`` in bin ``i``, and ``r_j`` is the multiplicative coefficient determining how significant template ``j`` is to the complex population. In Dolphin 2002, he normalizes the templates to identical star formation rates (SFRs) and so the ``r_j`` are SFRs as well. In this package, we prefer to normalize our templates to identical population stellar masses, so our ``r_j`` are stellar masses, but the principal is the same.

Construction of the templates is, however, not a trivial exercise. Ideally, a template constructed to represent a particular SSP would accurately reflect the expectation of how such a population would be observed. Thus, these templates must be adjusted for photometric error, incompleteness, and other effects such as those caused by unresolved binary- and multi-star systems. Observational effects such as photometric error and incompleteness are best measured from artificial star tests (ASTs). It is worth mentioning that ASTs can often return "best case" results, as they typically neglect systematics like uncertainty in the point-spread model used for the photometry; as such it is sometimes necessary to add a systematic error floor to photometric error results from ASTs.

Such templates can be constructed by sampling many mock stars from an initial mass function (IMF), interpolating their absolute magnitudes from an isochrone of the relevant SSP, and "mock observing" them by applying photometric error and completeness functions (for example, by looking up the ``1\sigma`` photometric error and completeness value from a catalog of artificial stars). Such Monte Carlo templates can be slow to construct and exhibit Poisson shot-noise, requiring a statistical data--data comparison rather than a model--data comparison. Thus this method is non-optimal from both a practical and statistical perspective.

It is better to form what Dolphin 2002 calls a "blurred isochrone;" in this form of template, the SSP isochrone is first interpolated in initial stellar mass to improve the point density along the isochrone. The number of interpolated points is generally a function of the size of the bins in the Hess diagram and the observational error; more points are required as the bin size or photometric errors become smaller. These points are then weighted according to the IMF and the photometric completeness, and this weight is distributed into the Hess diagram following the photometric error distribution determined by similar artificial stars. Dolphin 2002 also mentions interpolating across stellar age/metallicity when constructing such templates; for example, for an SSP with an age of 1 Gyr and a metallicity of [M/H]=-1.0, you could interpolate the isochrones to introduce a Gaussian metallicity spread of 0.05 dex or an age spread of 100 Myr. The general effects of this form of interpolation is to broaden the model templates, particularly features that are very sharp in true SSP models. We neglect this form of interpolation in our implementation as it adds significant complexity and requires users to provide more information about the isochrones that are providing. Such widening of the individual templates is most impactful when photometric errors in the observational data are low (perhaps <0.10 mag).

## [Constructing Templates](@id templates)

While the above description summarizes the necessary components for constructing such a blurred isochrone, it can be a bit difficult to figure out how best to actually construct them. Specifically there are many ways that one could implement the observational effects of photometric error and incompleteness. We provide a method [`partial_cmd_smooth`](@ref) to construct such templates under the assumption of Gaussian photometric error distributions, which is often a good approximation in the high-completeness regime. This method makes use of user-defined functions for the mean photometric error and completeness as a function of magnitude and filter, such that these can be defined in a number of ways; for example, as direct lookups from a large table of ASTs or as simple function evaluations of analytic approximations or fits to the ASTs.

This method begins by interpolating the provided SSP isochrone to increase point density. For every such point with ``i`` band apparent magnitude ``m_i``, it calls a user-defined function to estimate the ``1\sigma`` photometric error as ``\sigma_i = f_i(m_i)``. The photometric error on the x-axis color for the Hess diagram is estimated from the individual-band ``\sigma_i``. These errors are used to define an asymmetric Gaussian kernel for each point in the interpolated isochrone. This kernel describes the shape of the probability distribution for where in the Hess diagram the isochrone point would be observed. However, it also must be normalized (weighted) according to the IMF and observational completeness functions.

Assume that the vector of initial stellar masses for the points in the interpolated isochrone are ``m_i`` and that they are sorted such that ``m_i < m_{i+1}``. The IMF weight on point ``m_i`` can be approximated as the number fraction of stars born between ``m_i`` and ``m_{i+1}`` divided by the mean mass per star born ``\langle m \rangle``, such that the weight effectively represents **the number of stars expected to be born with masses between ``m_i`` and ``m_{i+1}`` per solar mass of star formation**:

```math
\begin{aligned}
w_{i,\text{IMF}} &= \frac{ \int_0^{m_{i+1}} \frac{dN(m)}{dm} dm - \int_0^{m_{i}} \frac{dN(m)}{dm} dm }{\int_0^\infty m \times \frac{dN(m)}{dm} dm} = \frac{ \int_{m_i}^{m_{i+1}} \frac{dN(m)}{dm} dm }{\langle m \rangle}
\end{aligned}
```

The numerator can either be calculated as the difference in the cumulative distribution function across the bin or approximated efficiently via the trapezoidal rule. The denominator is a function only of the IMF and need only be calculated once. Multiplying this weight by the probability of detection in the relevant bands gives the final weight.

Below we show a comparison of a smooth Hess diagram template constructed with [`partial_cmd_smooth`](@ref) with a Monte Carlo realization created with [`generate_stars_mass`](@ref) and mock-observed with [`model_cmd`](@ref). These use an SSP isochrone of age 10 Gyr and metallicity [M/H] of -2 from PARSEC with identical observational error and completeness models. For the provided stellar mass of $10^7 \, \text{M}_\odot$, the Monte Carlo model is fairly well-sampled but still noticably noisy in regions of the Hess diagram that are less well-populated. 

![Comparison of smooth Hess diagram template from `partial_cmd_smooth` and a Monte Carlo model made with `generate_stars_mass`.](figures/model_cmd.png)


```@docs
partial_cmd_smooth
```

We note that in many cases it can also be helpful to add in a foreground/background template that models contamination of the Hess diagram from stars not in your population of interest -- this is often done using observations of parallel fields though there are several other possible methods.

## High-Level Methods for Fitting

Template construction is by far the most complicated step in the fitting procedure. Once your templates have been constructed, fitting them to an observed Hess diagram amounts to a bounded minimization of the Poisson likelihood ratio (Dolphin 2002). We provide the [`StarFormationHistories.construct_x0`](@ref) method to help with setting the initial guess for this optimization. 

```@docs
StarFormationHistories.construct_x0
```

When it comes to performing the optimization, the simplest method we offer is [`StarFormationHistories.fit_templates_lbfgsb`](@ref). This will optimize one coefficient per template; there is no overarching metallicity evolution or other constraint, besides that the stellar masses of the populations cannot be negative. This performs a maximum likelihood optimization with the bounded quasi-Newton LBFGS method as implemented in [L-BFGS-B](http://users.iems.northwestern.edu/~nocedal/lbfgsb.html) and wrapped in [LBFGS.jl](https://github.com/Gnimuc/LBFGSB.jl) with analytic gradients. It is fast and converges fairly reliably, even when the initial guess is not particularly close to the maximum likelihood estimate. It provides no uncertainty estimation. It is normal for some of the coefficients to converge to zero.

```@docs
StarFormationHistories.fit_templates_lbfgsb
```

We more generally recommend using the exported [`fit_templates`](@ref) method, which ...
```@docs
StarFormationHistories.fit_templates
```

Once you have obtained stellar mass coefficients from the above methods, you can convert them into star formation rates and compute per-age mean metallicities with [`StarFormationHistories.calculate_cum_sfr`](@ref)

```@docs
StarFormationHistories.calculate_cum_sfr
```

## Constrained Metallicity Evolution

## Low-Level Building Blocks

```@docs
StarFormationHistories.composite!
StarFormationHistories.loglikelihood
StarFormationHistories.∇loglikelihood
StarFormationHistories.fg!
```