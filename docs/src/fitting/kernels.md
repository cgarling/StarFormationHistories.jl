# [Kernels](@id kernels)
One of the most important parts of our method is modelling the 2-D probability distribution of observing a star with particular intrinsic magnitudes in the Hess diagram space. We refer to these 2-D probability distributions as "kernels" throughout, as our process is conceptually similar to constructing a kernel density estimate. This document describes how these kernels are constructed and what assumptions are made to do so.

Our implementation is underpinned by the assumption that all kernels can be described the family of 2-D Gaussian probability distributions. To derive these 2-D distributions, we begin with 1-D distributions for each photometric filter used in the analysis.

We assume random photometric errors in the observed photometric catalog can be approximated as Gaussian. This is generally a good assumption for the space-based imaging used for resolved SFHs (e.g., HST and JWST) as long as the photometric completeness is high (greater than, perhaps, 50%). At low completeness, the error distributions become skewed and would be better modelled by something like a skew-normal distribution, which we do not presently consider.

We additionally require that the user can construct functions for each observed photomeric filter which, given a star's input intrinsic magnitude in a single filter ``m_i``, return reliable measures for the average single-band photometric error; ``\sigma(m_i)``. Assuming that the photometric errors in each band can be modelled as independent, we can derive, for a given ``m_i``, the probability distribution of the observed magnitude ``m_o``. We therefore have the 1-D probability distributions ``P(m_o|m_i)`` for all filters.

In the case that imaging in three photometric filters are available and are used to construct a Hess diagram with a magnitude on the y-axis that does not appear on the x-axis (e.g., y=``R`` and x=``B-V``), then there is no covariance between the x and y axes and the 2-D Gaussian kernel is fully separable. We call this the "non-covariant" case.

In order to inject the kernel into the model Hess diagram, it must be integrated across the grid that defines the discretization. The integral of the non-covariant kernel across the Hess diagram pixel grid is entirely analytic, making evaluation easy and efficient. By integrating the kernel over the grid, the numerical precision is nearly independent of the grid resolution, which is *not* true of Monte Carlo (MC) based template construction algorithms.

In the case that imaging in only two photometric filters are available and are used to construct a Hess diagram with a magnitude on the y-axis that *does* appear on the x-axis (e.g., y=``B`` and x=``B-V``), then there is covariance between x and y axes. We model the covariance explicitly. The covariance pattern does not permit a fully analytic integral over each pixel as in the non-covariant case. Because the y-axis filter also enters the color, a star scattered by ``\Delta y`` along the y-axis is also scattered by ``\mp \Delta y`` in color, so the error distribution in color at fixed ``y`` is a Gaussian of width ``\sigma_x`` (the error in the other filter of the color) whose mean shifts linearly with ``y``. Within a single row of Hess diagram pixels, this mean therefore shifts by as much as the row height, which can be several times ``\sigma_x``, so the color distribution integrated over the row is not Gaussian. We handle this by oversampling each row along the y-axis at a sub-pixel resolution fine enough that the shift of the mean color across one sub-row is no larger than ``\sigma_x``. For each sub-row, the fraction of the kernel falling within it is computed exactly from the Gaussian error distribution in ``y``, and its color distribution is approximated by a single Gaussian with the mean and variance of the color over the sub-row, which is integrated exactly over each pixel along the x-axis. The sub-row contributions are summed into the original pixel row, so the output Hess diagram retains the requested resolution. This conserves the total weight of the kernel exactly and remains accurate when the photometric errors are much smaller than the pixel size, as is common for bright stars. The same implementation handles the non-covariant case, in which the mean color does not shift within a row and the integral over each pixel is exact. Kernels are evaluated within four standard deviations of their centers.

The types representing these kernels are not part of our public API, but we provide an example script `examples/templates/kernels_example.jl` which illustrates the performance of these kernels. The output of this script is reproduced and explained below.

Consider a star taken from an isochrone with intrinsic magnitudes ``B=20``, ``V=19``, and ``R=18`` with expected random photometric errors ``\sigma_B=0.02``, ``\sigma_V=0.03``, and ``\sigma_R=0.05``. Our example randomly samples a large population of possible observed magnitudes given these intrinsic properties and compares the distribution of the MC sample to our kernels.

We consider first the non-covariant case:

```@example
mv("../../../examples/templates/covar_0.svg", "figures/covar_0.svg") # hide
mv("../../../examples/templates/covar_1.svg", "figures/covar_1.svg") # hide
mv("../../../examples/templates/covar_m1.svg", "figures/covar_m1.svg") # hide
nothing # hide
```

![noncovariant](figures/covar_0.svg)

which has the expected, non-rotated morphology.

We next consider the covariant case with the covariance pattern y=``B`` and x=``B-V``:

![covariantm1](figures/covar_m1.svg)

which we can see correctly models the covariance pattern.

And finally, the covariant case with the covariance pattern y=``V`` and x=``B-V``:
![covariant1](figures/covar_1.svg)
