# [Metallicity Dispersion Models](@id dispersion_models)

In order to reproduce the broadness of features in observed color-magnitude diagrams it is common to introduce some dispersion in metallicity for stars formed in each time bin, where each time bin is associated with a mean metallicity ``\mu_j``. We implement a generic hierarchical model to accomodate this functionality.

The hierarchical model forms the per-SSP weights ``r_{j,k}``, which are indexed by population age ``j`` and metallicity ``k``, as a function of a linear coefficient ``R_j`` which describes the stellar mass formed in the time bin, and a relative weight ``A_{j,k}`` which depends on the mean metallicity ``\mu_j`` and the metallicity of the SSP under consideration. In the case of a Gaussian metallicity dispersion at fixed age, which is often used in practice, we can write 

```math
A_{j,k} = \text{exp} \left( - \frac{1}{2} \left( \frac{ [\text{M}/\text{H}]_k - \mu_j}{\sigma} \right)^2 \right)
```

If we take ``R_j`` to be the total stellar mass formed in the time bin, then it is clear that we require the sum over the relative weights for the SSPs of age ``j`` to equal one, i.e., ``\sum_k r_{j,k} = 1``. We therefore require that the relative weight on each SSP template of age ``j`` be normalized by the sum ``\sum_k A_{j,k}``, so that the relative weights are

```math
r_{j,k} = R_j \, \frac{A_{j,k}}{\sum_k A_{j,k}}
```

We provide a generic interface for describing the analytic form of the ``A_{j,k}`` so that it is easy to define new dispersion models that will integrate with our fitting routines. Built-in, ready to use models are described below, and the API for defining new models is described in [the API section](@ref dispersion_API).

## Built-In Models

```@docs
GaussianDispersion
```

## [Metallicity Dispersion API](@id dispersion_API)

Below we describe the API that must be followed in order to implement new types for describing the ``A_{j,k}``, such that they will work with our provided fitting and sampling methods.

```@docs
StarFormationHistories.AbstractDispersionModel
StarFormationHistories.npar(::StarFormationHistories.AbstractDispersionModel)
StarFormationHistories.gradient(::StarFormationHistories.AbstractDispersionModel, ::Real, ::Real)
StarFormationHistories.update_params(::StarFormationHistories.AbstractDispersionModel, ::Any)
StarFormationHistories.transforms(::StarFormationHistories.AbstractDispersionModel)
StarFormationHistories.free_params(::StarFormationHistories.AbstractDispersionModel)
```