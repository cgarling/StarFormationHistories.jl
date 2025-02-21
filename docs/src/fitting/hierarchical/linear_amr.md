# [Linear Age-Metallicity Relation](@id linear_amr_section)

Here we describe the linear age-metallicity relation ``\langle [\text{M}/\text{H}] \rangle (t) = \alpha \, \left( T_\text{max} - t \right) + \beta`` with a Gaussian distribution in metallicity at fixed age as described by the [`GaussianDispersion`](@ref) dispersion model. ``T_\text{max}`` here is the earliest lookback time under consideration such that ``\langle [\text{M}/\text{H}] \rangle (T_\text{max}) = \beta``. If the per-age-bin stellar mass coefficients are ``R_j``, the age of the stellar population ``j`` is ``t_j``, and the metallicity of population ``k`` is ``[\text{M}/\text{H}]_k``, then we can write the per-model ``r_{j,k}`` (where we are now using separate indices for age and metallicity) as

```math
\begin{aligned}
\mu_j &= \alpha \, \left( T_\text{max} - t_j \right) + \beta \\
r_{j,k} &= R_j \, \frac{ \text{exp} \left( - \left( \frac{ [\text{M}/\text{H}]_k - \mu_j}{\sigma} \right)^2 \right)}{\sum_k \text{exp} \left( - \left( \frac{ [\text{M}/\text{H}]_k - \mu_j}{\sigma} \right)^2 \right)}
\end{aligned}
```

where the numerator is the MDF at fixed age evaluated at metallicity ``[\text{M}/\text{H}]_k`` and the denominator is a normalizing coefficient that ensures ``\sum_k r_{j,k} = R_j``. In this notation, bin ``i`` of the complex model Hess diagram (equation 1 of Dolphin 2002) is

```math
m_i = \sum_{j,k} \, r_{j,k} \; c_{i,j,k}
```

Below we show a fit using this hierarchical model to the same data as was used to derive the unconstrained fit in [the introduction](@ref metal_evo_intro). 

![Example of a SFH fit with a linear metallicity evolution.](figures/mdf_model.png)

This model is represented by the [`LinearAMR`](@ref) type, which is a subtype of [`AbstractAMR`](@ref StarFormationHistories.AbstractAMR).

```@docs
StarFormationHistories.AbstractAMR
LinearAMR
```

## Fitting Functions

[`fit_sfh`](@ref) and [`sample_sfh`](@ref) both work with this AMR model.

The method [`StarFormationHistories.construct_x0_mdf`](@ref) can be used to construct the stellar mass components ``R_j`` of the initial guess vector `x0`.

```@docs
StarFormationHistories.construct_x0_mdf
```

and [`calculate_coeffs`](@ref) can be used to calculate per-template stellar mass coefficients (the ``r_{j,k}`` above) given the results of a fit (which will be the ``R_j`` in the equations above).

## [Implementation](@id linear_amr_implementation)

While one could optimize the above model without an analytic gradient, such gradient-free methods are typically slower and less robust. One could also calculate the gradient numerically using finite differences or auto-differentiation, but these are still slower than analytic calculations. We will show that the gradient of this hierarchical model is analytic, allowing us to design an efficient optimization scheme.

Equation 21 in Dolphin 2001 gives the gradient of our objective function with respect to the underlying coefficients

```math
\begin{aligned}
F \equiv - \text{ln} \, \mathscr{L} &= \sum_i m_i - n_i \times \left( 1 - \text{ln} \, \left( \frac{n_i}{m_i} \right) \right) \\
\frac{\partial \, F}{\partial \, r_{j,k}} &= \sum_i c_{i,j,k} \left( 1 - \frac{n_i}{m_i} \right)
\end{aligned}
```

where ``c_{i,j,k}`` is the value of template ``j,k`` in bin ``i`` and ``n_i`` is bin ``i`` of the observed Hess diagram. These partial derivatives are easy to obtain, but we need partials with respect to the per-age-bin fitting parameters ``R_j``. Given the above relation between ``r_{j,k}`` and ``R_j``, we can calculate these derivatives as

```math
\begin{aligned}
\frac{\partial \, F}{\partial \, R_j} &= \sum_k \, \frac{\partial \, F}{\partial \, r_{j,k}} \, \frac{\partial \, r_{j,k}}{\partial \, R_j} \\
\frac{\partial \, r_{j,k}}{\partial \, R_j} &= \frac{ \text{exp} \left( - \frac{1}{2} \left( \frac{ [\text{M}/\text{H}]_k - \mu_j}{\sigma} \right)^2 \right)}{\sum_k \text{exp} \left( - \frac{1}{2} \left( \frac{ [\text{M}/\text{H}]_k - \mu_j}{\sigma} \right)^2 \right)} = \frac{r_{j,k}}{R_j}
\end{aligned}
```

Then we need only the partial derivatives of the objective function ``F`` with respect to the MDF parameters, which in this case are ``\alpha, \beta, \sigma``. For convenience we will rewrite

```math
r_{j,k} = R_j \, \frac{ \text{exp} \left( - \frac{1}{2} \left( \frac{ [\text{M}/\text{H}]_k - \mu_j}{\sigma} \right)^2 \right)}{\sum_k \text{exp} \left( - \frac{1}{2} \left( \frac{ [\text{M}/\text{H}]_k - \mu_j}{\sigma} \right)^2 \right)} = R_j \, \frac{A_{j,k}}{\sum_k A_{j,k}}
```

as many different types of models can be expressed via this simplified notation by substituting the ``A_{j,k}`` with different distributions. This allows us to write 

```math
\begin{aligned}
\frac{\partial \, F}{\partial \, \beta} &= \sum_{j,k} \frac{\partial \, F}{\partial \, r_{j,k}} \, \frac{\partial \, r_{j,k}}{\partial \, \beta} \\
\frac{\partial \, r_{j,k}}{\partial \, \beta} &= R_j \left( \frac{1}{\sum_k \, A_{j,k}} \, \frac{\partial \, A_{j,k}}{\partial \, \beta} - \frac{A_{j,k}}{\left( \sum_k \, A_{j,k} \right)^2} \, \frac{\partial \, \sum_k \, A_{j,k}}{\partial \, \beta} \right)  \\
&= \frac{R_j}{\sum_k \, A_{j,k}} \left( \frac{\partial \, A_{j,k}}{\partial \, \beta} - \frac{A_{j,k}}{\sum_k \, A_{j,k}} \sum_k \frac{\partial \, A_{j,k}}{\partial \, \beta} \right) \\
\end{aligned}
```

Given our specific definition of ``A_{j,k}`` being a Gaussian distribution, we have

```math
\begin{aligned}
\mu_j &= \alpha \, \left( T_\text{max} - t_j \right) + \beta \\
\frac{\partial \, A_{j,k}}{\partial \, \beta} &= \frac{\partial}{\partial \, \beta} \, \left[ \text{exp} \left( - \frac{1}{2} \left( \frac{ [\text{M}/\text{H}]_k - \mu_j}{\sigma} \right)^2 \right) \right] \\
&= \frac{A_{j,k}}{\sigma^2} \left( [\text{M}/\text{H}]_k - \mu_j \right)
\end{aligned}
```

We can now substitute this result into the above expressions to write

```math
\begin{aligned}
\frac{\partial \, F}{\partial \, \beta} &= \sum_{j,k} \frac{\partial \, F}{\partial \, r_{j,k}} \, \frac{\partial \, r_{j,k}}{\partial \, \beta} \\
&= \sum_{j,k} \frac{\partial \, F}{\partial \, r_{j,k}} \, \frac{R_j}{\sum_k \, A_{j,k}} \left( \frac{\partial \, A_{j,k}}{\partial \, \beta} - \frac{A_{j,k}}{\sum_k \, A_{j,k}} \sum_k \frac{\partial \, A_{j,k}}{\partial \, \beta} \right) \\
&= \sum_{j,k} \frac{\partial \, F}{\partial \, r_{j,k}} \, \frac{R_j}{\sigma^2 \, \sum_k \, A_{j,k}} \left( A_{j,k} \left( [\text{M}/\text{H}]_k - \mu_j \right) - \frac{A_{j,k}}{\sum_k \, A_{j,k}} \sum_k A_{j,k} \left( [\text{M}/\text{H}]_k - \mu_j \right) \right)
\end{aligned}
```

It can be shown that the partial derivative of ``F`` with respect to ``\alpha`` is simply

```math
\frac{\partial \, F}{\partial \, \alpha} = \sum_{j,k} \frac{\partial \, F}{\partial \, r_{j,k}} \, \frac{\partial \, r_{j,k}}{\partial \, \alpha} = \sum_{j,k} \frac{\partial \, F}{\partial \, r_{j,k}} \, \frac{\partial \, r_{j,k}}{\partial \, \beta} \times \left( T_\text{max} - t_j \right) \\
```

The partial derivative with respect to ``\sigma`` is slightly more complicated, but we can start identically to how we started above when deriving ``\frac{\partial \, F}{\partial \, \beta}`` with

```math
\begin{aligned}
\frac{\partial \, F}{\partial \, \sigma} &= \sum_{j,k} \frac{\partial \, F}{\partial \, r_{j,k}} \, \frac{\partial \, r_{j,k}}{\partial \, \sigma} \\
\frac{\partial \, r_{j,k}}{\partial \, \sigma} &= R_j \left( \frac{1}{\sum_k \, A_{j,k}} \, \frac{\partial \, A_{j,k}}{\partial \, \sigma} - \frac{A_{j,k}}{\left( \sum_k \, A_{j,k} \right)^2} \, \frac{\partial \, \sum_k \, A_{j,k}}{\partial \, \sigma} \right)  \\
&= \frac{R_j}{\sum_k \, A_{j,k}} \left( \frac{\partial \, A_{j,k}}{\partial \, \sigma} - \frac{A_{j,k}}{\sum_k \, A_{j,k}} \sum_k \frac{\partial \, A_{j,k}}{\partial \, \sigma} \right) \\
\end{aligned}
```

Then all we need is

```math
\frac{\partial \, A_{j,k}}{\partial \, \sigma} = \frac{A_{j,k} \, \left( [\text{M}/\text{H}]_k - \mu_j \right)^2}{\sigma^3}
```

which we can substitute into the above expressions to find ``\frac{\partial \, F}{\partial \, \sigma}``.