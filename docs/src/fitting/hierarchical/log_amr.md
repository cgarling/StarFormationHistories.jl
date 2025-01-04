# Logarithmic Age-Metallicity Relation

## Definition

This model differs from the [linear age-metallicity relation (AMR)](@ref linear_amr_section) in its definition of the mean metallicity at lookback time $t$. In the linear model, we defined the mean metallicity as ``\langle [\text{M}/\text{H}] \rangle (t) = \alpha \, \left( T_\text{max} - t_j \right) + \beta``, whereas in this model we define the *metal mass fraction* $Z$ to be linear with lookback time, such that [M/H] scales logarithmically with lookback time,

```math
\begin{aligned}
\langle Z \rangle (t) &= \alpha \, \left( T_\text{max} - t \right) + \beta \\
\langle [\text{M}/\text{H}]\rangle (t) &\equiv \text{log} \left( \frac{\langle Z \rangle \left(t\right)}{X} \right) - \text{log} \left( \frac{Z_\odot}{X_\odot} \right)
\end{aligned}
```

with ``T_\text{max}`` being the earliest lookback time under consideration, such that ``\langle Z \rangle (T_\text{max}) = \beta``. We choose this parameterization so that positive ``\alpha`` and ``\beta`` result in an age-metallicity relation that is monotonically increasing with decreasing lookback time ``t``. If we model the spread in metallicities at fixed ``t`` as Gaussian in [M/H] with the [`GaussianDispersion`](@ref) dispersion model, this implies the spread is asymmetric in ``Z``; this can be seen in the output of `examples/log_amr/log_amr_example.jl`, shown below, which illustrates the relative weights due to a logarithmic AMR across a grid of ages and metallicities.

This logarithmic AMR is implemented with the [`LogarithmicAMR`](@ref) type, which is a subtype of [`AbstractAMR`](@ref StarFormationHistories.AbstractAMR).

```@docs
LogarithmicAMR
```

The per-model coefficients (the ``r_{j,k}`` above) implied by a such a logarithmic AMR can be calculated with [`calculate_coeffs`](@ref) and an example is shown below.

```@example
ENV["GKSwstype"] = "100" # https://discourse.julialang.org/t/generation-of-documentation-fails-qt-qpa-xcb-could-not-connect-to-display/60988 # hide
include("../../../../examples/log_amr/log_amr_example.jl") # hide
savefig("log_amr_plot.svg"); nothing # hide
```

![Visualization of the relative weights across a grid of logAge and metallicity under a logarithmic age-metallicity relation.](log_amr_plot.svg)

## Fitting Functions

[`fit_sfh`](@ref) and [`sample_sfh`](@ref) both work with this AMR model.

The method [`StarFormationHistories.construct_x0_mdf`](@ref) can be used to construct the stellar mass components ``R_j`` of the initial guess vector `x0`.

## Implementation

As the only part of the model that differs from the linear AMR case is the mean age-metallicity relation, most of the [derivation for the linear AMR case](@ref linear_amr_implementation) is still valid here. In particular, only the partial derivatives of the relative weights ``A_{j,k} \equiv \text{exp} \left( -\frac{1}{2 \, \sigma^2} \, \left( [\text{M}/\text{H}]_k - \mu_j \right)^2\right)`` with respect to the fitting parameters ``\alpha`` and ``\beta`` need to be recalculated under the new model. The partial derivative with respect to ``\sigma`` is the same, as the mean metallicity in time bin ``j``, denoted ``\mu_j``, does not depend on ``\sigma``.

```math
\begin{aligned}
Z_j &\equiv \langle Z \left(t_j\right) \rangle = \alpha \, \left( T_\text{max} - t_j \right) + \beta \\
 \\
\mu_j &\equiv \langle [\text{M}/\text{H}] \rangle \left(t_j\right) = \text{log} \left( \frac{\langle Z\left(t_j\right) \rangle}{X_j} \right) - \text{log} \left( \frac{Z_\odot}{X_\odot} \right) \\
&= \text{log} \left[ \frac{ \alpha \, \left( T_\text{max} - t_j \right) + \beta}{X_j} \right] - \text{log} \left( \frac{Z_\odot}{X_\odot} \right)
\end{aligned}
```

We can use the chain rule to write

```math
\begin{aligned}
A_{j,k} &\equiv \text{exp} \left( - \frac{1}{2} \left( \frac{ [\text{M}/\text{H}]_k - \mu_j}{\sigma} \right)^2 \right) \\
\frac{\partial \, A_{j,k}}{\partial \, \beta} &= \frac{\partial \, A_{j,k}}{\partial \, \mu_j} \, \frac{\partial \mu_j}{\partial \beta} \\
\frac{\partial \, A_{j,k}}{\partial \, \alpha} &= \frac{\partial \, A_{j,k}}{\partial \, \mu_j} \, \frac{\partial \mu_j}{\partial \alpha} \\
\end{aligned} \\
```

By definition the hydrogen, helium, and metal mass fractions, ``X``, ``Y``, and ``Z`` respectively, must sum to 1. For PARSEC models, ``Y`` is a function of ``Z`` (see [`Y_from_Z`](@ref StarFormationHistories.Y_from_Z)) such that ``X_j`` is a function of ``Z_j`` and therefore also ``\alpha`` and ``\beta``. Under the PARSEC model, ``Y = Y_p + \gamma \, Z``, we have ``X(Z) = 1 - \left( Y_p + \gamma \, Z \right) - Z`` such that we can rewrite the ``\mu_j`` as 

```math
\mu_j = \text{log} \left( \frac{Z_j}{1 - \left( Y_p + \gamma \, Z_j \right) - Z_j} \right) - \text{log} \left( \frac{Z_\odot}{X_\odot} \right)
```

and we can further expand the partial derivatives of ``\mu_j`` as

```math
\begin{aligned}
\frac{\partial \mu_j}{\partial \beta} &= \frac{\partial \mu_j}{\partial Z_j} \, \frac{\partial Z_j}{\partial \beta} \\
\frac{\partial \mu_j}{\partial \alpha} &= \frac{\partial \mu_j}{\partial Z_j} \, \frac{\partial Z_j}{\partial \alpha} \\
\end{aligned} \\
```

such that the model-dependent portion ``\left( \frac{\partial Z_j}{\partial \beta} \right)`` can be separated from what is essentially a calibration defining how [M/H] is calculated from ``Z`` ``\left( \frac{\partial \mu_j}{\partial Z_j} \right)``. Given our model ``Z_j = \alpha \, \left( T_\text{max} - t_j \right) + \beta`` and the PARSEC calibration for conversion between ``Z_j`` and ``\mu_j`` (i.e., [M/H]), we have

```math
\begin{aligned}
\frac{\partial \mu_j}{\partial Z_j} &= \frac{Y_p - 1}{\text{ln}10 \, Z_j \, \left( Y_p + Z_j + \gamma \, Z_j - 1 \right)} \\
\frac{\partial Z_j}{\partial \beta} &= \frac{\partial \left[\alpha \, \left( T_\text{max} - t_j \right) + \beta \right]}{\partial \beta} = 1 \\
\frac{\partial Z_j}{\partial \alpha} &= \frac{\partial \left[ \alpha \, \left( T_\text{max} - t_j \right) + \beta \right]}{\partial \alpha} = T_\text{max} - t_j \\
\end{aligned}
```

[M/H] as a function of ``Z`` for the PARSEC calibration is available as [`MH_from_Z`](@ref StarFormationHistories.MH_from_Z) and the partial derivative ``\frac{\partial \mu_j}{\partial Z_j}`` is available as [`dMH_dZ`](@ref StarFormationHistories.dMH_dZ).


Which gives us final results

```math
\begin{aligned}
\frac{\partial \, A_{j,k}}{\partial \, \mu_j} &= \frac{A_{j,k} \, \left( [\text{M}/\text{H}]_k - \mu_j \right)}{\sigma^2} \\
\frac{\partial \mu_j}{\partial \beta} &= \frac{\partial \mu_j}{\partial Z_j} \, \frac{\partial Z_j}{\partial \beta} = \left( \frac{Y_p - 1}{\text{ln}(10) \, Z_j \, \left( Y_p + Z_j + \gamma \, Z_j - 1 \right)} \right) \ \left( 1 \right) \\
\frac{\partial \mu_j}{\partial \alpha} &= \frac{\partial \mu_j}{\partial Z_j} \, \frac{\partial Z_j}{\partial \alpha} = \left( \frac{Y_p - 1}{\text{ln}(10) \, Z_j \, \left( Y_p + Z_j + \gamma \, Z_j - 1 \right)} \right) \ \left( T_\text{max} - t_j \right) \\

%% \frac{\partial \mu_j}{\partial \beta} &= \frac{1}{\left( t_j \, \alpha + \beta \right) \, \text{ln}(10)} \\
%% \frac{\partial \mu_j}{\partial \alpha} &= \frac{t}{\left( t_j \, \alpha + \beta \right) \, \text{ln}(10)} = t \, \frac{\partial \mu_j}{\partial \beta} \\
\end{aligned}
```

such that

```math
\begin{aligned}
\frac{\partial \, A_{j,k}}{\partial \, \beta} &= \frac{\partial \, A_{j,k}}{\partial \, \mu_j} \, \frac{\partial \mu_j}{\partial \beta} = \frac{\partial \, A_{j,k}}{\partial \, \mu_j} \, \frac{\partial \mu_j}{\partial Z_j} \, \frac{\partial Z_j}{\partial \beta} \\
&= \left( \frac{A_{j,k} \, \left( [\text{M}/\text{H}]_k - \mu_j \right)}{\sigma^2} \right) \ \left( \frac{Y_p - 1}{\text{ln}10 \, Z_j \, \left( Y_p + Z_j + \gamma \, Z_j - 1 \right)} \right) \\
\frac{\partial \, A_{j,k}}{\partial \, \alpha} &= \frac{\partial \, A_{j,k}}{\partial \, \mu_j} \, \frac{\partial \mu_j}{\partial \alpha} = \frac{\partial \, A_{j,k}}{\partial \, \mu_j} \, \frac{\partial \mu_j}{\partial Z_j} \, \frac{\partial Z_j}{\partial \alpha} \\
&= \left( \frac{A_{j,k} \, \left( [\text{M}/\text{H}]_k - \mu_j \right)}{\sigma^2} \right) \ \left( \frac{Y_p - 1}{\text{ln}10 \, Z_j \, \left( Y_p + Z_j + \gamma \, Z_j - 1 \right)} \right) \ \left( T_\text{max} - t_j \right) \\
&= \left( T_\text{max} - t_j \right)\ \frac{\partial \, A_{j,k}}{\partial \, \beta} \\

%% \frac{\partial \, A_{j,k}}{\partial \, \beta} &= \frac{\partial \, A_{j,k}}{\partial \, \mu_j} \, \frac{\partial \mu_j}{\partial \beta} = \left( \frac{A_{j,k} \, \left( [\text{M}/\text{H}]_k - \mu_j \right)}{\sigma^2} \right) \, \left( \frac{1}{\left( t_j \, \alpha + \beta \right) \, \text{ln}(10)} \right) \\
%% &= \frac{A_{j,k} \, \left( [\text{M}/\text{H}]_k - \mu_j \right)}{\text{ln}(10) \, \sigma^2 \, \left( t_j \, \alpha + \beta \right)} \\
%% \frac{\partial \, A_{j,k}}{\partial \, \alpha} &= \frac{\partial \, A_{j,k}}{\partial \, \mu_j} \, \frac{\partial \mu_j}{\partial \alpha} = t \, \frac{\partial \, A_{j,k}}{\partial \, \beta}
\end{aligned}
```