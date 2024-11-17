# [Mass-Metallicity Relations (MZRs)](@id MZR)

The parametric age-metallicity relations (AMRs) provided by this package are typically sufficient to provide good fits to observed CMDs with very few degrees of freedom. However, they are somewhat arbitrary and unphysical as they have no direct relation to the underlying star formation activity, which is what enrichs the ISM in the first place. As such, physical interpretation of AMRs is dubious -- rather, it is often the population-integrated metallicity distribution functions (MDFs) which are compared against external datasets that directly probe stellar metallicities (e.g., single-star spectroscopy).

As we are simultaneously fitting both the historical star formation rates (SFRs) and the metallicity at which those stars are forming over time, it is possible to design a framework in which the metallicity evolves in a self-consistent way with the star formation activity. As the AMR describes the mean metallicity of stars forming at different times, it should be most directly related to the metallicity evolution of the star-forming ISM. We therefore need to connect the star formation activity to the ISM metallicity.

This is complicated by the fact that in general one-zone chemical models, both star-formation driven outflows (which deplete the ISM of both metals and HI) and pristine gas inflows (composed majorly of HI which dilutes the metallicity of the ISM) must be modelled. While hydrodynamic simulations can provide predictions of outflow rates (i.e., through mass-loading factors), the inflow rates are time-variable and completely unconstrained on an object-to-object basis. As such, general one-zone chemical models are unattractive for our purposes.

A more attractive formulation can be found in the idea of an effective yield, which is the fraction of stellar mass that is composed of metals ``\gamma = M_{Z,*} / M_*``. As shown by [Torrey et al. 2019](https://ui.adsabs.harvard.edu/abs/2019MNRAS.484.5587T), who measured these yields in the Illustris TNG100 simulation, these yields are primarily a function of total galaxy stellar mass and not of redshift. Therefore, the rate of change of the yield with respect to stellar mass ``\frac{\partial \gamma}{\partial M_*}`` can be connected to the rate at which metals are accumulated in stars as galaxies grow.

Further, the existence of the gas-phase mass-metallicity relation (MZR) for star-forming galaxies gives us some empirical guidance for an implementation, as the AMR should be mostly connected to the gas-phase metallicity. While the MZR is mostly unconstrained for the low-mass galaxies that are typically studied in the Local Universe with resolved photometry (``M_* < 10^8 \text{M}_\odot``), the MZR is often extrapolated to low masses as a power law in stellar mass.

While it is known that higher-mass galaxies do not strictly evolve *along* the MZR, due in large part to the time variability of inflows mentioned above, the simple form of the power law extrapolation of the MZR allows for a simple experiment. If (on average) the ISM metallicity at time ``t`` is primarily driven by the total stellar mass at that time ``M_*(t)``, then a two-parameter MZR (power law slope and intercept) coupled to the SFRs in the SFH fitting process should result in better CMD models than a similar two-parameter AMR (e.g., the [linear AMR model](@ref linear_amr_section)) which has no mathematical link to the SFRs. In turn, the best-fit SFRs and population-integrated MDFs should also be more accurate if the modelled AMR is better.

# Derivation

We once again wish to derive the gradient of the objective function with respect to the fitting parameters to enable gradient-based optimization. Our derivation will reuse some of the notation developed in the section on the [linear AMR](@ref linear_amr_section). The main difference is that instead of expressing the mean metallicity as a function of time ``\langle [\text{M}/\text{H}] \rangle (t)``, with an MZR we express the mean metallicity as a function of stellar mass at that time ``\langle [\text{M}/\text{H}] \rangle (\text{M}_*(t))``. This means that the partial derivatives of the objective with respect to the stellar mass coefficients have a more complex form than in the AMR case, as changing the stellar mass formed 10 Gyr ago (for example) would change the total stellar mass at all more recent times, which in turn changes the mean metallicity expected at all more recent times.

In the derivations for AMRs we were agnostic about the choice of template normalization; templates could either be normalized to have units of expected number of stars per solar mass of stars formed ``[N / \text{M}_\odot]``, or expected number of stars per unit star formation rate ``[N / \dot{\text{M}}_\odot]``. Changing the units of the templates would, in turn, change the units of the fitting variables returned by the fitting routines, but as shown in our example Jupyter notebook, the fit results are the same no matter the choice of fitting units. For an MZR, we *must* know the units of the templates and fitting variables so that we may properly calculate the total cumulative stellar mass as a function of time. *For simplicity, we assume templates are normalized to number of stars per solar mass of stars formed ``[N / \text{M}_\odot]`` and that the fitting variables are therefore the total stellar mass ascribed to each time bin -- this is the default behavior for the template creation routines.*

Borrowing notation from [the fitting introduction](@ref fitting) and the [section on linear AMRs](@ref linear_amr_section), the bin ``m_i`` of the complex model Hess diagram can be written as the sum over the grid of templates with ages indexed by ``j`` and metallicities indexed by ``k`` as

```math
m_i = \sum_{j,k} \, r_{j,k} \; c_{i,j,k}
```

where ``m_i`` is the value of the complex model in bin ``i``, ``c_{i,j,k}`` is the value of the SSP template with age ``j`` and metallicity ``k`` in bin ``i``, and ``r_{j,k}`` is the multiplicative coefficient determining how significant the template is to the complex population.

The gradient of the objective with respect to the ``r_{j,k}`` is given by Equation 21 in Dolphin 2001 as shown in the [section on linear AMRs](@ref linear_amr_section),

```math
\begin{aligned}
F \equiv - \text{ln} \, \mathscr{L} &= \sum_i m_i - n_i \times \left( 1 - \text{ln} \, \left( \frac{n_i}{m_i} \right) \right) \\
\frac{\partial \, F}{\partial \, r_{j,k}} &= \sum_i c_{i,j,k} \left( 1 - \frac{n_i}{m_i} \right)
\end{aligned}
```

where ``n_i`` is bin ``i`` of the observed Hess diagram. These partial derivatives are easy to obtain, but we need partials with respect to the total stellar mass formed at each distinct age, ``R_j``. Here we must deviate from the derivation for AMRs.

For the purposes of illustration, we will consider a power law MZR with slope ``\alpha`` as is typically used to describe the extrapolation of gas-phase MZRs to masses below ``10^8 \text{M}_\odot``. Under this model, we can express the mean metallicity at time ``j``, notated as ``\mu_j``, as

```math
\begin{aligned}
\mu_j &= [\text{M}/\text{H}]_0 + \text{log} \left( \left( \frac{\text{M}_* (t_j)}{\text{M}_0} \right)^\alpha \right) \\
&= [\text{M}/\text{H}]_0 + \alpha \, \left( \text{log} \left( \text{M}_* (t_j) \right) - \text{log} \left( \text{M}_0 \right) \right) \\
\end{aligned}
```

where the power law MZR is normalized such that the mean metallicity is ``[\text{M}/\text{H}]_0`` at stellar mass ``\text{M}_0``. As in the AMR models, we use a Gaussian to introduce some metallicity dispersion at fixed time, such that the ``r_{j,k}`` are

```math
\begin{aligned}
r_{j,k} &= R_j \, \frac{ \text{exp} \left( - \left( \frac{ [\text{M}/\text{H}]_k - \mu_j}{\sigma} \right)^2 \right)}{\sum_k \text{exp} \left( - \left( \frac{ [\text{M}/\text{H}]_k - \mu_j}{\sigma} \right)^2 \right)}
A_{j,k} &= \text{exp} \left( - \left( \frac{ [\text{M}/\text{H}]_k - \mu_j}{\sigma} \right)^2 \right)
r_{j,k} &= R_j \, \frac{A_{j,k}}{\sum_k A_{j,k}}
\end{aligned}
```

where the ``R_j`` are the fitting variables representing the total stellar mass formed at each distinct age in the template grid. We define ``A_{j,k}`` to substitute in for the broadening PDF, as the Gaussian could easily be substituted for other forms with minimal changes to the derivation. The three parameters in the MZR model are therefore the power law slope ``\alpha``, a normalization/intercept parameter ``[\text{M}/\text{H}]_0``, and the metallicity broadening parameter ``\sigma``; this is the same number of parameters as in the linear AMR model.


