# High-Level Methods for Unconstrained Fitting

## Maximum Likelihood Optimization

Template construction is by far the most complicated step in the fitting procedure. Once your templates have been constructed, fitting them to an observed Hess diagram amounts to maximization of the Poisson likelihood ratio (Dolphin 2002). It is possible to construct more complicated hierarchical models including things like metallicity distribution functions; we discuss these in the next section. In this section we discuss methods for fitting where the only constraint is that star formation rates cannot be negative. We provide the [`StarFormationHistories.construct_x0`](@ref) method to help with setting the initial guess for this optimization. 

```@docs
StarFormationHistories.construct_x0
```

When it comes to performing the optimization, the simplest method we offer is [`StarFormationHistories.fit_templates_lbfgsb`](@ref). This will optimize one coefficient per template; there is no overarching metallicity evolution or other constraint, besides that the stellar masses of the populations cannot be negative. This performs a maximum likelihood optimization with the bounded quasi-Newton LBFGS method as implemented in [L-BFGS-B](http://users.iems.northwestern.edu/~nocedal/lbfgsb.html) and wrapped in [LBFGS.jl](https://github.com/Gnimuc/LBFGSB.jl) with analytic gradients. It is fast and converges fairly reliably, even when the initial guess is not particularly close to the maximum likelihood estimate. It provides no uncertainty estimation. It is normal for some of the coefficients to converge to zero.

```@docs
StarFormationHistories.fit_templates_lbfgsb
```

This method simply minimizes the negative logarithm of the Poisson likelihood ratio (Equation 10 in Dolphin 2002),

```math
- \text{ln} \, \mathscr{L} = \sum_i m_i - n_i \times \left( 1 - \text{ln} \, \left( \frac{n_i}{m_i} \right) \right)
```

where ``m_i`` is bin ``i`` of the complex model and ``n_i`` is bin ``i`` of the observed Hess diagram; this can therefore be thought of as computing the maximum likelihood estimate.

We also provide [`StarFormationHistories.fit_templates_fast`](@ref), which is the fastest method we offer for deriving a maximum likelihood estimate for the type of model described above.

```@docs
StarFormationHistories.fit_templates_fast
```

## Posterior Sampling: MCMC

For low-dimensional problems, Markov Chain Monte Carlo (MCMC) methods can be an efficient way to sample the posterior and obtain uncertainty estimates on the fitting coefficients ``r_j``. We provide [`StarFormationHistories.mcmc_sample`](@ref) for this purpose. Internally this uses the multi-threaded affine-invariant MCMC sampler from [KissMCMC.jl](https://github.com/mauro3/KissMCMC.jl) to perform the sampling, which is based on the same algorithm as Python's [emcee](https://emcee.readthedocs.io/en/stable/) (specifically, their `emcee.moves.StretchMove`). There are other MCMC packages like [AdvancedMH.jl](https://github.com/TuringLang/AdvancedMH.jl) that offer additional features like distributed execution. 

```@docs
StarFormationHistories.mcmc_sample
```

## Posterior Sampling: Change of Variables and HMC

[Dolphin 2013](https://ui.adsabs.harvard.edu/abs/2013ApJ...775...76D/abstract) examined methods for obtaining uncertainties on the fitted coefficients (the ``r_j`` in Equation 1 of Dolphin 2002) and found that the Hamiltonian Monte Carlo (HMC) approach allowed for relatively efficient sampling of the posterior distribution when considering many isochrones in the modelling process. HMC requires that the variables to be fit are continuous over the real numbers and so requires a change of variables. Rather than sampling the variables ``r_j`` directly, we can sample ``\theta_j = \text{ln} \left( r_j \right)`` such that the sampled variables are continuous over the real numbers ``-\infty < \theta_j < \infty`` while the ``r_j=\text{exp} \left( \theta_j \right)`` coefficients are bounded from ``0 < r_j < \infty``. Using a logarithmic transformation has the additional benefit that the gradient of the Poisson likelihood ratio is still continuous and easy to compute analytically.

While maximum likelihood estimates are invariant under variable transformations, sampling methods like HMC are not, as formally the posterior being sampled from is a *distribution* and therefore must be integrable over the sampling coefficients. We can write the posterior from which we wish to sample as

```math
\begin{aligned}
p(r_j | D) &= \frac{p(D | r_j) \; p(r_j)}{Z} \\
p(\boldsymbol{r} | D) &= \frac{1}{Z} \; \prod_j p(D | r_j) \; p(r_j) \\
-\text{ln} \; p(\boldsymbol{r} | D) &= \text{ln} \, Z - \sum_j \, \text{ln} \, p(D | r_j) + \text{ln} \, p(r_j) \\
&= \text{ln} \, Z - \text{ln} \, \mathscr{L} + \sum_j \text{ln} \, p(r_j)
\end{aligned}
```

where ``Z`` is the Bayesian evidence (a constant that can be neglected for sampling methods), ``p \left( r_j \right)`` is the prior on the star formation history, and ``\mathscr{L}`` is the Poisson likelihood ratio discussed above. An uninformative (and unnormalized) prior on the coefficients ``r_j`` could take the form of

```math
p(r_j) = \begin{cases}
1; & r_j \geq 0\\
0; & r_j < 0
\end{cases}
```

such that, if the coefficients ``r_j`` are guaranteed to be positive, the final term becomes zero (since ``\text{ln}(1)=0``) and

```math
-\text{ln} \; p(\boldsymbol{r} | D) = \text{ln} \, Z - \text{ln} \, \mathscr{L}
```

When sampling with methods like HMC, constants like ``\text{ln} \, Z`` can be neglected and ``-\text{ln} \; p(\boldsymbol{r} | D) \propto - \text{ln} \, \mathscr{L}`` such that the posterior is approximated by the likelihood surface.

Let us consider now what happens when we wish to do a variable transformation from ``r_j`` to ``\theta_j = \text{ln} (r_j)``. From above we can write the posterior as

```math
p(r_j | D) = \frac{p(D | r_j) \; p(r_j)}{Z} \\
```

Under the change of variables formula we can write

```math
\begin{aligned}
p(\theta_j | D) &= p(r_j | D) \left| \frac{d r_j}{d \theta_j} \right| \\
&= p(r_j | D) \left| \frac{d \theta_j}{d r_j} \right|^{-1}
\end{aligned}
```

where ``\left| \frac{d \theta_j}{d r_j} \right|^{-1}`` is often called the Jacobian correction. We choose ``\theta_j`` such that

```math
\begin{aligned}
\theta_j &= \text{ln} ( r_j ) \\
\left| \frac{d \theta_j}{d r_j} \right| &= \frac{1}{r_j} \\
r_j &= \text{exp} (\theta_j) \\
\end{aligned}
```

which leads to a posterior of

```math
p(\theta_j | D) = \text{exp} (\theta_j) \times p(\text{exp} (\theta_j) | D) = r_j \times p(r_j | D) \\
```

We can then write the product over the ``\theta_j`` as

```math
\begin{aligned}
p(\boldsymbol{\theta} | D) &= \frac{1}{Z} \; \prod_j r_j \; p(D | r_j) \; p(r_j) \\
-\text{ln} \, p(\boldsymbol{\theta} | D) &= \text{ln} \, Z - \sum_j \text{ln} \, (r_j) + \text{ln} \, p(D | r_j) + \text{ln} \, p(r_j) \\
&= \text{ln} \, Z - \sum_j \text{ln} \, p(D | r_j) + \text{ln} \, p(r_j) - \sum_j \theta_j \\
&= -\text{ln} \, p(\boldsymbol{r} | D) - \sum_j \theta_j \\
&= -\text{ln} \, p(\boldsymbol{r} | D) - \sum_j \text{ln} \, (r_j)
\end{aligned}
```

The choice of a logarithmic transformation means that the negative logarithm of the posterior (which is what HMC uses for its objective function) has this very simple form which allows for simple analytic gradients as well. Once samples of ``\theta`` have been obtained from this distribution via HMC or any other sampling method, they can be directly transformed back to the standard coefficients ``r_j = \text{exp}(\theta_j)``.

The method [`hmc_sample`](@ref) implements this approach for sampling the ``\theta_j`` coefficients; these samples can then be used to estimate random uncertainties on the derived star formation history.

```@docs
StarFormationHistories.hmc_sample
```

See the [DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl) documentation for more information on how to use the chains that are output by this method.

Inspection of the samples generated by `hmc_sample` shows that the posterior defined by the above model is typically smooth, well-behaved, and unimodal. In particular, we find that the sampled ``r_j`` for coefficients that are non-zero in the MLE are approximately Gaussian distributed while the logarithms of the sampled ``r_j`` are roughly Gaussian distributed for coefficients that are zero in the MLE; i.e.

```math
\begin{cases}
X_j \sim \mathcal{N}; & \hat r_j > 0 \\
\text{ln} \left( X_j \right) \sim \mathcal{N}; & \hat r_j = 0 \\
\end{cases}
```

where ``X_j`` are the samples of ``r_j`` obtained from the posterior and ``\hat r_j`` is the maximum likelihood estimate of ``r_j``. 

This indicates we may be able to approximate the posterior in the region surrounding the maximum a posteriori (MAP) value by the inverse of the Hessian matrix (see, e.g., [Dovi et al. 1991](https://doi.org/10.1016/0893-9659(91)90129-J)), allowing us to estimate parameter uncertainties very cheaply. The inverse of the Hessian matrix is exactly equal to the variance-covariance matrix of the parameters for a Gaussian probability distribution; for other probability distributions, the inverse of the Hessian approximates the variance-covariance matrix of the parameters when the second-order expansion defined by the Hessian at the maximum is a reasonable approximation to the real objective function being optimized. A particularly simple form arises when the logarithm of the objective is quadratic in the fitting parameters, as in the Gaussian case, because the second derivatives of the objective are constant and do not depend on the fitting parameters or the MAP estimate.

## Maximum a Posteriori Optimization

Direct computation of the Hessian and its inverse is expensive, so we'd like another way to obtain it. The first-order, quasi-Newton BFGS optimization algorithm provides such a method as it iteratively builds a dense approximation to the inverse Hessian using the change in the gradient of the objective, which we can compute analytically. It is, however, much less memory efficient than the LBFGS algorithm we use in [`StarFormationHistories.fit_templates_lbfgsb`](@ref). For moderate isochrone grids up to a few hundred model templates, this is not a problem. Beyond this it may be better to use [`StarFormationHistories.fit_templates_lbfgsb`](@ref) to obtain the MLE and [`hmc_sample`](@ref) to obtain posterior samples.

We implement this optimization scheme in [`fit_templates`](@ref), **which is our recommended method for unconstrained SFH fitting** (i.e., direct fitting of the ``r_j`` coefficients). See the next section for notes on more complicated, hierarchical models that can incorporate features like metallicity distribution functions.

```@docs
StarFormationHistories.fit_templates
StarFormationHistories.LogTransformFTResult
```

Once you have obtained stellar mass coefficients from the above methods, you can convert them into star formation rates and compute per-age mean metallicities with [`StarFormationHistories.calculate_cum_sfr`](@ref).

```@docs
StarFormationHistories.calculate_cum_sfr
```
