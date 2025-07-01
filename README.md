# StarFormationHistories.jl

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://cgarling.github.io/StarFormationHistories.jl/stable/)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://cgarling.github.io/StarFormationHistories.jl/dev/)
[![Build Status](https://github.com/cgarling/StarFormationHistories.jl/workflows/CI/badge.svg)](https://github.com/cgarling/StarFormationHistories.jl/actions)
[![codecov](https://codecov.io/github/cgarling/StarFormationHistories.jl/graph/badge.svg?token=L69R23H29M)](https://codecov.io/github/cgarling/StarFormationHistories.jl)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)


This package implements methods for modelling observed Hess diagrams (which are just binned color-magnitude diagrams, also called CMDs) and using them to fit astrophysical star formation histories (SFHs). The paper describing the implemented methodologies has been accepted to ApJS (see below). This package also provides utilities for simulating CMDs given input SFHs and photometric error and completeness functions, which can be useful for planning observations and writing proposals. Please see our documentation (linked in the badge above) for more information. A rendered Jupyter notebook with example usage of this package is available [here](https://nbviewer.org/github/cgarling/StarFormationHistories.jl/blob/main/examples/fitting1.ipynb). This package requires stellar isochrones to operate -- these can be generated with [StellarTracks.jl](https://github.com/cgarling/StellarTracks.jl) or sourced from online resources like [the CMD webform](http://stev.oapd.inaf.it/cgi-bin/cmd) for PARSEC models.

## Installation

This package is registered to Julia's General registry and can be installed via Pkg from the Julia REPL by executing

```julia
import Pkg;
Pkg.add("StarFormationHistories");
```

## Paper and Citation
The initial paper describing the methodologies used in this package is available on ArXiv [here](http://arxiv.org/abs/2407.19534) with permanent DOI identifier [here](https://doi.org/10.3847/1538-4365/adbb64). The ApJS published version is available [here](doi.org/10.3847/1538-4365/adbb64). We ask that published work using this package cite this paper.

The software has [its own DOI](https://doi.org/10.5281/zenodo.14963317) issued through Zenodo which can be used to refer to the software separately from the academic work. Zenodo issues version-specific DOIs that are useful for referring to exact versions used for particular work.

Having (and using) both identifiers is useful. For example, in the text of an academic paper we ask that you reference the above journal paper that outlines the scientific foundations of the software. However, many journals now include special *software* sections after the main text where you are asked to reference software you used. If you used StarFormationHistories.jl, it would be appropriate to reference the software via the Zenodo DOI in the *software* reference section.

## Template Construction

Given an isochrone, one of the main challenges in the SFH fitting process is generating a template (sometimes called a partial CMD) that contains the expected number of stars in each Hess diagram bin per unit solar mass of stars formed (or per unit of star formation rate). The following figure, produced by `examples/templates/smooth_template.jl`, shows a mock population at left sampled from an isochrone assuming photometric error and completeness functions typical of the JWST/NIRCAM imaging obtained as part of the Resolved Stellar Populations Early Release Science Program ([Weisz et al. 2023](https://ui.adsabs.harvard.edu/abs/2023ApJS..268...15W), [Weisz et al. 2024](https://ui.adsabs.harvard.edu/abs/2024ApJS..271...47W)). We discretize it to form a Hess diagram in panel B. In panel C we show the model template we construct from this isochrone. Panel D shows the residual significance, which is the difference between the observed bin counts and those in our template, divided by the Poisson error. The distribution of residuals shows that the template is a robust model of the distribution of the mock population in the Hess diagram space. 

![template_compare](https://github.com/cgarling/StarFormationHistories.jl/assets/20730107/55720670-d508-4102-894a-fe8a81033670)

Below we additionally show example templates for different combinations of stellar population age and metallicity that bracket the range typically considered in these analyses.

![template_example](https://github.com/cgarling/StarFormationHistories.jl/assets/20730107/fc8f0b8f-0c96-43fd-a8bd-42621997a0b6)

## SFH Fitting

Once templates have been generated for a reasonable set of isochrones, they can be used to estimate the SFH of an observed population by modelling the observed Hess diagram as a linear combination of the templates. The coefficients on the linear combination are simply the desired star formation rates. While this can work, and we provide fitting methods that support this use, this approach is crude as it does not guarantee that the solution has a physically realistic age-metallicity relation (AMR). We additionally define an API for describing general AMR models as well as mass-metallicity models (MZRs) for constraining the metallicity evolution of the population. We provide a few parametric AMRs and MZRs and describe how to define custom models.

## Acknowledgements
Support for this work was provided by the Owens Family Foundation and by NASA through grant HST-AR-17560 from the Space Telescope Science Institute, which is operated by AURA, Inc., under NASA contract NAS5-26555.
