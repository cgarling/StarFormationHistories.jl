using Documenter
using StarFormationHistories

# Run examples
import PyPlot as plt
plt.ioff()
ENV["MPLBACKEND"] = "agg"
# Run examples to generate plots
# Set environment variable to save figures
ENV["DOCSBUILD"] = "true"
@info "Running example: smooth_template.jl"
include("../examples/templates/smooth_template.jl")
@info "Running example: smooth_template_binaries.jl"
include("../examples/templates/smooth_template_binaries.jl")
@info "Running example: kernels_example.jl"
include("../examples/templates/kernels_example.jl")
@info "Finished examples"
# Can't move yet as makedocs will clear the build folder.
# Moving and showing in docs/src/fitting/fitting_intro.md.

###########################################################

# The `format` below makes it so that urls are set to "pretty" if you are pushing them to a hosting service, and basic if you are just using them locally to make browsing easier.

# We check link validity with `linkcheck=true`, but we don't want this to fail the build
# so we add `:linkcheck` to `warnonly`. Additionally, we are setting
# `modules = [StarFormationHistories]` so a warning will be raised if any inline
# documentation strings are not included in the document. In v1.0 of Documenter, this warning
# will raise an error and prevent running. By adding `:missing_docs` to `warnonly`, we will
# see these warnings but they will not raise an error.

makedocs(
    sitename = "StarFormationHistories.jl",
    modules = [StarFormationHistories],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true",
                             size_threshold_warn = 409600, # v1.0.0 default: 102400 (bytes)
                             size_threshold = 819200,      # v1.0.0 default: 204800 (bytes)
                             example_size_threshold=0),    # Write all @example to file
    authors = "Chris Garling",
    pages = ["index.md",
             "Deriving Star Formation Histories from Hess Diagrams" =>
                 ["fitting/fitting_intro.md",
                  "fitting/unconstrained.md",
                  "Constrained Metallicity Evolution" => 
                      ["fitting/linear_amr.md",
                       "fitting/log_amr.md",
                       "fitting/fixed_amr.md",
                       "MZRs" =>
                           ["fitting/MZR/MZR.md",
                            "fitting/MZR/MZR_old.md"],
                       "fitting/dispersion_models.md"],
                  "Internals" => ["fitting/internals.md",
                                  "fitting/kernels.md"]],
             "examples.md",
             "simulate.md",
             "binaries.md",
             "helpers.md",
             "doc_index.md"],
    doctest = false,
    linkcheck = true,
    warnonly = [:missing_docs, :linkcheck]
)

deploydocs(;
    repo = "github.com/cgarling/StarFormationHistories.jl.git",
    versions = ["stable" => "v^", "v#.#"],
    push_preview=true,
)
