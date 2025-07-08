using Documenter
import Changelog
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

Changelog.generate(
    Changelog.Documenter(),
    joinpath(@__DIR__, "..", "CHANGELOG.md"),
    joinpath(@__DIR__, "src", "release-notes.md");
    repo = "cgarling/StarFormationHistories.jl",
    branch = "main")

linkcheck_ignore = [
    # We'll ignore the links to tags in CHANGELOG.md, since when you tag
    # a release, the release link does not exist yet, and this will cause the linkcheck
    # CI job to fail on the PR that tags a new release.
    r"https://github.com/JuliaDocs/Documenter.jl/releases/tag/v\d+.\d+.\d+",
]

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
                      ["fitting/hierarchical/overview.md",
                       "AMRs" =>
                           ["fitting/hierarchical/linear_amr.md",
                            "fitting/hierarchical/log_amr.md",
                            "fitting/hierarchical/fixed_amr.md"],
                       "MZRs" =>
                           ["fitting/hierarchical/MZR/MZR.md"],
                       "fitting/hierarchical/dispersion_models.md"],
                  "Internals" => ["fitting/internals.md",
                                  "fitting/kernels.md"]],
             "examples.md",
             "simulate.md",
             "binaries.md",
             "helpers.md",
             "release-notes.md",
             "doc_index.md"],
    doctest = false,
    linkcheck = true,
    linkcheck_ignore = linkcheck_ignore,
    warnonly = [:missing_docs, :linkcheck]
)

deploydocs(;
    repo = "github.com/cgarling/StarFormationHistories.jl.git",
    versions = ["stable" => "v^", "v#.#"],
    push_preview=true,
)
