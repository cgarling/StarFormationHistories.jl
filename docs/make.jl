using Documenter
using StarFormationHistories

# The `format` below makes it so that urls are set to "pretty" if you are pushing them to a hosting service, and basic if you are just using them locally to make browsing easier.

# DocMeta.setdocmeta!(StarFormationHistories, :DocTestSetup, :(using StarFormationHistories; import Unitful; import UnitfulAstro); recursive=true)
DocMeta.setdocmeta!(StarFormationHistories, :DocTestSetup, :(using StarFormationHistories); recursive=true)

# We check link validity with `linkcheck=true`, but we don't want this to fail the build so we add `:linkcheck` to `warnonly`.
# Additionally, we are setting `modules = [StarFormationHistories]` so a warning will be raised if any inline documentation strings are not included in the document.
# In v1.0 of Documenter, this warning will raise an error and prevent running. By adding `:missing_docs` to `warnonly`, we will see these warnings but they will not raise an error.

makedocs(
    sitename = "StarFormationHistories.jl",
    modules = [StarFormationHistories],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true",
                             size_threshold_warn = 409600, # v1.0.0 default: 102400 (bytes)
                             size_threshold = 819200),     # v1.0.0 default: 204800 (bytes)
    authors = "Chris Garling",
    pages = ["index.md","simulate.md","fitting.md","binaries.md","helpers.md","doc_index.md"],
    doctest = true,
    linkcheck = true,
    warnonly = [:missing_docs, :linkcheck]
)

# deploydocs(;
#     repo = "github.com/cgarling/StarFormationHistories.jl.git",
#     versions = ["stable" => "v^", "v#.#"],
#     push_preview=true,
# )
