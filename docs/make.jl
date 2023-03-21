using Documenter
using StarFormationHistories

# The `format` below makes it so that urls are set to "pretty" if you are pushing them to a hosting service, and basic if you are just using them locally to make browsing easier.

# DocMeta.setdocmeta!(StarFormationHistories, :DocTestSetup, :(using StarFormationHistories; import Unitful; import UnitfulAstro); recursive=true)
DocMeta.setdocmeta!(StarFormationHistories, :DocTestSetup, :(using StarFormationHistories); recursive=true)

makedocs(
    sitename="StarFormationHistories.jl",
    modules = [StarFormationHistories],
    format = Documenter.HTML(;prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "Chris Garling",
    pages = ["index.md","simulate.md","fitting.md","binaries.md","helpers.md","doc_index.md"],
    doctest=true
)

# deploydocs(;
#     repo = "github.com/cgarling/StarFormationHistories.jl.git",
#     versions = ["stable" => "v^", "v#.#"],
#     push_preview=true,
# )
