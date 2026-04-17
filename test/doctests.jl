# Run doctests

import StarFormationHistories
import Documenter: DocMeta, doctest
DocMeta.setdocmeta!(StarFormationHistories, :DocTestSetup, :(using StarFormationHistories); recursive=true)
doctest(StarFormationHistories)