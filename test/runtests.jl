import StarFormationHistories as SFH
using Test, SafeTestsets
using Logging: with_logger, ConsoleLogger, Error

# Run doctests first
import Documenter: DocMeta, doctest
DocMeta.setdocmeta!(SFH, :DocTestSetup, :(using StarFormationHistories); recursive=true)
doctest(SFH)

@testset verbose=true "StarFormationHistories.jl" begin
    @safetestset "CMD Simulation" include("cmd_simulation.jl")

    #####################################################################
    @testset verbose=true "SFH Fitting" begin
        @safetestset "Core Fitting Functions" include("fitting/fitting_core.jl")

        # Benchmarking
        # let x=[1.0], M=[Float64[0 0 0; 0 0 0; 1 1 1]], N=Int64[0 0 0; 0 0 0; 3 3 3], C=zeros(3,3), G=[1.0]
        #     @btime SFH.loglikelihood($M[1], $N)
        #     @btime SFH.∇loglikelihood($M[1], $M[1], $N) # @btime SFH.∇loglikelihood($x, $M, $N)
        #     @btime SFH.fg($M[1], $M[1], $N)
        #     @btime SFH.fg!($true, $G, $x, $M, $N, $C)
        # end

        @safetestset "Template Kernels" include("templates/kernel_test.jl")
        @safetestset "Template Construction" include("templates/template_test.jl")

        @testset verbose=true "Solving + Sampling" begin
            @safetestset "Basic Linear Combinations" include("fitting/basic_linear_combinations.jl")
            @safetestset "Age-Metallicity Relations" include("fitting/amr_test.jl")
            @safetestset "Mass-Metallicity Relations" include("fitting/mzr_test.jl")
            @safetestset "Fixed AMR" include("fitting/fixed_amr.jl")
        end
    end


    @testset "utilities" begin
        # Artifical star tests use extensions, requires Julia >= 1.9
        if VERSION >= v"1.9"
            # We are intentionally causing some warnings, they are not significant
            with_logger(ConsoleLogger(Error)) do
                @safetestset "process_ASTs" include("utilities/process_ASTs_test.jl")
            end
        end

        @safetestset "Utilities" include("utilities/utilities_test.jl")
    end
end
