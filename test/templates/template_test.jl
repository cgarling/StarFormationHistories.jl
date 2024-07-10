import DelimitedFiles: readdlm
using Distributions: Normal, ContinuousUnivariateDistribution
import Distributions: pdf
using Test
import StatsBase: Histogram
import InitialMassFunctions: Kroupa2001
using StarFormationHistories: mini_spacing, midpoints, dispatch_imf, mean, partial_cmd_smooth, Martin2016_complete, exp_photerr, NoBinaries, RandomBinaryPairs


@testset "mini_spacing" begin
    result = mini_spacing([0.08, 0.10, 0.12, 0.14, 0.16],
                          [1.0, 0.99, 0.98, 0.97, 0.96],
                          [13.545, 12.899, 12.355, 11.459, 10.947],
                          0.1, false)
    @test result isa Vector{Float64}
    @test length(result) > 5
    @test maximum(diff(result)) < 0.1
    result, spacing = mini_spacing([0.08, 0.10, 0.12, 0.14, 0.16],
                                   [1.0, 0.99, 0.98, 0.97, 0.96],
                                   [13.545, 12.899, 12.355, 11.459, 10.947],
                                   0.1, true)
    @test spacing isa Vector{Float64}
    @test spacing ≈ diff(result)
end

@testset "midpoints" begin
    @test midpoints(0.5:0.1:1.0) ≈ 0.55:0.1:0.95
    @test midpoints(1.0:-0.1:0.5) ≈ 0.95:-0.1:0.55
    @test midpoints(collect(0.5:0.1:1.0), true) ≈ 0.55:0.1:0.95
    @test midpoints(collect(1.0:-0.1:0.5), true) ≈ 0.95:-0.1:0.55
    @test midpoints([1.0,2.0,2.2,2.1]) ≈ [1.5, 2.1, 2.15]
    @test midpoints([1.0,2.0,2.2,2.1]) ≈ [1.5, 2.1, 2.15]
end

# Make a test type that is a duplicate of Normal
struct TestType{T,S} <: ContinuousUnivariateDistribution # Distribution{Univariate, Continuous}
    μ::T
    σ::S
end
pdf(d::TestType, x::Real) = pdf(Normal(d.μ, d.σ), x)
Base.extrema(d::TestType) = (-Inf, Inf)

@testset "dispatch_imf" begin
    @test dispatch_imf(TestType(0.0,1.0), 1.0) == pdf(Normal(0.0,1.0), 1.0)
    testfunc(x) = x^2
    @test dispatch_imf(testfunc, 2.0) == 4.0
    testdist = Normal(0.0, 1.0)
    @test dispatch_imf(testdist, 1.0) == pdf(testdist, 1.0)
end

@testset "mean" begin
    @test mean(TestType(1.0,1.0)) ≈ mean(Normal(1.0,1.0))
end

@testset "partial_cmd_smooth" begin
    # Load example isochrone
    isochrone, mag_names = readdlm(joinpath(@__DIR__, "../../data/isochrone.txt"), ' ',
                                   Float64, '\n'; header=true)
    # Unpack
    m_ini = isochrone[:,1]
    F090W = isochrone[:,2]
    F150W = isochrone[:,3]
    F277W = isochrone[:,4]
    # Set distance modulus
    distmod = 25.0
    # Set bins for Hess diagram
    edges = (range(-0.2, 1.2, length=75),
             range(distmod-6.0, distmod+5.0, length=100))
    # Set total stellar mass to normalize template to
    template_norm = 1e7
    # Construct error and completeness functions
    F090W_complete(m) = Martin2016_complete(m, 1.0, 28.5, 0.7)
    F150W_complete(m) = Martin2016_complete(m, 1.0, 27.5, 0.7)
    F277W_complete(m) = Martin2016_complete(m, 1.0, 26.5, 0.7)
    F090W_error(m) = min(exp_photerr(m, 1.03, 15.0, 36.0, 0.02), 0.4)
    F150W_error(m) = min(exp_photerr(m, 1.03, 15.0, 35.0, 0.02), 0.4)
    F277W_error(m) = min(exp_photerr(m, 1.03, 15.0, 34.0, 0.02), 0.4)
    # Set IMF
    imf = Kroupa2001(0.08, 100.0)
    # Construct template
    for (y_index, color_indices) in ((2, (1,2)), (1, (1,2)), (3, (1,2)))
        if !in(y_index, color_indices) # RandomBinaryPairs not implemented for this case
            binary_models = (NoBinaries(),)
        else
            binary_models = (NoBinaries(), RandomBinaryPairs(0.3))
        end
        for binary_model in binary_models
            template = partial_cmd_smooth(m_ini,
                                          [F090W, F150W, F277W],
                                          [F090W_error, F150W_error, F277W_error],
                                          y_index,
                                          color_indices,
                                          imf,
                                          [F090W_complete, F150W_complete, F277W_complete]; 
                                          dmod=distmod,
                                          normalize_value=template_norm,
                                          edges=edges,
                                          binary_model=binary_model)
            @test template isa Histogram
            data = template.weights
            @test data isa Matrix{Float64}
            @test any(!=(0), data)
            @test size(data) == (length(edges[1])-1, length(edges[2])-1)
            @test template.edges == edges
            @test template.isdensity == false
        end
    end
end
# 2.5 ms
# ~1.25 ms in bin_cmd_smooth
# ~0.75 ms calculating weights
# ~0.75 ms interpolating isochrone mags and calculating errors and completeness values
# @benchmark partial_cmd_smooth($m_ini,
#                               $[F090W, F150W],
#                               $[F090W_error, F150W_error],
#                               $2,
#                               $[1,2],
#                               $imf,
#                               $[F090W_complete, F150W_complete];
#                               dmod=$distmod,
#                               normalize_value=$template_norm,
#                               edges=$edges)
