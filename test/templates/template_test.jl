using Distributions: Normal, ContinuousUnivariateDistribution
import Distributions: pdf
using Test
using StarFormationHistories: mini_spacing, dispatch_imf, mean


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
