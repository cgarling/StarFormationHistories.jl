import DelimitedFiles: readdlm
using Distributions: Normal, ContinuousUnivariateDistribution
import Distributions: pdf
using Test
import StatsBase: Histogram, fit
import InitialMassFunctions: Kroupa2001
using StableRNGs: StableRNG
import StarFormationHistories
using StarFormationHistories: bin_cmd, mini_spacing, midpoints, dispatch_imf, mean, partial_cmd_smooth, Martin2016_complete, exp_photerr, NoBinaries, RandomBinaryPairs, BinaryMassRatio, generate_stars_mass, model_cmd


@testset "calculate_weights" begin
    # Trapezoidal rule is exact for a constant IMF and linear completeness
    @test StarFormationHistories.calculate_weights([1.0, 2.0, 3.0], [0.0, 0.5, 1.0], m -> 1.0, 1, 1) == [0.25, 0.75]
end

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
    # Unsorted input gives the same result as sorted input
    p = [3, 1, 5, 2, 4]
    @test mini_spacing([0.08, 0.10, 0.12, 0.14, 0.16][p], [1.0, 0.99, 0.98, 0.97, 0.96][p], [13.545, 12.899, 12.355, 11.459, 10.947][p], 0.1, true) == (result, spacing)
end

@testset "bin_cmd" begin
    h = bin_cmd([0.1, 0.25, 0.25, 0.9], [20.1, 20.6, 20.6, 21.5]; weights=[1.0, 2.0, 0.5, 1.0], edges=(0.0:0.5:1.0, 20.0:1.0:22.0))
    @test h isa Histogram
    @test h.weights == [3.5 0.0; 0.0 1.0]
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
        for binary_model in (NoBinaries(), RandomBinaryPairs(0.3), BinaryMassRatio(0.3))
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

    @testset "Joint photometric models" begin
        filters = [F090W, F150W, F277W]
        comp, err = [F090W_complete, F150W_complete, F277W_complete], [F090W_error, F150W_error, F277W_error]
        bias = [m -> 0.01 * (m - 25), m -> -0.02 * (m - 25), zero]
        # Joint models that are products of the per-filter models give the same templates
        jerr(ms...) = map((f, m) -> f(m), Tuple(err), ms)
        jbias(ms...) = map((f, m) -> f(m), Tuple(bias), ms)
        for (y_index, color_indices) in ((2, (1,2)), (1, (1,2)), (3, (1,2)))
            # Per-filter completeness uses only the filters in the Hess diagram
            jcomp(ms...) = prod(comp[k](ms[k]) for k in union(color_indices, y_index))
            for binary_model in (NoBinaries(), RandomBinaryPairs(0.3), BinaryMassRatio(0.3))
                tmpl(e, c, b) = partial_cmd_smooth(m_ini, filters, e, y_index, color_indices, imf, c, b; dmod=distmod, normalize_value=template_norm, edges, binary_model).weights
                ref = tmpl(err, comp, bias)
                @test tmpl(err, jcomp, bias) ≈ ref
                @test tmpl(jerr, comp, jbias) ≈ ref
                @test tmpl(jerr, jcomp, jbias) ≈ ref
            end
        end
        # A non-separable selection (a color-dependent magnitude limit) removes stars from the template
        diag_comp(m1, m2) = (m1 + m2) / 2 < distmod + 1.5
        sep = partial_cmd_smooth(m_ini, filters[1:2], err[1:2], 2, (1,2), imf; dmod=distmod, normalize_value=template_norm, edges).weights
        joint = partial_cmd_smooth(m_ini, filters[1:2], err[1:2], 2, (1,2), imf, diag_comp; dmod=distmod, normalize_value=template_norm, edges).weights
        @test 0 < sum(joint) < sum(sep)
        # Templates and mock observations with the same joint completeness model agree
        smooth_comp(m1, m2) = (1 - tanh(((m1 + m2) / 2 - (distmod + 2.5)) / 0.2)) / 2
        smooth_tmpl = partial_cmd_smooth(m_ini, filters[1:2], err[1:2], 2, (1,2), imf, smooth_comp; dmod=distmod, normalize_value=3e6, edges).weights
        _, mock_mags = generate_stars_mass(copy(m_ini), filters[1:2], ["F090W", "F150W"], 3e6, imf; dist_mod=distmod, rng=StableRNG(1), binary_model=NoBinaries())
        obs = model_cmd(mock_mags, err[1:2], smooth_comp; rng=StableRNG(2))
        nobs = count(m -> (first(edges[1]) <= m[1] - m[2] < last(edges[1])) & (first(edges[2]) <= m[2] < last(edges[2])), obs)
        @test nobs ≈ sum(smooth_tmpl) rtol=0.03
        # Per-filter models must have one callable per filter
        @test_throws ArgumentError partial_cmd_smooth(m_ini, filters, err[1:2], 2, (1,2), imf, diag_comp; dmod=distmod, normalize_value=template_norm, edges)
    end

    @testset "Templates match mock observations" begin
        # Templates should agree with the Hess diagram of a mock catalog sampled with the same binary model to within
        # Poisson noise, including near the edges of the Hess diagram where stars scatter in from outside it
        filters, err = [F090W, F150W, F277W], [F090W_error, F150W_error, F277W_error]
        mock_mass = 5e6
        for (binary_model, y_index) in ((NoBinaries(), 2), (RandomBinaryPairs(0.3), 2), (RandomBinaryPairs(0.3), 3), (BinaryMassRatio(0.3), 1), (BinaryMassRatio(0.3), 3))
            template = partial_cmd_smooth(m_ini, filters, err, y_index, (1,2), imf; dmod=distmod, normalize_value=mock_mass, edges, binary_model).weights
            _, mock_mags = generate_stars_mass(copy(m_ini), filters, ["F090W", "F150W", "F277W"], mock_mass, imf; dist_mod=distmod, rng=StableRNG(1), binary_model, mag_lim=last(edges[2]) + 1, mag_lim_name=["F090W", "F150W", "F277W"][y_index])
            obs = model_cmd(mock_mags, err, [one, one, one]; rng=StableRNG(2))
            mock = fit(Histogram, ([m[1] - m[2] for m in obs], [m[y_index] for m in obs]), edges; closed=:left).weights
            @test sum(mock) ≈ sum(template) rtol=0.01
            good = template .> 20
            pulls = (mock[good] .- template[good]) ./ sqrt.(template[good])
            @test sqrt(sum(abs2, pulls) / length(pulls)) < 1.15
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
