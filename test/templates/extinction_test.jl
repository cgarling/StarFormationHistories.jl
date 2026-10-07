import DelimitedFiles: readdlm
using Distributions: Uniform, Normal, Dirac, cdf, pdf, mean, var
using Test
import InitialMassFunctions: Kroupa2001
using StableRNGs: StableRNG
using StarFormationHistories: partial_cmd_smooth, ssp_components, match_extinction, UniformSum, prepare_extinction,
    Martin2016_complete, exp_photerr, NoBinaries, RandomBinaryPairs, binary_mass_fraction

@testset "Extinction distributions" begin
    @test prepare_extinction(Uniform(0.2, 1.2)) == Uniform(0.2, 1.2)
    # Unbounded sides are truncated at the 3σ-equivalent quantiles, and never below zero extinction
    @test all(extrema(prepare_extinction(Normal(0.7, 0.15))) .≈ (0.25, 1.15))
    @test extrema(prepare_extinction(Normal(0.1, 0.15)))[1] == 0
    @test_throws "point masses" prepare_extinction(Dirac(0.5))

    # The sum of two uniform draws is a trapezoid
    d = UniformSum(0.2, 1.0, 0.5)
    @test extrema(d) == (0.2, 1.7)
    x = range(0.2, 1.7; length=3001)
    @test cdf.(d, x) ≈ [0; cumsum((pdf.(d, x[1:end-1]) .+ pdf.(d, x[2:end])) ./ 2 .* step(x))] atol = 1e-6
    @test pdf(d, 0.9) ≈ 1 / 1.0 # flat top between 0.7 and 1.2
    s = rand(StableRNG(1), d, 10^5)
    @test mean(s) ≈ mean(d) atol = 0.01
    @test var(s) ≈ var(d) rtol = 0.02

    # MATCH: dAvy tapers linearly from full at t1 = 40 Myr to zero at t2 = 100 Myr
    @test match_extinction(0.2, 0.6, 0.5, 9.0) == Uniform(0.2, 0.8)
    @test match_extinction(0.2, 0.0, 0.5, 7.0) == Uniform(0.2, 0.7)
    @test all(extrema(match_extinction(0.2, 0.6, 0.5, 7.0)) .≈ (0.2, 1.3))
    @test all(extrema(match_extinction(0.2, 0.6, 0.5, log10(7e7))) .≈ (0.2, 1.05))
    @test_throws ArgumentError match_extinction(0.2, 0.0, 0.0, 7.0)
end

@testset "partial_cmd_smooth with differential extinction" begin
    isochrone, _ = readdlm(joinpath(@__DIR__, "../../data/isochrone.txt"), ' ', Float64, '\n'; header=true)
    m_ini = isochrone[:, 1]
    mags0 = [isochrone[:, i] for i in 2:4]
    dmod = 25.0
    edges = (range(-0.2, 1.2, length=75), range(dmod - 6.0, dmod + 5.0, length=100))
    comps = [m -> Martin2016_complete(m, 1.0, 28.5, 0.7), m -> Martin2016_complete(m, 1.0, 27.5, 0.7),
             m -> Martin2016_complete(m, 1.0, 26.5, 0.7)]
    errs = [m -> min(exp_photerr(m, 1.03, 15.0, 36.0, 0.02), 0.4), m -> min(exp_photerr(m, 1.03, 15.0, 35.0, 0.02), 0.4),
            m -> min(exp_photerr(m, 1.03, 15.0, 34.0, 0.02), 0.4)]
    imf = Kroupa2001(0.08, 100.0)
    # Synthetic reddening that depends on each star's color, with a change of slope at the knot A = 0.7
    color0 = mags0[1] .- mags0[2]
    R = [0.6 .+ 0.05 .* (color0 .- 0.5), 0.2 .+ 0.01 .* (color0 .- 0.5), fill(0.08, length(m_ini))]
    redden(A) = [m .+ r .* (A + 0.1 * max(A - 0.7, 0)) for (m, r) in zip(mags0, R)]
    knots = [0.0, 0.4, 0.7, 1.0, 1.5, 2.5]
    kmags = redden.(knots)
    plain(A; y=2, bm=NoBinaries(), c=comps) = partial_cmd_smooth(m_ini, redden(A), errs, y, (1, 2), imf, c; dmod, edges, binary_model=bm).weights
    new(dist; y=2, bm=NoBinaries(), c=comps, kw...) = partial_cmd_smooth(m_ini, kmags, knots, dist, errs, y, (1, 2), imf, c; dmod, edges, binary_model=bm, kw...).weights
    L1(ref, t) = sum(abs, t .- ref) / sum(ref)
    # Mean increase of the Poisson fit statistic for N stars drawn from `ref` and fit with `t`, 2N KL(ref || t); paths
    # end beyond the faint and red edges, so `t` is zero in some bins where `ref` has negligible, but nonzero, weight
    Δfit(ref, t; N=10^4) = (p = ref ./ sum(ref); q = t ./ sum(t); 2N * sum(p[i] * log(p[i] / max(q[i], floatmin())) for i in eachindex(p) if p[i] > 0))

    @testset "Narrow distributions reproduce the method without extinction" begin
        for y in (2, 1, 3)
            @test L1(plain(0.85; y), new(Uniform(0.85, 0.85 + 1e-9); y)) < 1e-6
        end
        # Binary systems are aggregated in cells along their paths, so agreement is approximate
        @test L1(plain(0.85; bm=RandomBinaryPairs(0.3)), new(Uniform(0.85, 0.85 + 1e-9); bm=RandomBinaryPairs(0.3))) < 1e-3
    end

    @testset "Accuracy against quadrature over extinction" begin
        for dist in (Uniform(0.2, 1.2), Normal(0.7, 0.15))
            d = prepare_extinction(dist)
            a = range(extrema(d)...; length=201)
            ref = sum((cdf(d, a[k+1]) - cdf(d, a[k])) .* plain((a[k] + a[k+1]) / 2) for k in 1:length(a)-1)
            # Δfit ≈ 0.01-0.02 at the default max_shift = 0.5, against a statistical scatter of the fit statistic of ~70
            Δ = Δfit(ref, new(dist))
            @test Δ < 0.05
            @test Δfit(ref, new(dist; max_shift=0.25)) < Δ / 4
        end
    end

    @testset "Components and photometric models" begin
        dist, bm = match_extinction(0.2, 0.5, 0.5, 7.5), RandomBinaryPairs(0.3)
        single, binary = ssp_components(m_ini, kmags, knots, dist, errs, 2, (1, 2), imf, comps; dmod, edges, binary_model=bm)
        φ = binary_mass_fraction(bm, imf)
        @test (1 - φ) .* single.weights .+ φ .* binary.weights ≈ new(dist; bm)
        @test single.weights ≈ new(dist)
        # A joint completeness model that is the product of the per-filter models gives the same template
        @test new(dist; bm, c=(m1, m2, m3) -> comps[1](m1) * comps[2](m2)) ≈ new(dist; bm)
    end

    @test_throws "bracket the support" new(Uniform(0.2, 3.0))
end
