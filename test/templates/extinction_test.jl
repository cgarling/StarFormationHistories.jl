import DelimitedFiles: readdlm
using Distributions: Uniform, Normal, Laplace, Dirac, cdf, ccdf, cquantile, pdf, quantile, mean, var, truncated
using QuadGK: quadgk
using Test
import InitialMassFunctions: Kroupa2001, partial_mean
using StableRNGs: StableRNG
using StarFormationHistories: partial_cmd_smooth, ssp_components, match_extinction, UniformSum, prepare_extinction,
    ExponentialDiskExtinction, UniformConvolution, cdf_integral, segment_moments, Martin2016_complete, exp_photerr, NoBinaries,
    RandomBinaryPairs, binary_mass_fraction

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

    # Exponential disks: perfectly mixed stars and dust give MATCH's uniform distribution
    x = range(0.1, 1.3; length=101)
    @test cdf.(ExponentialDiskExtinction(0.2, 1.0, 1.0), x) ≈ cdf.(Uniform(0.2, 1.2), x)
    @test pdf.(ExponentialDiskExtinction(0.2, 1.0, 1.0), x) ≈ pdf.(Uniform(0.2, 1.2), x)
    u = range(0.01, 0.99; length=99)
    s = 3.0
    @test pdf.(ExponentialDiskExtinction(0.2, 1.0, s), 0.2 .+ u) ≈ (2 .* min.(u, 1 .- u)) .^ (1 / s - 1) ./ s
    @test pdf(ExponentialDiskExtinction(0.2, 1.0, s), 0.2) == Inf
    @test pdf(ExponentialDiskExtinction(0.2, 1.0, 0.3), 0.2) == 0
    for s in (0.3, 1.0, 20.0), f_red in (0.2, 0.5, 0.8)
        d = ExponentialDiskExtinction(0.2, 1.0, s; f_red)
        # U is the fraction of a Laplace(z0, 1) dust layer in front of a star drawn from Laplace(0, s), and a fraction
        # f_red of the stars lie behind the dust midplane
        @test cdf.(d, 0.2 .+ u) ≈ ccdf.(Laplace(0, s), cquantile.(Laplace(d.z0, 1), u)) rtol = 1e-12
        @test cdf(Laplace(0, s), d.z0) ≈ f_red
        @test cdf(d, 0.7) ≈ 1 - f_red
        F(a) = cdf(d, a)
        # The density can diverge at both ends and has kinks where the dust and stellar midplanes lie
        edges = sort([0.2, 0.7, 1.2, clamp(0.2 + exp(d.z0) / 2, 0.2, 1.2), clamp(1.2 - exp(-d.z0) / 2, 0.2, 1.2)])
        ∫(f, x) = quadgk(f, filter(<(x), edges)..., x; rtol=1e-12)[1]
        @test mean(d) ≈ 1.2 - ∫(F, 1.2)
        @test var(d) ≈ 2 * ∫(a -> (1.2 - a) * F(a), 1.2) - (1.2 - mean(d))^2
        @test partial_mean(d, 0.9) ≈ 0.9 * F(0.9) - ∫(F, 0.9)
        @test cdf_integral(d, 0.9) ≈ ∫(F, 0.9) && cdf_integral(d, 1.5) ≈ 1.5 - mean(d) && cdf_integral(d, 0.1) == 0
        s < 20 && @test cdf.(d, quantile.(d, u)) ≈ u # For s = 20, most quantiles round to the ends of the support
        smp = rand(StableRNG(1), d, 10^5)
        @test mean(smp) ≈ mean(d) atol = 0.01
        @test sum(smp .<= 0.5) / 10^5 ≈ cdf(d, 0.5) atol = 0.01
        # Moments from the cdf across a short interval at the end of the support, where the density can diverge,
        # against quadrature of the quantile function
        F0, F1 = F(0.2), F(0.25)
        Ā = quadgk(p -> quantile(d, p), F0, F1)[1] / (F1 - F0)
        v = quadgk(p -> (quantile(d, p) - Ā)^2, F0, F1)[1] / (F1 - F0)
        @test all(segment_moments(d, 0.2, 0.25, F0, F1) .≈ (Ā, v))
    end
    d32 = ExponentialDiskExtinction(0.2f0, 1.0f0, 3.0f0; f_red=0.7f0)
    @test all(f(d, 0.9f0) isa Float32 for f in (cdf, pdf, partial_mean), d in (d32, UniformConvolution(d32, 0.5f0)) if !(f === partial_mean && d isa UniformConvolution))
    @test mean(d32) isa Float32 && var(d32) isa Float32
    @test_throws "f_red" ExponentialDiskExtinction(0.2, 1.0, 3.0; f_red=1.0)
    @test_throws "dAv > 0" ExponentialDiskExtinction(0.2, 0.0, 3.0)
    @test_throws "σ >= 0" ExponentialDiskExtinction(0.2, 1.0, 3.0; σ=-0.1)

    # Log-normal total extinction of the dust: A = Av + dAv exp(σ z) U, with z ~ Normal(0, 1) truncated at 3 and U the
    # fraction of the dust in front of a star, as for a fixed total extinction
    for (s, f_red, σ) in ((3.0, 0.7, 0.4), (20.0, 0.8, 0.3))
        d = ExponentialDiskExtinction(0.2, 0.7, s; f_red, σ)
        @test all(extrema(d) .≈ (0.2, 0.2 + 0.7 * exp(3σ)))
        rng = StableRNG(1)
        z = filter(<=(3), randn(rng, 2 * 10^6))[1:10^6]
        smp = 0.2 .+ 0.7 .* exp.(σ .* z) .* rand(rng, ExponentialDiskExtinction(0.0, 1.0, s; f_red), 10^6)
        x = range(0.25, 1.5; length=11)
        @test [count(<=(xi), smp) / 10^6 for xi in x] ≈ cdf.(d, x) atol = 0.002
        @test mean(smp) ≈ mean(d) atol = 0.005
        @test var(smp) ≈ var(d) rtol = 0.02
        @test mean(rand(StableRNG(2), d, 10^5)) ≈ mean(d) atol = 0.005
        @test [(cdf(d, xi + 1e-5) - cdf(d, xi - 1e-5)) / 2e-5 for xi in x] ≈ pdf.(d, x) rtol = 1e-4
        @test partial_mean(d, 0.9) ≈ 0.9 * cdf(d, 0.9) - quadgk(a -> cdf(d, a), 0.2, 0.9; rtol=1e-10)[1] rtol = 1e-8
        @test cdf_integral(d, 0.9) ≈ quadgk(a -> cdf(d, a), 0.2, 0.9; rtol=1e-10)[1] rtol = 1e-8
        @test cdf_integral(d, maximum(d) + 0.3) ≈ maximum(d) + 0.3 - mean(d)
        @test cdf(match_extinction(d, 0.4, 7.0), 0.9) ≈ quadgk(a -> cdf(d, a), 0.5, 0.9; rtol=1e-10)[1] / 0.4 rtol = 1e-8
        @test cdf.(d, quantile.(d, [0.3, 0.6, 0.95])) ≈ [0.3, 0.6, 0.95]
    end
    # Sub-segment moments across the support against quadrature of the cdf, for the disk and with the additional
    # extinction of young stars; f_red = 0.7 and 0.8 put the kink of the density of U at the dust midplane close to u = 1
    for (s, f_red, σ) in ((3.0, 0.7, 0.4), (20.0, 0.8, 0.3), (3.0, 0.7, 0.3), (3.0, 0.8, 0.3))
        d = ExponentialDiskExtinction(0.2, 0.7, s; f_red, σ)
        for dd in (d, match_extinction(d, 0.4, 7.0)), b0 in 0.25:0.1:1.55
            b1 = b0 + 0.05
            F0, F1 = cdf(dd, b0), cdf(dd, b1)
            m1 = quadgk(a -> F1 - cdf(dd, a), b0, b1; rtol=1e-10)[1] / (F1 - F0)
            m2 = quadgk(a -> 2 * (a - b0) * (F1 - cdf(dd, a)), b0, b1; rtol=1e-10)[1] / (F1 - F0)
            Ā, v = segment_moments(dd, b0, b1, F0, F1)
            @test abs(Ā - b0 - m1) < 0.005 * 0.05
            @test v ≈ m2 - m1^2 rtol = 0.01
        end
    end
    # In the limit of a thin dust layer, a fraction 1 - f_red of the stars has extinction Av
    @test cdf(ExponentialDiskExtinction(0.2, 0.7, 1000.0; f_red=0.7, σ=0.4), 0.2 + 1e-6) ≈ 0.3 atol = 0.01
    d32σ = ExponentialDiskExtinction(0.2f0, 0.7f0, 3.0f0; σ=0.4f0)
    @test cdf(d32σ, 0.6f0) isa Float32 && mean(d32σ) isa Float32 && var(d32σ) isa Float32

    # Additional extinction of young stars: X + Uniform(0, w), with the cdf from the closed form of partial_mean
    base = ExponentialDiskExtinction(0.2, 1.0, 20.0; f_red=0.8)
    @test match_extinction(base, 0.5, 9.0) === base
    d = match_extinction(base, 0.5, 7.0)
    @test d isa UniformConvolution && d.w == 0.5 && extrema(d) == (0.2, 1.7)
    x = range(0.15, 1.75; length=161)
    kinks = [0.2, 0.2 + exp(base.z0) / 2, 0.7, 1.2 - exp(-base.z0) / 2, 1.2]
    @test cdf.(d, x) ≈ [quadgk(a -> cdf(base, a), xi - 0.5, filter(k -> xi - 0.5 < k < xi, kinks)..., xi; rtol=1e-12)[1] / 0.5 for xi in x] atol = 1e-9
    @test maximum(pdf.(d, x)) <= 1 / 0.5
    @test cdf.(match_extinction(Uniform(0.2, 1.2), 0.5, 7.0), x) ≈ cdf.(UniformSum(0.2, 1.0, 0.5), x)
    smp = rand(StableRNG(1), d, 10^5)
    @test mean(smp) ≈ mean(d) atol = 0.01
    @test var(smp) ≈ var(d) rtol = 0.02
    F0, F1 = cdf(d, 0.6), cdf(d, 0.75)
    Ā = quadgk(a -> a * pdf(d, a), 0.6, 0.7, 0.75)[1] / (F1 - F0)
    @test all(segment_moments(d, 0.6, 0.75, F0, F1) .≈ (Ā, quadgk(a -> (a - Ā)^2 * pdf(d, a), 0.6, 0.7, 0.75)[1] / (F1 - F0)))
    @test_throws "bounded" UniformConvolution(Normal(), 0.5)
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
        disk = ExponentialDiskExtinction(0.2, 1.0, 20.0; f_red=0.8)
        diskσ = ExponentialDiskExtinction(0.2, 0.7, 3.0; f_red=0.7, σ=0.3)
        for dist in (Uniform(0.2, 1.2), Normal(0.7, 0.15), disk, match_extinction(disk, 0.5, 7.0), diskσ)
            d = prepare_extinction(dist)
            # Points at equal steps in extinction and, for the exponential disk, whose density diverges at the ends of
            # the support, in probability
            a = sort(unique([range(extrema(d)...; length=201); d isa ExponentialDiskExtinction ? quantile.(d, (1:199) ./ 200) : []]))
            ref = sum((cdf(d, a[k+1]) - cdf(d, a[k])) .* plain(d isa ExponentialDiskExtinction ? quantile(d, (cdf(d, a[k]) + cdf(d, a[k+1])) / 2) : (a[k] + a[k+1]) / 2)
                      for k in 1:length(a)-1 if cdf(d, a[k+1]) > cdf(d, a[k]))
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

    @testset "Paths that leave the diagram" begin
        # Paths end beyond the faint and red edges of the diagram; a larger diagram with the same bins gives the same
        # template in the bins they share
        big = (range(-0.2, 1.2 + 20 * step(edges[1]); length=95), range(dmod - 6.0, dmod + 5.0 + 30 * step(edges[2]); length=130))
        dist = ExponentialDiskExtinction(0.2, 1.0, 3.0)
        t = partial_cmd_smooth(m_ini, kmags, knots, dist, errs, 2, (1, 2), imf, comps; dmod, edges=big).weights
        @test L1(view(t, axes(new(dist))...), new(dist)) < 1e-6
    end

    @test_throws "bracket the support" new(Uniform(0.2, 3.0))
end
