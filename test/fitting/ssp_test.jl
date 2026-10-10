using DelimitedFiles: readdlm
using Distributions: Normal, Uniform, Poisson, quantile, MixtureModel, MvNormal, logpdf
using InitialMassFunctions: Kroupa2001
import LogDensityProblems
import StarFormationHistories as SFH
import Optim
using StableRNGs: StableRNG
using StarFormationHistories: SSPModel, fit_ssp, sample_ssp, ssp_components, ssp_hess, ssp_mass, ssp_logdensity, with_fraction,
    partial_cmd_smooth, binary_mass_fraction, exp_photerr, Martin2016_complete, NoBinaries, RandomBinaryPairs, BinaryMassRatio, dθdz, profiled_logdensity
using Test

isochrone, _ = readdlm(joinpath(@__DIR__, "../../data/isochrone.txt"), ' ', Float64, '\n'; header=true)
m_ini, mags = isochrone[:, 1], [isochrone[:, 2], isochrone[:, 3]]
# Age, metallicity, and extinction are fixed in these tests, so the isochrone does not depend on them
isofunc(logAge, MH, Av) = (m_ini, mags)
edges = (range(-0.2, 1.2; length=60), range(19.0, 30.0; length=90))
nbins = length.(edges) .- 1
errs = [m -> min(exp_photerr(m, 1.03, 15.0, 36.0, 0.02), 0.4), m -> min(exp_photerr(m, 1.03, 15.0, 35.0, 0.02), 0.4)]
comps = [m -> Martin2016_complete(m, 1.0, 28.5, 0.7), m -> Martin2016_complete(m, 1.0, 27.5, 0.7)]
imf = Kroupa2001(0.08, 100.0)

@testset "ssp_components" begin
    for bm in (RandomBinaryPairs(0.3), BinaryMassRatio(0.3))
        single, binary = ssp_components(m_ini, mags, errs, 2, (1, 2), imf, comps; binary_model=bm, dmod=25.0, edges)
        full = partial_cmd_smooth(m_ini, mags, errs, 2, (1, 2), imf, comps; binary_model=bm, dmod=25.0, edges).weights
        φ = binary_mass_fraction(bm, imf)
        @test (1 - φ) .* single.weights .+ φ .* binary.weights ≈ full
    end
    single, binary = ssp_components(m_ini, mags, errs, 2, (1, 2), imf, comps; dmod=25.0, edges)
    @test isnothing(binary)
    @test single.weights ≈ partial_cmd_smooth(m_ini, mags, errs, 2, (1, 2), imf, comps; dmod=25.0, edges).weights
    @test with_fraction(BinaryMassRatio(0.3, Uniform(0.2, 1.0)), 0.5) == BinaryMassRatio(0.5, Uniform(0.2, 1.0))
    @test with_fraction(RandomBinaryPairs(0.3), 0.5) == RandomBinaryPairs(0.5)
    @test_throws "cannot have a nonzero binary fraction" with_fraction(NoBinaries(), 0.5)
end

@testset "SSPModel" begin
    data = ones(nbins)
    kws = (logAge=10.0, MH=-1.0, dmod=Normal(25.0, 0.3))
    model = SSPModel(isofunc, data, edges, errs, 2, (1, 2), imf, comps, [zero, zero]; kws...)
    @test model.free == [:dmod, :background_fraction]
    @test LogDensityProblems.dimension(model) == 2
    @test sum(model.background[model.keep]) ≈ 1
    # Outside the support of the background_fraction prior
    @test first(ssp_logdensity(model, [25.0, 1.5])) == -Inf
    @test isfinite(LogDensityProblems.logdensity(model, [25.0, 0.1]))
    # The expected Hess diagram sums to the number of observed stars in the unmasked bins, and covers masked bins
    mask = falses(nbins); mask[1:10, :] .= true
    masked = SSPModel(isofunc, data, edges, errs, 2, (1, 2), imf, comps, [zero, zero]; kws..., mask)
    h = ssp_hess(masked, (logAge=10.0, MH=-1.0, dmod=25.0, Av=0.0, binary_fraction=0.0, background_fraction=0.1))
    @test size(h) == nbins
    @test sum(h[.!mask]) ≈ sum(data[.!mask])
    @test all(>(0), h[mask])
    # A background shape is normalized over the unmasked bins, with a uniform floor
    bg = zeros(nbins); bg[30, 40] = 1
    model = SSPModel(isofunc, data, edges, errs, 2, (1, 2), imf, comps, [zero, zero]; kws..., background=bg, background_floor=0.1)
    @test model.background[(40 - 1) * nbins[1] + 30] ≈ 0.9 + 0.1 / prod(nbins)
    @test_throws "requires a binary model" SSPModel(isofunc, data, edges, errs, 2, (1, 2), imf, comps, [zero, zero]; kws..., binary_fraction=Uniform(0, 1))
    @test_throws "must be a number (fixed) or a univariate distribution" SSPModel(isofunc, data, edges, errs, 2, (1, 2), imf, comps, [zero, zero]; kws..., Av="0.1")
    @test_throws "excludes every bin" SSPModel(isofunc, data, edges, errs, 2, (1, 2), imf, comps, [zero, zero]; kws..., mask=trues(nbins))
end

@testset "fit_ssp and sample_ssp" begin
    kws = (logAge=10.0, MH=-1.0, dmod=Normal(25.0, 0.3), binary_model=BinaryMassRatio(0.3), binary_fraction=Uniform(0, 1))
    truth = (logAge=10.0, MH=-1.0, dmod=25.0, Av=0, binary_fraction=0.3, background_fraction=0.1)
    # Mock observations of 30,000 stars, drawn from the expected Hess diagram of a model with uniform data
    expected = ssp_hess(SSPModel(isofunc, fill(30_000 / prod(nbins), nbins), edges, errs, 2, (1, 2), imf, comps, [zero, zero]; kws...), truth)
    data = rand.(StableRNG(2), Poisson.(expected))
    model = SSPModel(isofunc, data, edges, errs, 2, (1, 2), imf, comps, [zero, zero]; kws...)
    fit = fit_ssp(model; ncandidates=50, rng=StableRNG(1))
    @test fit.map.dmod ≈ 25.0 atol=0.02
    @test fit.map.binary_fraction ≈ 0.3 atol=0.08 # The MAP of these data is 0.35, with a posterior width of 0.04
    @test fit.map.background_fraction ≈ 0.1 atol=0.01
    # The free distance modulus, whose normal prior is unbounded, is searched between the 0.001 and 0.999 quantiles of its
    # prior, beyond the 0.02 and 0.98 quantiles
    @test all(θ -> 25 - 0.928 < θ[1] < 25 + 0.928, fit.candidates.θ) && any(θ -> abs(θ[1] - 25) > 0.617, fit.candidates.θ)
    # Starting with a fraction at a bound of its prior, far from z = 0 in the unconstrained coordinates
    @test all(isfinite, dθdz(model, [0.0, 800.0, -800.0]))
    fb = fit_ssp(model; x0=[25.0, 0.3, 0.0], restarts=0)
    @test all(isfinite, fb.θ) && isfinite(fb.logdensity)
    @test isnothing(fb.candidates)
    # Settings far too small for reliable estimates, so the precision warning is expected
    s = @test_logs (:warn, r"effective samples") match_mode=:any sample_ssp(fit; pilot_steps=40, pilot_walkers=8, is_rounds=3,
                                                                         is_samples=200, rng=StableRNG(3))
    @test keys(s.samples) == (:dmod, :binary_fraction, :background_fraction, :mass, :background_stars, :lp, :mode)
    @test all(length(c) == 600 for c in s.samples)
    @test all(isfinite, s.samples.lp)
    @test all(>(0), s.samples.mass)
    @test length(s.modes) == 1 && all(==(1), s.samples.mode) && s.modes[1].weight ≈ 1
    @test isfinite(s.logZ) && s.logZ ≈ s.modes[1].logZ
    @test any(contains("effective samples"), s.warnings)
    s2 = @test_logs (:warn, r"effective samples") match_mode=:any sample_ssp(fit; pilot_steps=40, pilot_walkers=8, is_rounds=3,
                                                                          is_samples=200, rng=StableRNG(3))
    @test s2.samples == s.samples # Reproducible for a given rng and number of threads
    @test sum(s.samples.dmod) / 600 ≈ 25.0 atol=0.02
    # The mass at the MAP converts the expected number of SSP stars per solar mass formed (the blob of ssp_logdensity)
    @test ssp_mass(fit) ≈ sum(model.counts) * (1 - fit.map.background_fraction) / last(ssp_logdensity(model, fit.θ))
    @test_throws "must be even" sample_ssp(fit; pilot_walkers=7)
    @test_throws "must be at least 3" sample_ssp(fit; is_rounds=2)
    # Free age, metallicity, and distance modulus are searched together
    # Magnitudes that depend on age and metallicity, so the log density varies over the candidates
    isofunc_shift(logAge, MH, Av) = (m_ini, [m .+ (logAge - 10) .+ (MH + 1) for m in mags])
    gridmodel = SSPModel(isofunc_shift, data, edges, errs, 2, (1, 2), imf, comps, [zero, zero]; kws..., logAge=Uniform(9.9, 10.1), MH=Uniform(-1.1, -0.9))
    gridfit = fit_ssp(gridmodel; ncandidates=40, options=Optim.Options(iterations=200), rng=StableRNG(4))
    @test all(θ -> 9.904 <= θ[1] <= 10.096 && -1.096 <= θ[2] <= -0.904, gridfit.candidates.θ)
    # The threaded candidates match serial evaluation, where profiling the binary and background fractions can only
    # improve on their prior medians
    x0 = [quantile(gridmodel.params[k], 0.5) for k in gridmodel.free]
    jmix = [findfirst(==(k), gridmodel.free) for k in (:binary_fraction, :background_fraction)]
    @test gridfit.candidates.logdensity == [first(profiled_logdensity(gridmodel, [θ[1:3]; x0[4:end]], jmix)) for θ in gridfit.candidates.θ]
    @test all(gridfit.candidates.logdensity .>= [first(ssp_logdensity(gridmodel, [θ[1:3]; x0[4:end]])) for θ in gridfit.candidates.θ])
    @test 1 <= length(gridfit.starts) <= 16
    @test gridfit.logdensity == maximum(s.logdensity for s in gridfit.starts)
    @test gridmodel.free == [:logAge, :MH, :dmod, :binary_fraction, :background_fraction]
    @test_throws "one value per free parameter" fit_ssp(gridmodel; x0=[25.0])
end

@testset "Starting points" begin
    t = SFH.r2_points(500, 2, StableRNG(1))
    @test size(t) == (2, 500) && all(x -> 0 <= x < 1, t)
    @test t == SFH.r2_points(500, 2, StableRNG(1))
    # Each coordinate is spread evenly: 50 points in each tenth of the unit interval, to within 5 (independent uniform
    # points would vary by about 7)
    @test all(j -> all(k -> abs(count(x -> (k - 1) / 10 <= x < k / 10, view(t, j, :)) - 50) <= 5, 1:10), 1:2)
    # Two separated peaks give one starting point each, best first
    lds = [100 * max(-sum(abs2, x .- 0.25), 0.01 - sum(abs2, x .- 0.75)) for x in eachcol(t)]
    idx = SFH.select_starts(t, lds, 16)
    @test length(idx) == 2 && sum(abs2, t[:, idx[1]] .- 0.75) < 0.01 && sum(abs2, t[:, idx[2]] .- 0.25) < 0.01
    @test (@test_logs (:warn, r"Increase `nstarts`") SFH.select_starts(t, lds, 1)) == idx[1:1]
end

@testset "Adaptive importance sampling" begin
    # A mixture of two Gaussians with evidence e^10 and mode weights 0.3 and 0.7, started from offset means and
    # covariances that are too narrow
    μ1, μ2 = [0.0, 0.0], [5.0, 3.0]
    Σ1, Σ2 = [1.0 0.6; 0.6 1.0], [0.5 -0.2; -0.2 0.8]
    mix = MixtureModel([MvNormal(μ1, Σ1), MvNormal(μ2, Σ2)], [0.3, 0.7])
    rounds = SFH.adaptive_is(θ -> (logpdf(mix, θ) + 10.0, 1.0), [μ1 .+ 0.5, μ2 .- 0.4], [0.3 .* Σ1, 0.3 .* Σ2]; rounds=3, n=2000,
                             rng=StableRNG(1))
    a = SFH.amis_combine(rounds, 5)
    W = SFH.softmax(a.lw)
    @test SFH.logsumexp(a.lw) - log(length(a.lw)) ≈ 10 atol=0.05
    @test [sum(W[a.mode .== k]) for k in 1:2] ≈ [0.3, 0.7] atol=0.02
    d = SFH.round_diagnostics(last(rounds), 5, 1.5)
    @test d.khat < 0.7 && maximum(d.shift) < 0.5
    # A single round from proposals much too narrow has not converged
    early = SFH.adaptive_is(θ -> (logpdf(mix, θ) + 10.0, 1.0), [μ1, μ2], [0.05 .* Σ1, 0.05 .* Σ2]; rounds=1, n=2000, rng=StableRNG(1))
    d1 = SFH.round_diagnostics(last(early), 5, 1.5)
    @test maximum(d1.shift) > 0.5 || d1.khat > 0.7
    # Pareto k̂ recovers the shape of a Pareto tail of weights w = U^-k
    rng = StableRNG(2)
    @test sort([SFH.pareto_khat(-0.3 .* log.(rand(rng, 4000))) for _ in 1:30])[15] ≈ 0.3 atol=0.05
    @test sort([SFH.pareto_khat(-0.9 .* log.(rand(rng, 4000))) for _ in 1:30])[15] ≈ 0.9 atol=0.05
    @test SFH.pareto_khat(zeros(10)) == Inf # Too few weights for a fit
    @test SFH.pareto_khat([fill(-1000.0, 1999); 0.0]) == Inf # One weight dominates the rest, which underflow to 0
    # Systematic resampling draws each weight's share exactly, up to one sample
    idx = SFH.systematic_resample([0.1, 0.6, 0.3], 1000, StableRNG(3))
    @test [count(==(i), idx) for i in 1:3] == [100, 600, 300]
end
