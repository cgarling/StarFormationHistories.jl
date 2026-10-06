using DelimitedFiles: readdlm
using Distributions: Normal, Uniform, Poisson, quantile
using InitialMassFunctions: Kroupa2001
import LogDensityProblems
import MCMCChains
import Optim
using StableRNGs: StableRNG
using StarFormationHistories: SSPModel, fit_ssp, sample_ssp, ssp_components, ssp_hess, ssp_mass, ssp_logdensity, with_fraction,
    partial_cmd_smooth, binary_mass_fraction, exp_photerr, Martin2016_complete, NoBinaries, RandomBinaryPairs, BinaryMassRatio
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
    fit = fit_ssp(model)
    @test fit.map.dmod ≈ 25.0 atol=0.02
    @test fit.map.binary_fraction ≈ 0.3 atol=0.08 # The MAP of these data is 0.35, with a posterior width of 0.04
    @test fit.map.background_fraction ≈ 0.1 atol=0.01
    @test isnothing(fit.grid) # Neither age nor metallicity is free
    # Starting with a fraction at a bound of its prior, far from z = 0 in the unconstrained coordinates
    @test all(isfinite, StarFormationHistories.dθdz(model, [0.0, 800.0, -800.0]))
    fb = fit_ssp(model; x0=[25.0, 0.3, 0.0], restarts=0)
    @test all(isfinite, fb.θ) && isfinite(fb.logdensity)
    chain = sample_ssp(fit, 40; nwalkers=8, rng=StableRNG(3), use_progress_meter=false)
    @test chain isa MCMCChains.Chains
    @test size(chain) == (20, 6, 8) # free parameters, mass, background_stars, and lp for 8 walkers
    @test names(chain, :parameters) == [:dmod, :binary_fraction, :background_fraction, :mass, :background_stars]
    @test all(isfinite, chain[:lp])
    @test all(>(0), chain[:mass])
    # The mass at the MAP converts the expected number of SSP stars per solar mass formed (the blob of ssp_logdensity)
    @test ssp_mass(fit) ≈ sum(model.counts) * (1 - fit.map.background_fraction) / last(ssp_logdensity(model, fit.θ))
    @test length(chain.info.accept_ratio) == 8
    @test_throws "must be even" sample_ssp(fit, 40; nwalkers=7)
    # Free age and metallicity start from a grid at quantiles of their priors
    # Magnitudes that depend on age and metallicity, so the log density varies over the grid
    isofunc_shift(logAge, MH, Av) = (m_ini, [m .+ (logAge - 10) .+ (MH + 1) for m in mags])
    gridmodel = SSPModel(isofunc_shift, data, edges, errs, 2, (1, 2), imf, comps, [zero, zero]; kws..., logAge=Uniform(9.9, 10.1), MH=Uniform(-1.1, -0.9))
    gridfit = fit_ssp(gridmodel; ngrid=(4, 2), options=Optim.Options(iterations=200))
    @test gridfit.grid.logAge ≈ [9.925, 9.975, 10.025, 10.075]
    @test gridfit.grid.MH ≈ [-1.05, -0.95]
    # The threaded grid matches serial evaluation at each grid point, where profiling the binary and background fractions
    # can only improve on their prior medians
    x0 = [quantile(gridmodel.params[k], 0.5) for k in gridmodel.free]
    jmix = [findfirst(==(k), gridmodel.free) for k in (:binary_fraction, :background_fraction)]
    @test gridfit.grid.logdensity == [first(StarFormationHistories.grid_logdensity(gridmodel, [la; mh; x0[3:end]], jmix)) for la in gridfit.grid.logAge, mh in gridfit.grid.MH]
    @test all(gridfit.grid.logdensity .>= [first(ssp_logdensity(gridmodel, [la; mh; x0[3:end]])) for la in gridfit.grid.logAge, mh in gridfit.grid.MH])
    @test 1 <= length(gridfit.starts) <= 4
    @test gridfit.logdensity == maximum(s.logdensity for s in gridfit.starts)
    @test gridmodel.free == [:logAge, :MH, :dmod, :binary_fraction, :background_fraction]
    @test_throws "one value per free parameter" fit_ssp(gridmodel; x0=[25.0])
end
