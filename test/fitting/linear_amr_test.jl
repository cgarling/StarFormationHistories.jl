import Distributions: Poisson
import DynamicHMC
import StarFormationHistories as SFH
import StableRNGs: StableRNG

import LinearAlgebra: Diagonal
import Random: rand!
using Test


@testset "Linear AMR Fitting" begin
    T = Float64
    rng = StableRNG(94823)
    unique_logAge = 8.0:0.1:10.0
    unique_MH = -2.5:0.1:0.0
    logAge = repeat(unique_logAge; inner=length(unique_MH))
    MH = repeat(unique_MH; outer=length(unique_logAge))
    α, β, σ = -0.05, -1.0, 0.2
    # Now generate models, data, and try to solve
    hist_size = (100,100)
    N_models = length(logAge)
    SFRs = rand(rng, T, length(unique_logAge))
    x = SFH.calculate_coeffs_mdf(SFRs, logAge, MH, α, β, σ)
    x0 = vcat(SFH.construct_x0_mdf(logAge, convert(T,log10(13.7e9)); normalize_value=sum(x)), α, β, σ)
    models = [rand(rng,T,hist_size...) .* 100 for i in 1:N_models]
    # Poisson sampled data
    data = rand.(rng, Poisson.(sum(x .* models)))
    # Perfect data, no noise
    data2 = sum(x .* models)
    @testset "fg_mdf!" begin
        G = Vector{T}(undef, length(unique_logAge) + 3)
        SFH.fg_mdf!(true, G, x0, models, data, sum(x .* models), logAge, MH)
        # fd_result = ForwardDiff.gradient(X -> SFH.fg_mdf!(nothing, nothing, X, models, data, sum(models .* x), logAge, MH), x0)
        fd_result = [-3249.4040663407814, -4221.866368087739, -3132.0879991898933, -4350.323064391401, -2276.013598903432, -2632.4753460764423, -3806.4346175934143, -3817.398883938306, -2651.0619744385262, -2614.150482703261, -2458.718705322313, -2705.6299022720023, -2587.52472179188, -3014.048671470281, -1911.8544339961975, -1529.7641988181176, -1686.0224945335829, -656.6740579960083, -1409.6186615152087, -701.6866749955076, 3475.179502178964, 1180.8881284601534, -679.130862468919, -55663.12068696887] # from ForwardDiff.gradient
        @test G ≈ fd_result rtol=1e-5

        # Test with stacked models / data
        rand!(rng, G) # Fill G with random numbers so we aren't reusing last correct result
        SFH.fg_mdf!(true, G, x0, SFH.stack_models(models), vec(data), vec(sum(x .* models)), logAge, MH)
        @test G ≈ fd_result rtol=1e-5
    end
    @testset "fit_templates_mdf free σ" begin
        # Noisy data
        result = SFH.fit_templates_mdf(models, data, logAge, MH; x0=x0)
        @test result.mle.μ ≈ vcat(SFRs, α, β, σ) rtol=0.05
        @test result.map.μ ≈ vcat(SFRs, α, β, σ) rtol=0.1
        # Noise-free data
        result2 = SFH.fit_templates_mdf(models, data2, logAge, MH; x0=x0)
        @test result2.mle.μ ≈ vcat(SFRs, α, β, σ) rtol=1e-5
        @test result2.map.μ ≈ vcat(SFRs, α, β, σ) rtol=1e-2
        # Test with stacked models / data
        result = SFH.fit_templates_mdf(SFH.stack_models(models), vec(data2), logAge, MH; x0=x0)
        @test result.mle.μ ≈ vcat(SFRs, α, β, σ) rtol=1e-5
        @test result.map.μ ≈ vcat(SFRs, α, β, σ) rtol=1e-2
    end
    @testset "fit_templates_mdf fixed σ" begin
        # Noisy data
        result = SFH.fit_templates_mdf(models, data, logAge, MH, σ; x0=x0[begin:end-1])
        @test result.mle.μ ≈ vcat(SFRs, α, β) rtol=0.05
        @test result.map.μ ≈ vcat(SFRs, α, β) rtol=0.05
        # Noise-free data
        result = SFH.fit_templates_mdf(models, data2, logAge, MH, σ; x0=x0[begin:end-1])
        @test result.mle.μ ≈ vcat(SFRs, α, β) rtol=1e-5
        @test result.map.μ ≈ vcat(SFRs, α, β) rtol=1e-2
        # Test with stacked models / data
        result = SFH.fit_templates_mdf(SFH.stack_models(models), vec(data2), logAge, MH, σ; x0=x0[begin:end-1])
        @test result.mle.μ ≈ vcat(SFRs, α, β) rtol=1e-5
        @test result.map.μ ≈ vcat(SFRs, α, β) rtol=1e-2
    end

end

@testset "Linear AMR Sampling" begin
    T = Float64
    rng = StableRNG(94823)
    unique_logAge = 8.0:1.0:10.0
    unique_MH = -3.0:1.0:0.0
    logAge = repeat(unique_logAge; inner=length(unique_MH))
    MH = repeat(unique_MH; outer=length(unique_logAge))
    α, β, σ = -0.05, -1.0, 0.2
    # Now generate models, data, and try to solve
    hist_size = (100,100)
    N_models = length(logAge)
    SFRs = rand(rng, T, length(unique_logAge))
    x = SFH.calculate_coeffs_mdf(SFRs, logAge, MH, α, β, σ)
    x0 = vcat(SFH.construct_x0_mdf(logAge, convert(T,log10(13.7e9)); normalize_value=sum(x)), α, β, σ)
    models = [rand(rng,T,hist_size...) .* 100 for i in 1:N_models]
    # Poisson sampled data
    data = rand.(rng, Poisson.(sum(x .* models)))
    # Perfect data, no noise
    data2 = sum(x .* models)
    @testset "hmc_sample_mdf" begin
        nsteps = 100
        # For speed we will use a preset configuration for the sampler
        # rather than using the automated warmup
        init_state = (q = [-0.7503439274317939, -0.027508163311039963, -0.7898194602933509, -0.08534770834583656, -0.6459701952353248, -1.6777167463320444],
                      ϵ = 0.004946593249939694,
                      κ = DynamicHMC.GaussianKineticEnergy(Diagonal([0.007342053593861451, 0.004550720897205579, 0.00794496306124024, 0.013278393906055555, 0.1279775281620597, 0.16875139300493644])))
        # Noisy data
        result = SFH.hmc_sample_mdf(models, data, logAge, MH, nsteps;
                                    rng=rng, reporter=DynamicHMC.NoProgressReport(),
                                    initialization = init_state,
                                    warmup_stages = ()) # No warmup
        @test size(result.posterior_matrix) == (length(SFRs)+3, nsteps)

        # Test flattened input
        result = SFH.hmc_sample_mdf(SFH.stack_models(models), vec(data), logAge, MH, nsteps;
                                    rng=rng, reporter=DynamicHMC.NoProgressReport(),
                                    initialization = init_state,
                                    warmup_stages = ()) # No warmup
        @test size(result.posterior_matrix) == (length(SFRs)+3, nsteps)
    end
end
