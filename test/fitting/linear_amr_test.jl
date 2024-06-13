using Distributions: Poisson
import DynamicHMC
using StableRNGs: StableRNG
import StarFormationHistories as SFH

using LinearAlgebra: Diagonal
using Random: rand!
using Test


@testset "Linear AMR Fitting" begin
    T = Float64
    rng = StableRNG(94823)
    unique_logAge = 8.0:0.1:10.0
    unique_MH = -2.5:0.1:0.0
    logAge = repeat(unique_logAge; inner=length(unique_MH))
    MH = repeat(unique_MH; outer=length(unique_logAge))
    T_max = 12.0 # 12 Gyr
    α, β, σ = 0.05, (-0.05*T_max + -1.0), 0.2
    # Now generate models, data, and try to solve
    hist_size = (100,100)
    N_models = length(logAge)
    SFRs = rand(rng, T, length(unique_logAge))
    x = SFH.calculate_coeffs_mdf(SFRs, logAge, MH, T_max, α, β, σ)
    x0 = vcat(SFH.construct_x0_mdf(logAge, convert(T,13.7); normalize_value=sum(x)), α, β, σ)
    models = [rand(rng,T,hist_size...) .* 100 for i in 1:N_models]
    # Poisson sampled data
    data = rand.(rng, Poisson.(sum(x .* models)))
    # Perfect data, no noise
    data2 = sum(x .* models)
    @testset "fg_mdf!" begin
        G = Vector{T}(undef, length(unique_logAge) + 3)
        SFH.fg_mdf!(true, G, x0, models, data, sum(x .* models), logAge, MH, T_max)
        # fd_result = ForwardDiff.gradient(X -> SFH.fg_mdf!(nothing, nothing, X, models, data, sum(models .* x), logAge, MH, T_max), x0)
        fd_result = [-3249.4040663407613, -4221.866368087713, -3132.0879991898732, -4350.323064391376, -2276.013598903408, -2632.4753460764086, -3806.434617593392, -3817.3988839382796, -2651.0619744385044, -2614.150482703236, -2458.7187053222883, -2705.6299022719722, -2587.5247217918572, -3014.0486714702593, -1911.854433996178, -1529.7641988181008, -1686.0224945335553, -656.674057995984, -1409.6186615151864, -701.686674995488, 3475.1795021789867, -9330.458478087246, -679.1308624689251, -55663.12068696883] # from ForwardDiff.gradient
        @test G ≈ fd_result rtol=1e-5

        # Test with stacked models / data
        rand!(rng, G) # Fill G with random numbers so we aren't reusing last correct result
        SFH.fg_mdf!(true, G, x0, SFH.stack_models(models), vec(data), vec(sum(x .* models)), logAge, MH, T_max)
        @test G ≈ fd_result rtol=1e-5
    end
    @testset "fit_templates_mdf free σ" begin
        # Noisy data
        result = SFH.fit_templates_mdf(models, data, logAge, MH, T_max; x0=x0)
        @test result.mle.μ ≈ vcat(SFRs, α, β, σ) rtol=0.05
        @test result.map.μ ≈ vcat(SFRs, α, β, σ) rtol=0.1
        # Noise-free data
        result2 = SFH.fit_templates_mdf(models, data2, logAge, MH, T_max; x0=x0)
        @test result2.mle.μ ≈ vcat(SFRs, α, β, σ) rtol=1e-5
        @test result2.map.μ ≈ vcat(SFRs, α, β, σ) rtol=1e-2
        # Test with stacked models / data
        result = SFH.fit_templates_mdf(SFH.stack_models(models), vec(data2), logAge, MH, T_max; x0=x0)
        @test result.mle.μ ≈ vcat(SFRs, α, β, σ) rtol=1e-5
        @test result.map.μ ≈ vcat(SFRs, α, β, σ) rtol=1e-2
    end
    @testset "fit_templates_mdf fixed σ" begin
        # Noisy data
        result = SFH.fit_templates_mdf(models, data, logAge, MH, T_max, σ; x0=x0[begin:end-1])
        @test result.mle.μ ≈ vcat(SFRs, α, β) rtol=0.05
        @test result.map.μ ≈ vcat(SFRs, α, β) rtol=0.05
        # Noise-free data
        result = SFH.fit_templates_mdf(models, data2, logAge, MH, T_max, σ; x0=x0[begin:end-1])
        @test result.mle.μ ≈ vcat(SFRs, α, β) rtol=1e-5
        @test result.map.μ ≈ vcat(SFRs, α, β) rtol=1e-2
        # Test with stacked models / data
        result = SFH.fit_templates_mdf(SFH.stack_models(models), vec(data2), logAge, MH, T_max, σ; x0=x0[begin:end-1])
        @test result.mle.μ ≈ vcat(SFRs, α, β) rtol=1e-5
        @test result.map.μ ≈ vcat(SFRs, α, β) rtol=1e-2
    end

end

@testset "Linear AMR Sampling" begin
    T = Float64
    rng = StableRNG(77483)
    unique_logAge = 8.0:1.0:10.0
    unique_MH = -3.0:1.0:0.0
    logAge = repeat(unique_logAge; inner=length(unique_MH))
    MH = repeat(unique_MH; outer=length(unique_logAge))
    T_max = 12.0 # 12 Gyr
    α, β, σ = 0.05, (-0.05*T_max + -1.0), 0.2
    # Now generate models, data, and try to solve
    hist_size = (100,100)
    N_models = length(logAge)
    SFRs = rand(rng, T, length(unique_logAge))
    x = SFH.calculate_coeffs_mdf(SFRs, logAge, MH, T_max, α, β, σ)
    x0 = vcat(SFH.construct_x0_mdf(logAge, convert(T,13.7); normalize_value=sum(x)), α, β, σ)
    models = [rand(rng,T,hist_size...) .* 100 for i in 1:N_models]
    # Poisson sampled data
    data = rand.(rng, Poisson.(sum(x .* models)))
    # Perfect data, no noise
    data2 = sum(x .* models)
    @testset "hmc_sample_mdf" begin
        nsteps = 100
        # For speed we will use a preset configuration for the sampler
        # rather than using the automated warmup
        init_state = (q = [-2.12, 0.17, -0.71, -2.99, -1.6, -1.6],
                      ϵ = 0.0025,
                      κ = DynamicHMC.GaussianKineticEnergy(Diagonal([0.00546909682955714, 0.004491498846133859, 0.012414031924027869, 0.30428437892641774, 0.028106750680594324, 0.24961057141537304])))
        # Noisy data
        result = SFH.hmc_sample_mdf(models, data, logAge, MH, T_max, nsteps;
                                    # rng=rng, reporter=DynamicHMC.ProgressMeterReport(),
                                    rng=rng, reporter=DynamicHMC.NoProgressReport(),
                                    initialization = init_state,
                                    warmup_stages = ()) # No warmup
        @test size(result.posterior_matrix) == (length(SFRs)+3, nsteps)

        # Test flattened input
        result = SFH.hmc_sample_mdf(SFH.stack_models(models), vec(data), logAge, MH, T_max,
                                    nsteps; rng=rng, reporter=DynamicHMC.NoProgressReport(),
                                    initialization = init_state,
                                    warmup_stages = ()) # No warmup
        @test size(result.posterior_matrix) == (length(SFRs)+3, nsteps)
    end
end
