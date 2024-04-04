import Distributions: Poisson
import StarFormationHistories as SFH
import StableRNGs: StableRNG
using Test

@testset "Linear AMR, Free σ" begin
    T = Float64
    rng = StableRNG(94823)
    let unique_logAge=8.0:0.1:10.0, unique_MH=-2.5:0.1:0.0
        logAge = repeat(unique_logAge; inner=length(unique_MH))
        MH = repeat(unique_MH; outer=length(unique_logAge))
        α, β, σ = -0.05, -1.0, 0.2
        # Now generate models, data, and try to solve
        hist_size = (100,100)
        N_models = length(logAge)
        let SFRs=rand(rng,T,length(unique_logAge)), x=SFH.calculate_coeffs_mdf(SFRs, logAge, MH, α, β, σ), x0=vcat(SFH.construct_x0_mdf(logAge, convert(T,log10(13.7e9)); normalize_value=sum(x)), α, β, σ), models=[rand(rng,T,hist_size...) .* 100 for i in 1:N_models], data=rand.(rng, Poisson.(sum(x .* models)))
            G = Vector{T}(undef, length(unique_logAge) + 3)
            SFH.fg_mdf!(true, G, x0, models, data, sum(x .* models), logAge, MH)
            # fd_result = ForwardDiff.gradient(X -> SFH.fg_mdf!(nothing, nothing, X, models, data, sum(models .* x), logAge, MH), x0)
            fd_result = [-3249.4040663407814, -4221.866368087739, -3132.0879991898933, -4350.323064391401, -2276.013598903432, -2632.4753460764423, -3806.4346175934143, -3817.398883938306, -2651.0619744385262, -2614.150482703261, -2458.718705322313, -2705.6299022720023, -2587.52472179188, -3014.048671470281, -1911.8544339961975, -1529.7641988181176, -1686.0224945335829, -656.6740579960083, -1409.6186615152087, -701.6866749955076, 3475.179502178964, 1180.8881284601534, -679.130862468919, -55663.12068696887] # from ForwardDiff.gradient
            @test G ≈ fd_result rtol=1e-5
            result = SFH.fit_templates_mdf(models, data, logAge, MH; x0=x0)
            @test result.map.μ ≈ vcat(SFRs, α, β, σ) rtol=0.1
            @test result.mle.μ ≈ vcat(SFRs, α, β, σ) rtol=0.05
        end
    end
end

@testset "Linear AMR, Fixed σ" begin
    T = Float64
    rng = StableRNG(94823)
    let unique_logAge=8.0:0.1:10.0, unique_MH=-2.5:0.1:0.0
        logAge = repeat(unique_logAge; inner=length(unique_MH))
        MH = repeat(unique_MH; outer=length(unique_logAge))
        α, β, σ = -0.05, -1.0, 0.2
        # Now generate models, data, and try to solve
        hist_size = (100,100)
        N_models = length(logAge)
        let SFRs=rand(rng,T,length(unique_logAge)), x=SFH.calculate_coeffs_mdf(SFRs, logAge, MH, α, β, σ), x0=vcat(SFH.construct_x0_mdf(logAge, convert(T,log10(13.7e9)); normalize_value=sum(x)), α, β), models=[rand(rng,T,hist_size...) .* 100 for i in 1:N_models], data=rand.(rng, Poisson.(sum(x .* models)))
            G = Vector{T}(undef, length(unique_logAge) + 2)
            SFH.fg_mdf_fixedσ!(true, G, x0, models, data, sum(x .* models), logAge, MH, σ)
            # fd_result = ForwardDiff.gradient(X -> SFH.fg_mdf!(nothing, nothing, X, models, data, sum(models .* x), logAge, MH), x0)
            fd_result = [-3249.4040663407814, -4221.866368087739, -3132.0879991898933, -4350.323064391401, -2276.013598903432, -2632.4753460764423, -3806.4346175934143, -3817.398883938306, -2651.0619744385262, -2614.150482703261, -2458.718705322313, -2705.6299022720023, -2587.52472179188, -3014.048671470281, -1911.8544339961975, -1529.7641988181176, -1686.0224945335829, -656.6740579960083, -1409.6186615152087, -701.6866749955076, 3475.179502178964, 1180.8881284601534, -679.130862468919] # from ForwardDiff.gradient
            @test G ≈ fd_result rtol=1e-5
            result = SFH.fit_templates_mdf(models, data, logAge, MH, σ; x0=x0)
            @test result.map.μ ≈ vcat(SFRs, α, β) rtol=0.05
            @test result.mle.μ ≈ vcat(SFRs, α, β) rtol=0.05
        end
    end
end
