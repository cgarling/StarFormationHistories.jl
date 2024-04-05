import Distributions: Poisson
# import ForwardDiff
import StarFormationHistories as SFH
import StableRNGs: StableRNG
using Test

@testset "Logarithmic AMR, Free σ" begin
    T = Float64
    rng = StableRNG(94823)
    let unique_logAge=8.0:0.1:10.0, unique_MH=-2.5:0.1:0.0
        logAge = repeat(unique_logAge; inner=length(unique_MH))
        MH = repeat(unique_MH; outer=length(unique_logAge))
        α, β, σ = -0.0001, 0.02, 0.2
        # Now generate models, data, and try to solve
        hist_size = (100,100)
        N_models = length(logAge)
        let SFRs=rand(rng,T,length(unique_logAge)), MH_func=x-> SFH.MH_from_Z(x, 0.01524; Y_p=0.2485, γ=1.78), dMH_dZ_func=x->SFH.dMH_dZ(x,0.01524; Y_p=0.2485, γ=1.78), x=SFH.calculate_coeffs_logamr(SFRs, logAge, MH, α, β, σ; MH_func=MH_func), x0=vcat(SFH.construct_x0_mdf(logAge, convert(T,log10(13.7e9)); normalize_value=sum(x)), α, β, σ), models=[rand(rng,T,hist_size...) .* 100 for i in 1:N_models], data=rand.(rng, Poisson.(sum(x .* models)))
            G = Vector{T}(undef, length(unique_logAge) + 3)
            C = sum(x .* models) # Composite model
            # @btime SFH.fg_logamr!(true, $G, $x0, $models, $data, $C, $logAge, $MH, $MH_func, $dMH_dZ_func) # 5.2 ms 
            SFH.fg_logamr!(true, G, x0, models, data, C, logAge, MH, MH_func, dMH_dZ_func)
            # println(ForwardDiff.gradient(X -> SFH.fg_log_amr!(nothing, nothing, X, models, data, sum(models .* x), logAge, MH, MH_func, dMH_dZ_func), x0))
            fd_result = [-7881.60440592152, -10396.247482331686, -7817.8251615474, -10143.132995266222, -5132.489651065174, -6620.526559309652, -8908.339497847246, -9790.650862697417, -6723.021989013198, -6072.055618913932, -5484.847019764899, -6020.580335153412, -6140.713253710487, -7151.1894445527105, -4660.953571900758, -3432.5666327797203, -3951.2652741859206, -1294.149159827791, -3675.669038945043, -1748.2505946022573, 7709.374688671554, 1.3442805896094717e7, 1.4580865807653132e6, -149692.47742731686] # from ForwardDiff.gradient
            @test G ≈ fd_result rtol=1e-5
            result = SFH.fit_templates_logamr(models, data, logAge, MH; x0=x0, MH_func=SFH.MH_from_Z, MH_deriv_Z=SFH.dMH_dZ)
            # @test result.map.μ ≈ vcat(SFRs, α, β, σ) rtol=0.1
            # @test result.mle.μ ≈ vcat(SFRs, α, β, σ) rtol=0.05
        end
    end
end

# @testset "Linear AMR, Fixed σ" begin
#     T = Float64
#     rng = StableRNG(94823)
#     let unique_logAge=8.0:0.1:10.0, unique_MH=-2.5:0.1:0.0
#         logAge = repeat(unique_logAge; inner=length(unique_MH))
#         MH = repeat(unique_MH; outer=length(unique_logAge))
#         α, β, σ = -0.05, -1.0, 0.2
#         # Now generate models, data, and try to solve
#         hist_size = (100,100)
#         N_models = length(logAge)
#         let SFRs=rand(rng,T,length(unique_logAge)), x=SFH.calculate_coeffs_mdf(SFRs, logAge, MH, α, β, σ), x0=vcat(SFH.construct_x0_mdf(logAge, convert(T,log10(13.7e9)); normalize_value=sum(x)), α, β), models=[rand(rng,T,hist_size...) .* 100 for i in 1:N_models], data=rand.(rng, Poisson.(sum(x .* models)))
#             G = Vector{T}(undef, length(unique_logAge) + 2)
#             SFH.fg_mdf_fixedσ!(true, G, x0, models, data, sum(x .* models), logAge, MH, σ)
#             # fd_result = ForwardDiff.gradient(X -> SFH.fg_mdf!(nothing, nothing, X, models, data, sum(models .* x), logAge, MH), x0)
#             fd_result = [-3249.4040663407814, -4221.866368087739, -3132.0879991898933, -4350.323064391401, -2276.013598903432, -2632.4753460764423, -3806.4346175934143, -3817.398883938306, -2651.0619744385262, -2614.150482703261, -2458.718705322313, -2705.6299022720023, -2587.52472179188, -3014.048671470281, -1911.8544339961975, -1529.7641988181176, -1686.0224945335829, -656.6740579960083, -1409.6186615152087, -701.6866749955076, 3475.179502178964, 1180.8881284601534, -679.130862468919] # from ForwardDiff.gradient
#             @test G ≈ fd_result rtol=1e-5
#             result = SFH.fit_templates_mdf(models, data, logAge, MH, σ; x0=x0)
#             @test result.map.μ ≈ vcat(SFRs, α, β) rtol=0.05
#             @test result.mle.μ ≈ vcat(SFRs, α, β) rtol=0.05
#         end
#     end
# end
