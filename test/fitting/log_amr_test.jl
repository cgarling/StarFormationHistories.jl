using Distributions: Poisson
# import ForwardDiff
import StarFormationHistories as SFH
using StableRNGs: StableRNG
using Random: rand!
using Test

# Now try fixed_log_amr that uses an AMR that is logarithmic in [M/H]
# or, equivalently, linear in the metal mass fraction Z.
# First test calculate_αβ_logamr, which takes [M/H] at two points in time
# and calculates the α and β coefficients for linear Z AMR.
# The constraints will be [M/H] = -2.5 at lookback time of 13.7 Gyr
# and [M/H] = -1 at present-day.

@testset "calculate_αβ_logamr" begin
    low_constraint = (-2.5, 13.7)
    high_constraint = (-1.0, 0.0)
    T_max = last(low_constraint)
    α, β = SFH.calculate_αβ_logamr( low_constraint, high_constraint, T_max, SFH.Z_from_MH )
    @test α ≈ 0.00011345962863988529
    @test β ≈ 5.106276580989658e-5
    @test SFH.Z_from_MH( min(first(low_constraint),first(high_constraint)) ) ≈ β
    @test α ≈ ( SFH.Z_from_MH( first(high_constraint) ) - # dZ / dt
        SFH.Z_from_MH( first(low_constraint) ) ) /
        (last(low_constraint) - last(high_constraint))
    # Test that passing different max_age works
    @test all(SFH.calculate_αβ_logamr( low_constraint,
                                       high_constraint, 14.0) .≈ (0.00011345962863988529, 1.7024877217930916e-5))
    # Test that passing different function to calculate Z from MH works
    @test all(SFH.calculate_αβ_logamr( low_constraint,
                                       high_constraint, T_max,
                                       x -> SFH.Z_from_MH(x, 0.017; Y_p = 0.25) ) .≈
                                           (0.0001273609414391884, 5.736185144138171e-5) )
end

@testset "calculate_coeffs_logamr" begin
    α = 0.00011345544581771879
    β = 5.106276398722378e-5
    σ = 0.2
    # Set logAge and metallicity grid
    unique_logAge=8.0:0.1:10.0
    unique_MH=-2.5:0.1:0.0
    T_max = maximum(unique_logAge)
    logAge = repeat(unique_logAge; inner=length(unique_MH))
    MH = repeat(unique_MH; outer=length(unique_logAge))
    x0 = SFH.calculate_coeffs_logamr( ones(length(unique_logAge)), logAge, MH, T_max, α, β, σ)
    @test x0 isa Vector{Float64}
    @test length(x0) == length(logAge)
    @test sum(x0) ≈ length(unique_logAge)
    # Test alternate call signature
    x0 = SFH.calculate_coeffs_logamr( vcat(ones(length(unique_logAge)), α, β, σ), logAge, MH, T_max)
    @test x0 isa Vector{Float64}
    @test length(x0) == length(logAge)
    @test sum(x0) ≈ length(unique_logAge)
end

@testset "fixed_log_amr" begin
    rng = StableRNG(94823)
    # Set logAge and metallicity grid
    unique_logAge=8.0:0.1:10.0
    unique_MH=-2.5:0.1:0.0
    logAge = repeat(unique_logAge; inner=length(unique_MH))
    MH = repeat(unique_MH; outer=length(unique_logAge))
    # Metallicity evolution parameters
    low_constraint = (-2.5, 13.7)
    high_constraint = (-1.0, 0.0)
    T_max = last(low_constraint)
    α, β = SFH.calculate_αβ_logamr( low_constraint, high_constraint, T_max) # , SFH.Z_from_MH )
    σ = 0.2
    # Now generate models, data, and try to solve
    hist_size = (100,100)
    N_models = length(logAge)
    SFRs = rand(rng,length(unique_logAge))
    x = SFH.calculate_coeffs_logamr(SFRs, logAge, MH, T_max, α, β, σ)
    x0 = SFH.construct_x0_mdf(logAge, 13.7; normalize_value=sum(x))
    models = [rand(rng,hist_size...) .* 100 for i in 1:N_models]
    data = sum(x .* models) # Perfect data, no noise
    # Calculate relative weights for input to fixed_amr
    relweights = SFH.calculate_coeffs_logamr( ones(length(unique_logAge)), logAge, MH, T_max, α, β, σ)
    result = SFH.fixed_amr(models, data, logAge, MH, relweights; x0=x0)
    @test result.mle.μ ≈ SFRs rtol=1e-5
    # Now try fixed_log_amr that will internally calculate the relweights
    result2 = SFH.fixed_log_amr(models, data, logAge, MH, T_max, α, β, σ; x0=x0)
    @test result2.mle.μ ≈ SFRs rtol=1e-5
    # Try second call signature that takes low_constraint and high_constraint
    result3 = SFH.fixed_log_amr(models, data, logAge, MH, T_max, low_constraint, high_constraint, σ; x0=x0)
    @test result3.mle.μ ≈ SFRs rtol=1e-5
    # Try stacked models / data
    result4 = SFH.fixed_log_amr(SFH.stack_models(models), vec(data), logAge, MH, T_max, low_constraint, high_constraint, σ; x0=x0)
    @test result4.mle.μ ≈ SFRs rtol=1e-5
end


@testset "Logarithmic AMR Fitting" begin
    T = Float64
    rng = StableRNG(94823)
    # Metallicity evolution parameters
    α = 0.00011345544581771879
    β = 5.106276398722378e-5
    σ = 0.2
    # Set logAge and metallicity grid
    unique_logAge = 8.0:0.1:10.0
    unique_MH = -2.5:0.1:0.0
    T_max = exp10(maximum(unique_logAge)) / 1e9 # Earliest time to normalize β in Gyr
    logAge = repeat(unique_logAge; inner=length(unique_MH))
    MH = repeat(unique_MH; outer=length(unique_logAge))
    # Now generate models, data, and try to solve
    hist_size = (100,100)
    N_models = length(logAge)
    SFRs = rand(rng,T,length(unique_logAge))
    true_params = vcat(SFRs, α, β, σ)
    MH_func= x -> SFH.MH_from_Z(x, 0.01524; Y_p=0.2485, γ=1.78)
    dMH_dZ_func = x -> SFH.dMH_dZ(x, 0.01524; Y_p=0.2485, γ=1.78)
    x = SFH.calculate_coeffs_logamr(SFRs, logAge, MH, T_max, α, β, σ; MH_func=MH_func)
    x0 = vcat(SFH.construct_x0_mdf(logAge, convert(T,13.7); normalize_value=sum(x)), α, β, σ)
    models=[rand(rng,T,hist_size...) .* 100 for i in 1:N_models]
    # data = sum(x .* models) # Perfect data, no noise
    data = rand.(rng, Poisson.(sum(x .* models))) # Poisson sampled data
    data2 = sum(x .* models) # Perfect data, no noise

    @testset "fg_logamr!" begin
        # Test gradient function fg_logamr!
        G = Vector{T}(undef, length(unique_logAge) + 3) # Gradient Vector
        C = sum(x .* models) # Composite model
        # @btime SFH.composite!($C,$x,$models) # 1.4 ms
        # let models2 = SFH.stack_models(models), C2=vec(C); @btime SFH.composite!($C2,$x,$models2); end # 650 μs
        # @btime SFH.fg_logamr!(true, $G, $x0, $models, $data, $C, $logAge, $MH, $T_max, $MH_func, $dMH_dZ_func) # 4.2 ms
        # let models2 = SFH.stack_models(models), C2=vec(C), data2=vec(data)
        #     @btime SFH.fg_logamr!(true, $G, $x0, $models2, $data2, $C2, $logAge, $MH, $T_max, $MH_func, $dMH_dZ_func)
        # end # 1.6 ms
        SFH.fg_logamr!(true, G, x0, models, data, C, logAge, MH, T_max, MH_func, dMH_dZ_func)
        # println(ForwardDiff.gradient(X -> SFH.fg_logamr!(nothing, nothing, X, models, data, sum(models .* x), logAge, MH, T_max, MH_func, dMH_dZ_func), x0))
        # Gradient result from ForwardDiff.gradient
        fd_result = [-4692.208596423888, -5709.440777267078, -4602.609442611384, -5659.5098733236055, -3482.5058046214494, -3911.1622798110625, -5203.138639495843, -5350.369874122347, -4199.566624544723, -4168.199959414579, -3637.9308920671747, -4100.023107334878, -3947.721902465209, -4586.3902128262025, -3436.4973114511427, -2756.6349289625155, -3077.966938787887, -1918.1569818089802, -2667.8394811694516, -1988.1042593286616, 6854.680232369424, 1.394019545321506e6, -3.12416814882545e8, -84926.32704239819] 
        @test G ≈ fd_result rtol=1e-5
        # Test stacked models / data
        rand!(rng, G) # Fill G with random numbers so we aren't reusing last correct result
        SFH.fg_logamr!(true, G, x0, SFH.stack_models(models), vec(data), vec(C), logAge, MH, T_max, MH_func, dMH_dZ_func)
        @test G ≈ fd_result rtol=1e-5
    end

    @testset "fit_templates_logamr free σ" begin
        # Test fitting
        result = SFH.fit_templates_logamr(models, data, logAge, MH, T_max; x0=x0, MH_func=SFH.MH_from_Z, MH_deriv_Z=SFH.dMH_dZ)
        # Test that MLE and MAP results are within 3σ of the true answer for all parameters
        @test all(isapprox(result.mle.μ[i], true_params[i]; atol=3 * result.mle.σ[i]) for i in eachindex(true_params))
        @test all(isapprox(result.map.μ[i], true_params[i]; atol=3 * result.map.σ[i]) for i in eachindex(true_params))
        # @test result.map.μ ≈ vcat(SFRs, α, β, σ) rtol=0.1
        # @test result.mle.μ ≈ vcat(SFRs, α, β, σ) rtol=0.05

        # Now test under perfect data
        result2 = SFH.fit_templates_logamr(models, data2, logAge, MH, T_max; x0=x0, MH_func=SFH.MH_from_Z, MH_deriv_Z=SFH.dMH_dZ)
        # @btime SFH.fit_templates_logamr($models, $data2, $logAge, $MH; x0=$x0, MH_func=$SFH.MH_from_Z, MH_deriv_Z=$SFH.dMH_dZ, max_logAge=log10($max_age*1e9)) # 1.6 s
        @test result2.mle.μ ≈ true_params rtol=1e-5
        @test result2.map.μ ≈ true_params rtol=1e-2

        # Now test with stacked models / data
        result3 = SFH.fit_templates_logamr(SFH.stack_models(models), vec(data2), logAge, MH, T_max; x0=x0, MH_func=SFH.MH_from_Z, MH_deriv_Z=SFH.dMH_dZ)
        # let models2 = SFH.stack_models(models), Data2=vec(data2); @btime SFH.fit_templates_logamr($models2, $Data2, $logAge, $MH; x0=$x0, MH_func=$SFH.MH_from_Z, MH_deriv_Z=$SFH.dMH_dZ, max_logAge=log10($max_age*1e9)); end # 729 ms
        @test result3.mle.μ ≈ true_params rtol=1e-5
        @test result3.map.μ ≈ true_params rtol=1e-2
    end
    
    @testset "fit_templates_logamr fixed σ" begin
        x0_1 = x0[begin:end-1] # Remove σ off end
        true_params_1 = true_params[begin:end-1]
        result = SFH.fit_templates_logamr(models, data, logAge, MH, T_max, σ; x0=x0_1, MH_func=SFH.MH_from_Z, MH_deriv_Z=SFH.dMH_dZ)
        # Test that MLE and MAP results are within 3σ of the true answer for all parameters
        @test all(isapprox(result.mle.μ[i], true_params[i]; atol=3 * result.mle.σ[i]) for i in eachindex(true_params_1))
        @test all(isapprox(result.map.μ[i], true_params[i]; atol=3 * result.map.σ[i]) for i in eachindex(true_params_1))
        
        # Now test under perfect data
        data2 = sum(x .* models) # Perfect data, no noise
        result2 = SFH.fit_templates_logamr(models, data2, logAge, MH, T_max, σ; x0=x0_1, MH_func=SFH.MH_from_Z, MH_deriv_Z=SFH.dMH_dZ)
        @test result2.mle.μ ≈ true_params_1 rtol=1e-5
        @test result2.map.μ ≈ true_params_1 rtol=1e-2
        
        # Now test with stacked models / data
        result3 = SFH.fit_templates_logamr(SFH.stack_models(models), vec(data2), logAge, MH, T_max, σ; x0=x0_1, MH_func=SFH.MH_from_Z, MH_deriv_Z=SFH.dMH_dZ)
        @test result3.mle.μ ≈ true_params_1 rtol=1e-5
        @test result3.map.μ ≈ true_params_1 rtol=1e-2
    end
end
