import Distributions: Poisson
import StarFormationHistories as SFH
using StableRNGs: StableRNG
using StatsBase: sample
using Random: rand!, randperm
using Test

@testset "calculate_coeffs" begin
    types = (Float32, Float64) # Float types to test most functions with
    type_labels = ("Float32", "Float64") # String labels for the above float_types

    for i in eachindex(types, type_labels)
        tl = type_labels[i]
        @testset "$tl" begin
            T = types[i]
            rng = StableRNG(94823)
            mzr = SFH.PowerLawMZR(T(1), T(-1), T(6), (true, true))
            disp = SFH.GaussianDispersion(T(2//10))
            unique_logAge = collect(T, 10:-0.1:8)
            unique_MH = collect(T, -2.5:0.1:0.0)
            logAge = repeat(unique_logAge; inner=length(unique_MH))
            MH = repeat(unique_MH; outer=length(unique_logAge))
            # T_max = 12 # 12 Gyr
            # Now generate models, data
            hist_size = (100, 100)
            Mstars = rand(rng, T, length(unique_logAge)) .* 10^6
            models = [rand(rng, T, hist_size...) ./ 10^5 for i in 1:length(logAge)]

            x = SFH.calculate_coeffs(mzr, disp, Mstars, logAge, MH)
            # Test that \sum_k r_{j,k} \equiv R_j
            for ii in 1:length(Mstars)
                @test sum(@view(x[(begin+length(unique_MH)*(ii-1)):(length(unique_MH)*(ii))])) ≈ Mstars[ii]
            end
            @test x isa Vector{T}
            @test length(x) == length(logAge)
            # Make sure order of argument logAge is accounted for in returned coefficients
            # y = SFH.calculate_coeffs(mzr, disp, reverse(Mstars), reverse(logAge), reverse(MH), T_max)
            # @test x ≈ reverse(y)
            # rperm = sample(eachindex(unique_logAge), length(unique_logAge); replace=false)
            rperm = randperm(rng, length(unique_logAge))
            let unique_logAge = unique_logAge[rperm]
                logAge = repeat(unique_logAge; inner=length(unique_MH))
                MH = repeat(unique_MH; outer=length(unique_logAge))
                y = SFH.calculate_coeffs(mzr, disp, Mstars[rperm], logAge, MH)
                @test x ≈ y[sortperm(logAge; rev=true)]
            end
        end
    end
end

@testset "fg_mzr!" begin
    T = Float64
    rng = StableRNG(94823)
    mzr = SFH.PowerLawMZR(T(1), T(-2), T(6), (true, true))
    disp = SFH.GaussianDispersion(T(2//10))
    unique_logAge = collect(T, 10:-0.1:8)
    unique_MH = collect(T, -2.5:0.1:0.0)
    logAge = repeat(unique_logAge; inner=length(unique_MH))
    MH = repeat(unique_MH; outer=length(unique_logAge))
    # Now generate models, data
    hist_size = (100, 100)
    Mstars = rand(rng, T, length(unique_logAge)) .* 10^6
    models = [rand(rng, T, hist_size...) ./ 10^5 for i in 1:length(logAge)]
    x = SFH.calculate_coeffs(mzr, disp, Mstars, logAge, MH)
    data = rand.(rng, Poisson.(sum(x .* models))) # Poisson sampled data
    data2 = sum(x .* models) # Perfect data, no noise

    G = Vector{T}(undef, length(unique_logAge) + 3) # Gradient Vector
    C = similar(first(models)) # Composite model

    true_vals = vcat(Mstars, mzr.α, mzr.MH0, disp.σ)
    nlogL = 4917.491550052553
    # Gradient result from ForwardDiff.gradient
    fd_result = [-0.00014821148519279933, -0.00013852612731050684, -0.00012883365448787643, -0.00013224499379724353, -0.00013304264503999908, -0.00013141207154566565, -0.00013956083036823194, -0.00011818954688379261, -8.652358169076432e-5, -0.00010979288408696192, -0.00010299174097368632, -8.968648332825651e-5, -0.00011051661215933656, -9.664530693628639e-5, -0.00011448658353345554, -0.00012390106090859644, -0.00011506434863758139, -0.0001358595304639743, -9.920490692529175e-5, -8.412262247606323e-5, -0.0001095265086536905, -49.091387751034475, -180.99883434459917, -207.04960375880725]

    @test SFH.fg_mzr!(true, nothing, mzr, disp, true_vals, models, data, C, logAge, MH) ≈ nlogL
    @test SFH.fg_mzr!(true, G, mzr, disp, true_vals, models, data, C, logAge, MH) ≈ nlogL
    @test G ≈ fd_result
    # Test stacked models / data
    rand!(rng, G) # Fill G with random numbers so we aren't reusing last correct result
    @test SFH.fg_mzr!(true, G, mzr, disp, true_vals, SFH.stack_models(models),
                      vec(data), vec(C), logAge, MH) ≈ nlogL
    @test G ≈ fd_result
    # Make sure order of argument logAge is accounted for in returned gradient
    # This is now supported in calculate_coeffs so it should be supported by
    # fg_mzr! as well
    rperm = randperm(rng, length(unique_logAge))
    let unique_logAge = unique_logAge[rperm]
        logAge = repeat(unique_logAge; inner=length(unique_MH))
        MH = repeat(unique_MH; outer=length(unique_logAge))
        @test SFH.fg_mzr!(true, G, mzr, disp, true_vals, models,
                      data, C, logAge, MH) ≈ nlogL
        # y = SFH.calculate_coeffs(mzr, disp, Mstars[rperm], logAge, MH)
        # @test x ≈ y[sortperm(logAge; rev=true)]
    end

end
