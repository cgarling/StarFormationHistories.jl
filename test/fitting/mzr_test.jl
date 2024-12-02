import StarFormationHistories as SFH
using StableRNGs: StableRNG
using StatsBase: sample
using Random: rand!
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
            T_max = 12 # 12 Gyr
            # Now generate models, data
            hist_size = (100, 100)
            Mstars = rand(rng, T, length(unique_logAge)) .* 10^6
            models = [rand(rng, T, hist_size...) ./ 10^5 for i in 1:length(logAge)]

            x = SFH.calculate_coeffs(mzr, disp, Mstars, logAge, MH, T_max)
            # Test that \sum_k r_{j,k} \equiv R_j
            for ii in 1:length(Mstars)
                @test sum(@view(x[(begin+length(unique_MH)*(ii-1)):(length(unique_MH)*(ii))])) ≈ Mstars[ii]
            end
            @test x isa Vector{T}
            @test length(x) == length(logAge)
            # Make sure order of argument logAge is accounted for in returned coefficients
            # y = SFH.calculate_coeffs(mzr, disp, reverse(Mstars), reverse(logAge), reverse(MH), T_max)
            # @test x ≈ reverse(y)
            rperm = sample(eachindex(unique_logAge), length(unique_logAge); replace=false)
            let unique_logAge = unique_logAge[rperm]
                logAge = repeat(unique_logAge; inner=length(unique_MH))
                MH = repeat(unique_MH; outer=length(unique_logAge))
                y = SFH.calculate_coeffs(mzr, disp, Mstars[rperm], logAge, MH, T_max)
                @test x ≈ y[sortperm(logAge; rev=true)]
            end
        end
    end
end
