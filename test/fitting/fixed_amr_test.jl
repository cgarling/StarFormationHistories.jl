# With the new fixed/free parameter API introduced in 1.0 we no longer
# really need dedicated fixed_amr methods, but keeping around for now

import StarFormationHistories as SFH
import Random
import StableRNGs: StableRNG
using StatsBase: median
using Test: @test, @testset, @test_logs

const seedval = 58392 # Seed to use when instantiating new StableRNG objects

unique_logAge = 8.0:0.1:10.0
unique_MH = -2.5:0.1:0.0

logAge = repeat(unique_logAge; inner=length(unique_MH))
MH = repeat(unique_MH; outer=length(unique_logAge))
T_max = 12.0 # 12.0 Gyr
α, β, σ = 0.05, (-1.0 + -0.05*T_max), 0.2

MH_model = SFH.LinearAMR(α, β)
disp_model = SFH.GaussianDispersion(σ)

hist_size = (100, 100)
T = Float64

relweights = SFH.calculate_coeffs(MH_model, disp_model, ones(length(unique_logAge)),
                                  logAge, MH)

@testset "calculate_coeffs" begin
    for (i, la) in enumerate(unique_logAge)
        @test sum(relweights[logAge .== la]) ≈ 1
    end
end

@testset "construct_x0_mdf" begin
    # The input logAge does not need to be in any particular order
    # in order to use this method. Test this by shuffling `logAge`.
    let logAge = Random.shuffle(logAge)
        x0 = SFH.construct_x0_mdf(logAge, 13.7)
        @test length(x0) == length(unique_logAge)
        idxs = sortperm(unique(logAge))
        sorted_ul = vcat(unique(logAge)[idxs], log10(13.7e9))
        dt = diff(exp10.(sorted_ul))
        sfr = [ begin
                   idx = findfirst( ==(sorted_ul[i]), unique(logAge) )
                   x0[i] / dt[idx]
                end for i in eachindex(unique(logAge)) ]
        @test all(sfr .≈ first(sfr)) # Test the SFR in each time bin is approximately equal
    end
    # Test normalize_value
    @test sum(SFH.construct_x0_mdf(logAge, 13.7)) ≈ 1
    @test sum(SFH.construct_x0_mdf(logAge, 13.7; normalize_value=1e5)) ≈ 1e5  
end

# Now generate models, data, and try to solve
@testset "fixed_amr" begin
    rng = StableRNG(seedval)
    SFRs = rand(rng,T,length(unique_logAge))
    x = SFH.calculate_coeffs(MH_model, disp_model, SFRs, logAge, MH)
    x0 = SFH.construct_x0_mdf(logAge, convert(T, 13.7); normalize_value=1)
    models = [rand(rng,T,hist_size...) .* 100 for i in 1:length(logAge)]
    data = sum(x .* models)
    # Calculate relative weights for input to fixed_amr
    relweights = SFH.calculate_coeffs(MH_model, disp_model,
                                      ones(length(unique_logAge)), logAge, MH)
    result = SFH.fixed_amr(models, data, logAge, MH, relweights; x0=x0)
    @test result.mle.μ ≈ SFRs rtol=1e-5
    
    # Test that improperly normalized relweights results in warning
    # Test currently fails on julia 1.7, I think due to a difference
    # in the way that the warnings are logged so, remove
    if VERSION >= v"1.8"
        @test_logs (:warn,) SFH.fixed_amr(models, data, logAge, MH, 2 .* relweights; x0=x0)
    end
    
    # Test how removing low-weight models from fixed_amr might impact fit
    relweightsmin = 0.05 # Include only models whose relative weights are > 10% of the maximum in the logAge bin
    keep_idx = Int[]
    for (i, la) in enumerate(unique_logAge)
        good = findall(logAge .== la) # Select models with correct logAge
        tmp_relweights = relweights[good]
        max_relweight = maximum(tmp_relweights) # Find maximum relative weight for this set of models
        high_weights = findall(tmp_relweights .>= (relweightsmin * max_relweight))
        keep_idx = vcat(keep_idx, good[high_weights])
    end

    # This takes ~0.1s compared to ~1.5s for the full result = SFH.fixed_amr ... above
    # This should throw an error because the relweights *were* properly normalized
    # to sum to 1 for each unique(logAge), but since we have removed some entries,
    # they are no longer properly normalized; see above note on Julia < 1.7
    if VERSION >= v"1.8"
        result3 = @test_logs (:warn,) SFH.fixed_amr(models[keep_idx], data, logAge[keep_idx], MH[keep_idx], relweights[keep_idx]; x0=x0)
    else
        result3 = SFH.fixed_amr(models[keep_idx], data, logAge[keep_idx], MH[keep_idx], relweights[keep_idx]; x0=x0)
    end

    # Not accurate to the same level as tested above with `result`
    @test ~isapprox(result3.mle.μ, SFRs; rtol=1e-5)
    # Is accurate to a lower level of precision
    @test isapprox(result3.mle.μ, SFRs; rtol=1e-2)
    # And on average, agreement is pretty good
    @test median( (result3.mle.μ .- SFRs) ./ SFRs) < 1e-3
    # Test that truncate_relweights does correct thing
    @test SFH.truncate_relweights(relweightsmin,relweights,logAge) == keep_idx
    # Test that setting relweightsmin keyword to fixed_amr gives same result as result3 above
    result4 = SFH.fixed_amr(models, data, logAge, MH, relweights;
                            relweightsmin=relweightsmin, x0=x0)
    @test isapprox(result3.mle.μ, result4.mle.μ)
   
end
