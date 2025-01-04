using Distributions: Poisson
using DynamicHMC: NoProgressReport
using StableRNGs: StableRNG
import StarFormationHistories as SFH
using StatsBase: median
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
            MHmodel = SFH.PowerLawMZR(T(1), T(-1), T(6), (true, true))
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

            x = SFH.calculate_coeffs(MHmodel, disp, Mstars, logAge, MH)
            # Test that \sum_k r_{j,k} \equiv R_j
            for ii in 1:length(Mstars)
                @test sum(@view(x[(begin+length(unique_MH)*(ii-1)):(length(unique_MH)*(ii))])) ≈ Mstars[ii]
            end
            @test x isa Vector{T}
            @test length(x) == length(logAge)
            # Make sure order of argument logAge is accounted for in returned coefficients
            rperm = randperm(rng, length(unique_logAge))
            let unique_logAge = unique_logAge[rperm]
                logAge = repeat(unique_logAge; inner=length(unique_MH))
                MH = repeat(unique_MH; outer=length(unique_logAge))
                y = SFH.calculate_coeffs(MHmodel, disp, Mstars[rperm], logAge, MH)
                @test x ≈ y[sortperm(logAge; rev=true)]
            end
        end
    end
end

@testset "MZR Fitting and Sampling" begin
    T = Float64
    rng = StableRNG(94823)
    MHmodel = SFH.PowerLawMZR(T(1), T(-2), T(6), (true, true))
    disp = SFH.GaussianDispersion(T(2//10))
    unique_logAge = collect(T, 10:-0.1:8)
    unique_MH = collect(T, -2.5:0.1:0.0)
    logAge = repeat(unique_logAge; inner=length(unique_MH))
    MH = repeat(unique_MH; outer=length(unique_logAge))
    # Now generate models, data
    hist_size = (100, 100)
    Mstars = rand(rng, T, length(unique_logAge)) .* 10^6
    models = [rand(rng, T, hist_size...) ./ 10^5 for i in 1:length(logAge)]
    smodels = SFH.stack_models(models)
    x = SFH.calculate_coeffs(MHmodel, disp, Mstars, logAge, MH)
    data = rand.(rng, Poisson.(sum(x .* models))) # Poisson sampled data
    sdata = vec(data)
    data2 = sum(x .* models) # Perfect data, no noise
    sdata2 = vec(data2)

    G = Vector{T}(undef, length(unique_logAge) + SFH.nparams(MHmodel) + SFH.nparams(disp)) # Gradient Vector
    C = similar(first(models)) # Composite model
    sC = vec(C)

    true_vals = vcat(Mstars, SFH.fittable_params(MHmodel)..., SFH.fittable_params(disp)...)
    nlogL = 4917.491550052553
    # Gradient result from ForwardDiff.gradient
    fd_result = [-0.00014821148519279933, -0.00013852612731050684, -0.00012883365448787643, -0.00013224499379724353, -0.00013304264503999908, -0.00013141207154566565, -0.00013956083036823194, -0.00011818954688379261, -8.652358169076432e-5, -0.00010979288408696192, -0.00010299174097368632, -8.968648332825651e-5, -0.00011051661215933656, -9.664530693628639e-5, -0.00011448658353345554, -0.00012390106090859644, -0.00011506434863758139, -0.0001358595304639743, -9.920490692529175e-5, -8.412262247606323e-5, -0.0001095265086536905, -49.091387751034475, -180.99883434459917, -207.04960375880725]
    @testset "fg!" begin
        @test SFH.fg!(true, nothing, MHmodel, disp, true_vals, models, data, C, logAge, MH) ≈ nlogL
        @test SFH.fg!(true, G, MHmodel, disp, true_vals, models, data, C, logAge, MH) ≈ nlogL
        @test G ≈ fd_result
        # Test stacked models / data
        rand!(rng, G) # Fill G with random numbers so we aren't reusing last correct result
        @test SFH.fg!(true, G, MHmodel, disp, true_vals, smodels,
                      sdata, sC, logAge, MH) ≈ nlogL

        # Test that results are not sensitive to the ordering of the logAge argument
        let rperm = randperm(length(unique_logAge))
            rlogAge = repeat(unique_logAge[rperm]; inner=length(unique_MH))
            # Doesn't matter if we permute rMH or not, so long as the same set of (logAge, MH)
            # pairs still exist
            rMH = repeat(unique_MH; outer=length(unique_logAge))
            rmodels = Vector{eltype(models)}(undef, length(models))
            longrperm = Vector{Int}(undef, length(models)) # Transforms logAge to rlogAge
            for i in eachindex(rmodels)
                la = rlogAge[i]
                mh = rMH[i]
                for j in eachindex(logAge, MH, models)
                    if (logAge[j] == la) && (MH[j] == mh)
                        longrperm[i] = j
                        rmodels[i] = models[j]
                    end
                end
            end
            @test rlogAge == logAge[longrperm]
            rvals = vcat(Mstars[rperm], SFH.fittable_params(MHmodel)..., SFH.fittable_params(disp)...)
            @test SFH.fg!(true, nothing, MHmodel, disp, rvals, rmodels,
                          data, C, rlogAge, rMH) ≈ nlogL
            @test SFH.fg!(true, G, MHmodel, disp, rvals, rmodels,
                          data, C, rlogAge, rMH) ≈ nlogL
            fdr_result = vcat(fd_result[begin:end-3][rperm], fd_result[end-2:end])
            @test G ≈ fdr_result
        end
    end

    @testset "logdensity_and_gradient all free" begin
        transformed_vals = vcat(log.(Mstars), log(MHmodel.α), MHmodel.MH0, log(disp.σ))
        # Gradient of objective with respect to transformed variables
        G_transformed = vcat(fd_result[begin:length(Mstars)] .* Mstars,
                             fd_result[end-2] * MHmodel.α, fd_result[end-1], fd_result[end] * disp.σ)
        # Test with jacobian corrections off, we get -nlogL as expected
        S = SFH.HierarchicalOptimizer(MHmodel, disp, smodels, sdata, sC, logAge,
                                      MH, true, G, false)
        result = SFH.LogDensityProblems.logdensity_and_gradient(S, transformed_vals)
        @test result[1] ≈ -nlogL # positive logL
        @test result[2] ≈ -G_transformed # positive ∇logL
        # To support Optim.jl, we need G to be updated in place with -∇logL,
        # with variable transformations applied
        @test G ≈ G_transformed
        # Test with jacobian corrections on
        SJ = SFH.HierarchicalOptimizer(MHmodel, disp, smodels, sdata, sC, logAge,
                                       MH, true, G, true)
        logLJ = -nlogL + sum(log.(Mstars)) + log(MHmodel.α) + log(disp.σ)
        resultj = SFH.LogDensityProblems.logdensity_and_gradient(SJ, transformed_vals)
        @test resultj[1] ≈ logLJ
    end

    @testset "logdensity_and_gradient σ fixed" begin
        let disp = SFH.GaussianDispersion(disp.σ, (false,))
            G2 = G[begin:end-1] # Fixed parameters are not included in G gradient
            transformed_vals = vcat(log.(Mstars), log(MHmodel.α), MHmodel.MH0)
            # Gradient of objective with respect to transformed variables
            G_transformed = vcat(fd_result[begin:length(Mstars)] .* Mstars,
                                 fd_result[end-2] * MHmodel.α, fd_result[end-1])
            # Test with jacobian corrections off, we get -nlogL as expected
            S = SFH.HierarchicalOptimizer(MHmodel, disp, smodels, sdata, sC, logAge,
                                          MH, true, G2, false)
            result = SFH.LogDensityProblems.logdensity_and_gradient(S, transformed_vals)
            @test result[1] ≈ -nlogL # positive logL
            @test result[2] ≈ -G_transformed # positive ∇logL
            # To support Optim.jl, we need G to be updated in place with -∇logL,
            # with variable transformations applied
            @test G2 ≈ G_transformed
            # Test with jacobian corrections on
            SJ = SFH.HierarchicalOptimizer(MHmodel, disp, smodels, sdata, sC, logAge,
                                           MH, true, G2, true)
            logLJ = -nlogL + sum(log.(Mstars)) + log(MHmodel.α)
            resultj = SFH.LogDensityProblems.logdensity_and_gradient(SJ, transformed_vals)
            @test resultj[1] ≈ logLJ
        end
    end
    
    @testset "logdensity_and_gradient MH0 fixed" begin
        let MHmodel = SFH.PowerLawMZR(MHmodel.α, MHmodel.MH0, MHmodel.logMstar0, (true, false))
            G2 = vcat(G[begin:end-2], G[end])  # Fixed parameters are not included in G gradient
            transformed_vals = vcat(log.(Mstars), log(MHmodel.α), log(disp.σ))
            # Gradient of objective with respect to transformed variables
            G_transformed = vcat(fd_result[begin:length(Mstars)] .* Mstars,
                                 fd_result[end-2] * MHmodel.α, fd_result[end] * disp.σ)
            # Test with jacobian corrections off, we get -nlogL as expected
            S = SFH.HierarchicalOptimizer(MHmodel, disp, smodels, sdata, sC, logAge,
                                          MH, true, G2, false)
            result = SFH.LogDensityProblems.logdensity_and_gradient(S, transformed_vals)
            @test result[1] ≈ -nlogL # positive logL
            @test result[2] ≈ -G_transformed # positive ∇logL
            # To support Optim.jl, we need G to be updated in place with -∇logL,
            # with variable transformations applied
            @test G2 ≈ G_transformed
            # Test with jacobian corrections on
            SJ = SFH.HierarchicalOptimizer(MHmodel, disp, smodels, sdata, sC, logAge,
                                  MH, true, G2, true)
            logLJ = -nlogL + sum(log.(Mstars)) + log(MHmodel.α) + log(disp.σ)
            resultj = SFH.LogDensityProblems.logdensity_and_gradient(SJ, transformed_vals)
            @test resultj[1] ≈ logLJ
        end
    end

    @testset "fit_sfh" begin
        # Run fit on perfect, noise-free data
        x0 = Mstars .+ rand(rng, length(Mstars)) .* (Mstars .* 5)
        result = SFH.fit_sfh(SFH.update_params(MHmodel, (MHmodel.α + 0.5, MHmodel.MH0 + 1.0)),
                             SFH.update_params(disp, (disp.σ + 0.1,)),
                             smodels, sdata2, logAge, MH,
                             x0=x0)
        @test result.mle.μ ≈ true_vals # With no error, we should converge exactly
        # MAP will always have some deviation from MLE under transformation, but it should be within
        # a few σ ...
        @test all(isapprox(result.map.μ[i], true_vals[i];
                           atol=result.map.σ[i]) for i in eachindex(true_vals))

        # Run fit on noisy data
        rresult = SFH.fit_sfh(SFH.update_params(MHmodel, (MHmodel.α + 0.5, MHmodel.MH0 + 1.0)),
                              SFH.update_params(disp, (disp.σ + 0.1,)),
                              smodels, sdata, logAge, MH, x0=x0)
        # Test that MLE and MAP results are within 3σ of the true answer for all parameters
        @test all(isapprox(rresult.mle.μ[i], true_vals[i];
                           atol=3 * rresult.mle.σ[i]) for i in eachindex(true_vals))
        @test all(isapprox(rresult.map.μ[i], true_vals[i];
                           atol=3 * rresult.map.σ[i]) for i in eachindex(true_vals))

        # Run with fixed parameters on noisy data, verify that best-fit values are unchanged
        fresult = SFH.fit_sfh(SFH.PowerLawMZR(MHmodel.α, MHmodel.MH0, MHmodel.logMstar0, (false, false)),
                              SFH.GaussianDispersion(disp.σ, (false,)),
                              smodels, sdata, logAge, MH, x0=x0)
        @test fresult.map.μ[end-2:end] ≈ [MHmodel.α, MHmodel.MH0, disp.σ]
        @test fresult.mle.μ[end-2:end] ≈ [MHmodel.α, MHmodel.MH0, disp.σ]
        @test all(fresult.map.σ[end-2:end] .== 0) # Uncertainties for fixed quantities should be 0

        @testset "sample_sfh" begin
            Nsteps = 10
            ϵ = 0.2
            # Test with all variables free
            sample_rresult = @test_nowarn SFH.sample_sfh(rresult, smodels, sdata, logAge, MH, Nsteps;
                                                         ϵ=ϵ, reporter = NoProgressReport(),
                                                         show_convergence=false)
            @test sample_rresult.posterior_matrix isa Matrix{T}
            @test size(sample_rresult.posterior_matrix) == (length(true_vals), Nsteps)
            # Test with fixed parameters
            sample_fresult = @test_nowarn SFH.sample_sfh(fresult, smodels, sdata, logAge, MH, Nsteps;
                                                         ϵ=ϵ, reporter = NoProgressReport(),
                                                         show_convergence=false)
            @test sample_fresult.posterior_matrix isa Matrix{T}
            @test size(sample_fresult.posterior_matrix) == (length(true_vals), Nsteps)
            # Test that all samples have correct fixed parameters
            @test all(sample_fresult.posterior_matrix[end-2:end,:] .≈ [MHmodel.α, MHmodel.MH0, disp.σ])
        end
        @testset "tsample_sfh" begin
            Nsteps = Threads.nthreads()
            ϵ = 0.2
            # Test with all variables free
            tsample_rresult = @test_nowarn SFH.tsample_sfh(rresult, smodels, sdata, logAge, MH, Nsteps;
                                                           ϵ=ϵ, reporter = NoProgressReport(),
                                                           show_convergence=false)
            @test tsample_rresult.posterior_matrix isa Matrix{T}
            @test size(tsample_rresult.posterior_matrix) == (length(true_vals), Nsteps)
            # Test with fixed parameters
            tsample_fresult = @test_nowarn SFH.tsample_sfh(fresult, smodels, sdata, logAge, MH, Nsteps;
                                                           ϵ=ϵ, reporter = NoProgressReport(),
                                                           show_convergence=false)
            @test tsample_fresult.posterior_matrix isa Matrix{T}
            @test size(tsample_fresult.posterior_matrix) == (length(true_vals), Nsteps)
            # Test that all samples have correct fixed parameters
            @test all(tsample_fresult.posterior_matrix[end-2:end,:] .≈ [MHmodel.α, MHmodel.MH0, disp.σ])
        end

        @testset "BFGSResult" begin
            @test rand(fresult.mle) isa Vector{promote_type(eltype(fresult.mle.μ), eltype(fresult.mle.invH))}
            @test length(rand(fresult.mle)) == length(true_vals)
            # Test that median of random samples ≈ median(fresult.mle)
            randmat = rand(rng, fresult.mle, 100_000)
            @test size(randmat) == (length(true_vals), 100_000)
            @test median.(eachrow(randmat)) ≈ median(fresult.mle) rtol=1e-3
            # @test std.(eachrow(randmat)) ≈ std(fresult.mle) rtol=1e-3
            # For fixed parameters, all samples should have same fixed value
            for idx in length(Mstars)+1:length(true_vals)
                @test all(isequal(randmat[idx,i], true_vals[idx]) for i in axes(randmat,2))
            end
            @test SFH.calculate_coeffs(fresult.mle, logAge, MH) ==
                SFH.calculate_coeffs(fresult.mle.MH_model, fresult.mle.disp_model,
                                 @view(fresult.mle.μ[begin:length(unique_logAge)]), logAge, MH)
        end
        
        @testset "CompositeBFGSResult" begin
            @test rand(fresult) isa Vector{promote_type(eltype(fresult.mle.μ), eltype(fresult.mle.invH))}
            @test length(rand(fresult)) == length(true_vals)
            # Test that median of random samples ≈ median(fresult.mle)
            randmat = rand(rng, fresult, 100_000)
            @test size(randmat) == (length(true_vals), 100_000)
            @test median.(eachrow(randmat)) ≈ median(fresult.mle) rtol=1e-3
            # For fixed parameters, all samples should have same fixed value
            for idx in length(Mstars)+1:length(true_vals)
                @test all(isequal(randmat[idx,i], true_vals[idx]) for i in axes(randmat,2))
            end
            @test SFH.calculate_coeffs(fresult, logAge, MH) ==
                SFH.calculate_coeffs(fresult.mle, logAge, MH)
        end
    end
end
