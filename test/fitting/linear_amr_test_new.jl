using Distributions: Poisson
using DynamicHMC: NoProgressReport
using StableRNGs: StableRNG
import StarFormationHistories as SFH

using Random: rand!, randperm
using Test


@testset "Linear AMR Fitting" begin
    T = Float64
    rng = StableRNG(94823)
    T_max = T(12.0) # 12 Gyr
    α, β, σ = T(0.05), T(-0.05*T_max + -1.0), T(0.2)
    Zmodel = SFH.LinearAMR(T(α), T(β), T_max, (true, true))
    disp = SFH.GaussianDispersion(σ)
    unique_logAge = collect(T, 8.0:0.1:10.0)
    unique_MH = collect(T, -2.5:0.1:0.0)
    logAge = repeat(unique_logAge; inner=length(unique_MH))
    MH = repeat(unique_MH; outer=length(unique_logAge))
    # Now generate models, data, and try to solve
    hist_size = (100, 100)
    Mstars = rand(rng, T, length(unique_logAge))
    models = [rand(rng, T, hist_size...) .* 100 for i in 1:length(logAge)]
    smodels = SFH.stack_models(models)
    x = SFH.calculate_coeffs(Zmodel, disp, Mstars, logAge, MH)
    data = rand.(rng, Poisson.(sum(x .* models))) # Poisson sampled data
    sdata = vec(data)
    data2 = sum(x .* models) # Perfect data, no noise
    sdata2 = vec(data2)

    G = Vector{T}(undef, length(unique_logAge) + SFH.nparams(Zmodel) + SFH.nparams(disp)) # Gradient Vector
    C = similar(first(models)) # Composite model
    sC = vec(C)

    true_vals = vcat(Mstars, SFH.fittable_params(Zmodel)..., SFH.fittable_params(disp)...)
    nlogL = 4903.0966770848445
    # fd_result = ForwardDiff.gradient(X -> SFH.fg_mdf!(nothing, nothing, X, models, data, sum(models .* x), logAge, MH, T_max), true_vals)
    fd_result = [-371.3165965358534, -374.59133163605867, -346.35819347830136, -481.2263116272047, -359.08725914205985, -334.5823872190722, -399.60014542321073, -348.8271518478801, -296.88702094012723, -303.66278027474607, -391.7026965397891, -424.9337130035604, -414.31659503952244, -344.58361511422214, -413.46309475310056, -415.77313946286205, -413.77279436149746, -344.54874278194427, -383.885352570131, -352.5617671307862, -349.2686134326029, -5820.410780372662, -548.2183834376877, 314.6632234464427] # from ForwardDiff.gradient
    @testset "calculate_coeffs" begin
        # Test that \sum_k r_{j,k} \equiv R_j
        for ii in 1:length(Mstars)
            @test sum(@view(x[(begin+length(unique_MH)*(ii-1)):(length(unique_MH)*(ii))])) ≈ Mstars[ii]
        end
        @test x isa Vector{T}
        @test length(x) == length(logAge)
        # Make sure order of argument logAge is accounted for in returned coefficients
        rperm = randperm(rng, length(unique_logAge))
        let unique_logAge = unique_logAge[rperm]
            # @testset is supposed to create a new local scope, so I would think
            # logAge and MH should be local here by default, but the code following
            # this testset will fail if these are not marked as local explicitly ...
            local logAge = repeat(unique_logAge; inner=length(unique_MH))
            local MH = repeat(unique_MH; outer=length(unique_logAge))
            y = SFH.calculate_coeffs(Zmodel, disp, Mstars[rperm], logAge, MH)
            @test x ≈ y[sortperm(logAge)]
        end
    end
    
    # x0 = vcat(SFH.construct_x0_mdf(logAge, convert(T, 13.7); normalize_value=sum(x)), α, β, σ)
    @testset "fg!" begin
        @test SFH.fg!(true, nothing, Zmodel, disp, true_vals, models, data, C, logAge, MH) ≈ nlogL
        @test SFH.fg!(true, G, Zmodel, disp, true_vals, models, data, C, logAge, MH) ≈ nlogL
        @test G ≈ fd_result

        # Test with stacked models / data
        rand!(rng, G) # Fill G with random numbers so we aren't reusing last correct result
        @test SFH.fg!(true, G, Zmodel, disp, true_vals, smodels, sdata, sC, logAge, MH) ≈ nlogL
        @test G ≈ fd_result

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
            rvals = vcat(Mstars[rperm], SFH.fittable_params(Zmodel)..., SFH.fittable_params(disp)...)
            @test SFH.fg!(true, nothing, Zmodel, disp, rvals, rmodels,
                          data, C, rlogAge, rMH) ≈ nlogL
            @test SFH.fg!(true, G, Zmodel, disp, rvals, rmodels,
                          data, C, rlogAge, rMH) ≈ nlogL
            fdr_result = vcat(fd_result[begin:end-3][rperm], fd_result[end-2:end])
            @test G ≈ fdr_result
        end
    end
    
    @testset "logdensity_and_gradient all free" begin
        transformed_vals = vcat(log.(Mstars), log(Zmodel.α), Zmodel.β, log(disp.σ))
        # Gradient of objective with respect to transformed variables
        G_transformed = vcat(fd_result[begin:length(Mstars)] .* Mstars,
                             fd_result[end-2] * Zmodel.α, fd_result[end-1], fd_result[end] * disp.σ)
        # Test with jacobian corrections off, we get -nlogL as expected
        S = SFH.HierarchicalOptimizer(Zmodel, disp, smodels, sdata, sC, logAge,
                                      MH, G, false)
        result = SFH.LogDensityProblems.logdensity_and_gradient(S, transformed_vals)
        @test result[1] ≈ -nlogL # positive logL
        @test result[2] ≈ -G_transformed # positive ∇logL
        # To support Optim.jl, we need G to be updated in place with -∇logL,
        # with variable transformations applied
        @test G ≈ G_transformed
        # Test with jacobian corrections on
        SJ = SFH.HierarchicalOptimizer(Zmodel, disp, smodels, sdata, sC, logAge,
                                       MH, G, true)
        logLJ = -nlogL + sum(log.(Mstars)) + log(Zmodel.α) + log(disp.σ)
        resultj = SFH.LogDensityProblems.logdensity_and_gradient(SJ, transformed_vals)
        @test resultj[1] ≈ logLJ
    end

    @testset "logdensity_and_gradient σ fixed" begin
        let disp = SFH.GaussianDispersion(disp.σ, (false,))
            G2 = G[begin:end-1] # Fixed parameters are not included in G gradient
            transformed_vals = vcat(log.(Mstars), log(Zmodel.α), Zmodel.β)
            # Gradient of objective with respect to transformed variables
            G_transformed = vcat(fd_result[begin:length(Mstars)] .* Mstars,
                                 fd_result[end-2] * Zmodel.α, fd_result[end-1])
            # Test with jacobian corrections off, we get -nlogL as expected
            S = SFH.HierarchicalOptimizer(Zmodel, disp, smodels, sdata, sC, logAge,
                                          MH, G2, false)
            result = SFH.LogDensityProblems.logdensity_and_gradient(S, transformed_vals)
            @test result[1] ≈ -nlogL # positive logL
            @test result[2] ≈ -G_transformed # positive ∇logL
            # To support Optim.jl, we need G to be updated in place with -∇logL,
            # with variable transformations applied
            @test G2 ≈ G_transformed
            # Test with jacobian corrections on
            SJ = SFH.HierarchicalOptimizer(Zmodel, disp, smodels, sdata, sC, logAge,
                                           MH, G2, true)
            logLJ = -nlogL + sum(log.(Mstars)) + log(Zmodel.α)
            resultj = SFH.LogDensityProblems.logdensity_and_gradient(SJ, transformed_vals)
            @test resultj[1] ≈ logLJ
        end
    end

    @testset "logdensity_and_gradient β fixed" begin
        let Zmodel = SFH.LinearAMR(Zmodel.α, Zmodel.β, Zmodel.T_max, (true, false))
            G2 = vcat(G[begin:end-2], G[end])  # Fixed parameters are not included in G gradient
            transformed_vals = vcat(log.(Mstars), log(Zmodel.α), log(disp.σ))
            # Gradient of objective with respect to transformed variables
            G_transformed = vcat(fd_result[begin:length(Mstars)] .* Mstars,
                                 fd_result[end-2] * Zmodel.α, fd_result[end] * disp.σ)
            # Test with jacobian corrections off, we get -nlogL as expected
            S = SFH.HierarchicalOptimizer(Zmodel, disp, smodels, sdata, sC, logAge,
                                          MH, G2, false)
            result = SFH.LogDensityProblems.logdensity_and_gradient(S, transformed_vals)
            @test result[1] ≈ -nlogL # positive logL
            @test result[2] ≈ -G_transformed # positive ∇logL
            # To support Optim.jl, we need G to be updated in place with -∇logL,
            # with variable transformations applied
            @test G2 ≈ G_transformed
            # Test with jacobian corrections on
            SJ = SFH.HierarchicalOptimizer(Zmodel, disp, smodels, sdata, sC, logAge,
                                           MH, G2, true)
            logLJ = -nlogL + sum(log.(Mstars)) + log(Zmodel.α) + log(disp.σ)
            resultj = SFH.LogDensityProblems.logdensity_and_gradient(SJ, transformed_vals)
            @test resultj[1] ≈ logLJ
        end
    end

    @testset "fit_sfh" begin
        # Run fit on perfect, noise-free data
        x0 = Mstars .+ rand(rng, length(Mstars)) .* (Mstars .* 5)
        result = SFH.fit_sfh(SFH.update_params(Zmodel, (Zmodel.α + 0.5, Zmodel.β + 1.0)),
                             SFH.update_params(disp, (disp.σ + 0.1,)),
                             smodels, sdata2, logAge, MH,
                             x0=x0)
        @test result.mle.μ ≈ true_vals # With no error, we should converge exactly
        # MAP will always have some deviation from MLE under transformation, but it should be within
        # a few σ ...
        @test all(isapprox(result.map.μ[i], true_vals[i];
                           atol=result.map.σ[i]) for i in eachindex(true_vals))
        
        # Run fit on noisy data
        rresult = SFH.fit_sfh(SFH.update_params(Zmodel, (Zmodel.α + 0.5, Zmodel.β + 1.0)),
                              SFH.update_params(disp, (disp.σ + 0.1,)),
                              smodels, sdata, logAge, MH, x0=x0)
        # Test that MLE and MAP results are within 3σ of the true answer for all parameters
        @test all(isapprox(rresult.mle.μ[i], true_vals[i];
                           atol=3 * rresult.mle.σ[i]) for i in eachindex(true_vals))
        @test all(isapprox(rresult.map.μ[i], true_vals[i];
                           atol=3 * rresult.map.σ[i]) for i in eachindex(true_vals))

        # Run with fixed parameters on noisy data, verify that best-fit values are unchanged
        fresult = SFH.fit_sfh(SFH.LinearAMR(Zmodel.α, Zmodel.β, Zmodel.T_max, (false, false)),
                              SFH.GaussianDispersion(disp.σ, (false,)),
                              smodels, sdata, logAge, MH, x0=x0)
        @test fresult.map.μ[end-2:end] ≈ [Zmodel.α, Zmodel.β, disp.σ]
        @test fresult.mle.μ[end-2:end] ≈ [Zmodel.α, Zmodel.β, disp.σ]
        @test all(fresult.map.σ[end-2:end] .== 0) # Uncertainties for fixed quantities should be 0

        @testset "sample_sfh" begin
            Nsteps = 10
            # Test with all variables free
            sample_rresult = @test_nowarn SFH.sample_sfh(rresult, smodels, sdata, logAge, MH, Nsteps;
                                                         ϵ=0.2, reporter = NoProgressReport(),
                                                         show_convergence=false)
            @test sample_rresult.posterior_matrix isa Matrix{T}
            @test size(sample_rresult.posterior_matrix) == (length(true_vals), Nsteps)
            # Test with fixed parameters
            sample_fresult = @test_nowarn SFH.sample_sfh(fresult, smodels, sdata, logAge, MH, Nsteps;
                                                         ϵ=0.2, reporter = NoProgressReport(),
                                                         show_convergence=false)
            @test sample_fresult.posterior_matrix isa Matrix{T}
            @test size(sample_fresult.posterior_matrix) == (length(true_vals), Nsteps)
            # Test that all samples have correct fixed parameters
            @test all(sample_fresult.posterior_matrix[end-2:end,:] .≈ [Zmodel.α, Zmodel.β, disp.σ])
        end
        # BFGSResult and CompositeBFGSResult are tested in mzr_test.jl
    end
end