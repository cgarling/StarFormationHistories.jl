using Distributions: Poisson
using DynamicHMC: NoProgressReport
using StableRNGs: StableRNG
import StarFormationHistories as SFH

using Random: rand!, randperm
using Test


@testset "AMR Fitting and Sampling" begin
    T = Float64
    rng = StableRNG(94823)
    T_max = T(12.0) # 12 Gyr
    α, β, σ = T(0.05), T(-0.05*T_max + -1.0), T(0.2)
    MHmodel = SFH.LinearAMR(T(α), T(β), T_max, (true, true))
    # Test alternate constructor based on two points
    let times = T.((11.0, 1.0))
        newmodel = SFH.LinearAMR((MHmodel(log10(times[1])+9), times[1]),
                                 (MHmodel(log10(times[2])+9), times[2]),
                                 T_max, (true, true))
        @test all(isapprox.(values(SFH.fittable_params(MHmodel)),
                            values(SFH.fittable_params(newmodel))))
    end
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
    x = SFH.calculate_coeffs(MHmodel, disp, Mstars, logAge, MH)
    data = rand.(rng, Poisson.(sum(x .* models))) # Poisson sampled data
    sdata = vec(data)
    data2 = sum(x .* models) # Perfect data, no noise
    sdata2 = vec(data2)

    G = Vector{T}(undef, length(unique_logAge) + SFH.nparams(MHmodel) + SFH.nparams(disp)) # Gradient Vector
    C = similar(first(models)) # Composite model
    sC = vec(C)

    true_vals = vcat(Mstars, SFH.fittable_params(MHmodel)..., SFH.fittable_params(disp)...)
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
            y = SFH.calculate_coeffs(MHmodel, disp, Mstars[rperm], logAge, MH)
            @test x ≈ y[sortperm(logAge)]
        end
    end
    
    # x0 = vcat(SFH.construct_x0_mdf(logAge, convert(T, 13.7); normalize_value=sum(x)), α, β, σ)
    @testset "fg!" begin
        @test SFH.fg!(true, nothing, MHmodel, disp, true_vals, models, data, C, logAge, MH) ≈ nlogL
        @test SFH.fg!(true, G, MHmodel, disp, true_vals, models, data, C, logAge, MH) ≈ nlogL
        @test G ≈ fd_result

        # Test with stacked models / data
        rand!(rng, G) # Fill G with random numbers so we aren't reusing last correct result
        @test SFH.fg!(true, G, MHmodel, disp, true_vals, smodels, sdata, sC, logAge, MH) ≈ nlogL
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
        transformed_vals = vcat(log.(Mstars), log(MHmodel.α), MHmodel.β, log(disp.σ))
        # Gradient of objective with respect to transformed variables
        G_transformed = vcat(fd_result[begin:length(Mstars)] .* Mstars,
                             fd_result[end-2] * MHmodel.α, fd_result[end-1], fd_result[end] * disp.σ)
        # Test with jacobian corrections off, we get -nlogL as expected
        S = SFH.HierarchicalOptimizer(MHmodel, disp, smodels, sdata, sC, logAge,
                                      MH, G, false)
        result = SFH.LogDensityProblems.logdensity_and_gradient(S, transformed_vals)
        @test result[1] ≈ -nlogL # positive logL
        @test result[2] ≈ -G_transformed # positive ∇logL
        # To support Optim.jl, we need G to be updated in place with -∇logL,
        # with variable transformations applied
        @test G ≈ G_transformed
        # Test with jacobian corrections on
        SJ = SFH.HierarchicalOptimizer(MHmodel, disp, smodels, sdata, sC, logAge,
                                       MH, G, true)
        logLJ = -nlogL + sum(log.(Mstars)) + log(MHmodel.α) + log(disp.σ)
        resultj = SFH.LogDensityProblems.logdensity_and_gradient(SJ, transformed_vals)
        @test resultj[1] ≈ logLJ
    end

    @testset "logdensity_and_gradient σ fixed" begin
        let disp = SFH.GaussianDispersion(disp.σ, (false,))
            G2 = G[begin:end-1] # Fixed parameters are not included in G gradient
            transformed_vals = vcat(log.(Mstars), log(MHmodel.α), MHmodel.β)
            # Gradient of objective with respect to transformed variables
            G_transformed = vcat(fd_result[begin:length(Mstars)] .* Mstars,
                                 fd_result[end-2] * MHmodel.α, fd_result[end-1])
            # Test with jacobian corrections off, we get -nlogL as expected
            S = SFH.HierarchicalOptimizer(MHmodel, disp, smodels, sdata, sC, logAge,
                                          MH, G2, false)
            result = SFH.LogDensityProblems.logdensity_and_gradient(S, transformed_vals)
            @test result[1] ≈ -nlogL # positive logL
            @test result[2] ≈ -G_transformed # positive ∇logL
            # To support Optim.jl, we need G to be updated in place with -∇logL,
            # with variable transformations applied
            @test G2 ≈ G_transformed
            # Test with jacobian corrections on
            SJ = SFH.HierarchicalOptimizer(MHmodel, disp, smodels, sdata, sC, logAge,
                                           MH, G2, true)
            logLJ = -nlogL + sum(log.(Mstars)) + log(MHmodel.α)
            resultj = SFH.LogDensityProblems.logdensity_and_gradient(SJ, transformed_vals)
            @test resultj[1] ≈ logLJ
        end
    end

    @testset "logdensity_and_gradient β fixed" begin
        let MHmodel = SFH.LinearAMR(MHmodel.α, MHmodel.β, MHmodel.T_max, (true, false))
            G2 = vcat(G[begin:end-2], G[end])  # Fixed parameters are not included in G gradient
            transformed_vals = vcat(log.(Mstars), log(MHmodel.α), log(disp.σ))
            # Gradient of objective with respect to transformed variables
            G_transformed = vcat(fd_result[begin:length(Mstars)] .* Mstars,
                                 fd_result[end-2] * MHmodel.α, fd_result[end] * disp.σ)
            # Test with jacobian corrections off, we get -nlogL as expected
            S = SFH.HierarchicalOptimizer(MHmodel, disp, smodels, sdata, sC, logAge,
                                          MH, G2, false)
            result = SFH.LogDensityProblems.logdensity_and_gradient(S, transformed_vals)
            @test result[1] ≈ -nlogL # positive logL
            @test result[2] ≈ -G_transformed # positive ∇logL
            # To support Optim.jl, we need G to be updated in place with -∇logL,
            # with variable transformations applied
            @test G2 ≈ G_transformed
            # Test with jacobian corrections on
            SJ = SFH.HierarchicalOptimizer(MHmodel, disp, smodels, sdata, sC, logAge,
                                           MH, G2, true)
            logLJ = -nlogL + sum(log.(Mstars)) + log(MHmodel.α) + log(disp.σ)
            resultj = SFH.LogDensityProblems.logdensity_and_gradient(SJ, transformed_vals)
            @test resultj[1] ≈ logLJ
        end
    end

    @testset "fit_sfh" begin
        # Run fit on perfect, noise-free data
        x0 = Mstars .+ rand(rng, length(Mstars)) .* (Mstars .* 5)
        result = SFH.fit_sfh(SFH.update_params(MHmodel, (MHmodel.α + 0.5, MHmodel.β + 1.0)),
                             SFH.update_params(disp, (disp.σ + 0.1,)),
                             smodels, sdata2, logAge, MH,
                             x0=x0)
        @test result.mle.μ ≈ true_vals # With no error, we should converge exactly
        # MAP will always have some deviation from MLE under transformation, but it should be within
        # a few σ ...
        @test all(isapprox(result.map.μ[i], true_vals[i];
                           atol=result.map.σ[i]) for i in eachindex(true_vals))
        
        # Run fit on noisy data
        rresult = SFH.fit_sfh(SFH.update_params(MHmodel, (MHmodel.α + 0.5, MHmodel.β + 1.0)),
                              SFH.update_params(disp, (disp.σ + 0.1,)),
                              smodels, sdata, logAge, MH, x0=x0)
        # Test that MLE and MAP results are within 3σ of the true answer for all parameters
        @test all(isapprox(rresult.mle.μ[i], true_vals[i];
                           atol=3 * rresult.mle.σ[i]) for i in eachindex(true_vals))
        @test all(isapprox(rresult.map.μ[i], true_vals[i];
                           atol=3 * rresult.map.σ[i]) for i in eachindex(true_vals))

        # Run with fixed parameters on noisy data, verify that best-fit values are unchanged
        fresult = SFH.fit_sfh(SFH.LinearAMR(MHmodel.α, MHmodel.β, MHmodel.T_max, (false, false)),
                              SFH.GaussianDispersion(disp.σ, (false,)),
                              smodels, sdata, logAge, MH, x0=x0)
        @test fresult.map.μ[end-2:end] ≈ [MHmodel.α, MHmodel.β, disp.σ]
        @test fresult.mle.μ[end-2:end] ≈ [MHmodel.α, MHmodel.β, disp.σ]
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
            @test all(sample_fresult.posterior_matrix[end-2:end,:] .≈ [MHmodel.α, MHmodel.β, disp.σ])
        end
        # BFGSResult and CompositeBFGSResult are tested in mzr_test.jl
    end


    # With the majority of functionality validated on LinearAMR,
    # we will test a subset of functionality on the other concrete subtypes.
    @testset "LogarithmicAMR" begin
        logMHmodel = SFH.LogarithmicAMR(T(1e-4), T(5e-5), T_max)
        logx = SFH.calculate_coeffs(logMHmodel, disp, Mstars, logAge, MH)
        logdata = rand.(rng, Poisson.(sum(logx .* models))) # Poisson sampled data
        logsdata = vec(logdata)
        logdata2 = sum(logx .* models) # Perfect data, no noise
        logsdata2 = vec(logdata2)

        G = Vector{T}(undef, length(unique_logAge) + SFH.nparams(logMHmodel) + SFH.nparams(disp)) # Gradient Vector
        C = similar(first(models)) # Composite model
        sC = vec(C)

        log_true_vals = vcat(Mstars, SFH.fittable_params(logMHmodel)..., SFH.fittable_params(disp)...)
        nlogL = 5006.412301383171
        fd_result = [-45.880611037122385, -112.19875116813807, -106.26450305841936, -116.52127060201546, -26.663241555084234, -169.55447041458274, -67.04985033603249, -35.35905744885527, -92.86160016053815, -44.32793487041835, -113.42118317025852, -84.26961680736353, -84.68620384441016, -135.91598184610075, -8.992117309554922, -162.65039039112145, -88.98882640956548, 11.131452949437946, -19.38622772557967, -94.57979940972025, -97.85231115208668, -2.067462235244748e6, -256072.8267379939, -164.0161118198025] # from ForwardDiff.gradient

        # Test alternate constructor based on two points
        let times = T.((11.0, 1.0))
            newmodel = SFH.LogarithmicAMR((logMHmodel(log10(times[1])+9), times[1]),
                                          (logMHmodel(log10(times[2])+9), times[2]),
                                          T_max)
            @test all(isapprox.(values(SFH.fittable_params(logMHmodel)),
                                values(SFH.fittable_params(newmodel))))
        end

        @testset "calculate_coeffs" begin
            # Test that \sum_k r_{j,k} \equiv R_j
            for ii in 1:length(Mstars)
                @test sum(@view(logx[(begin+length(unique_MH)*(ii-1)):(length(unique_MH)*(ii))])) ≈ Mstars[ii]
            end
            @test logx isa Vector{T}
            @test length(logx) == length(logAge)
            # Make sure order of argument logAge is accounted for in returned coefficients
            rperm = randperm(rng, length(unique_logAge))
            let unique_logAge = unique_logAge[rperm]
                # @testset is supposed to create a new local scope, so I would think
                # logAge and MH should be local here by default, but the code following
                # this testset will fail if these are not marked as local explicitly ...
                local logAge = repeat(unique_logAge; inner=length(unique_MH))
                local MH = repeat(unique_MH; outer=length(unique_logAge))
                y = SFH.calculate_coeffs(logMHmodel, disp, Mstars[rperm], logAge, MH)
                @test logx ≈ y[sortperm(logAge)]
            end
        end
        
        @testset "fg!" begin
            @test SFH.fg!(true, nothing, logMHmodel, disp, log_true_vals, models, logdata, C, logAge, MH) ≈ nlogL
            @test SFH.fg!(true, G, logMHmodel, disp, log_true_vals, models, logdata, C, logAge, MH) ≈ nlogL
            @test G ≈ fd_result

            # Test with stacked models / data
            rand!(rng, G) # Fill G with random numbers so we aren't reusing last correct result
            @test SFH.fg!(true, G, logMHmodel, disp, log_true_vals, smodels, logsdata, sC, logAge, MH) ≈ nlogL
            @test G ≈ fd_result
        end

        @testset "fit_sfh" begin
            # Run fit on perfect, noise-free data
            # x0 = Mstars .+ rand(rng, length(Mstars)) .* (Mstars .* 5)
            x0 = SFH.construct_x0_mdf(logAge, convert(T,T_max); normalize_value=sum(logx))
            result = SFH.fit_sfh(SFH.update_params(logMHmodel, (logMHmodel.α + 1e-4, logMHmodel.β + 1e-4)),
                                 SFH.update_params(disp, (disp.σ + 0.1,)),
                                 smodels, logsdata2, logAge, MH,
                                 x0=x0)
            @test result.mle.μ ≈ log_true_vals # With no error, we should converge exactly
            # MAP will always have some deviation from MLE under transformation, but it should be within
            # a few σ ...
            @test all(isapprox(result.map.μ[i], log_true_vals[i];
                               atol=result.map.σ[i]) for i in eachindex(log_true_vals))
            
            # Run fit on noisy data
            rresult = SFH.fit_sfh(SFH.update_params(logMHmodel, (logMHmodel.α + 1e-4, logMHmodel.β + 1e-4)),
                                  SFH.update_params(disp, (disp.σ - 0.1,)),
                                  smodels, logsdata, logAge, MH, x0=x0)
            # Test that MLE and MAP results are within 3σ of the true answer for all parameters
            @test all(isapprox(rresult.mle.μ[i], log_true_vals[i];
                               atol=3 * rresult.mle.σ[i]) for i in eachindex(log_true_vals))
            @test all(isapprox(rresult.map.μ[i], log_true_vals[i];
                               atol=3 * rresult.map.σ[i]) for i in eachindex(log_true_vals))

            # Run with fixed parameters on noisy data, verify that best-fit values are unchanged
            fresult = SFH.fit_sfh(SFH.LogarithmicAMR(logMHmodel.α, logMHmodel.β, logMHmodel.T_max,
                                                     logMHmodel.MH_func, logMHmodel.dMH_dZ, (false, false)),
                                  SFH.GaussianDispersion(disp.σ, (false,)),
                                  smodels, logsdata, logAge, MH, x0=x0)
            @test fresult.map.μ[end-2:end] ≈ [logMHmodel.α, logMHmodel.β, disp.σ]
            @test fresult.mle.μ[end-2:end] ≈ [logMHmodel.α, logMHmodel.β, disp.σ]
            @test all(fresult.map.σ[end-2:end] .== 0) # Uncertainties for fixed quantities should be 0

            @testset "sample_sfh" begin
                Nsteps = 10
                # Test with all variables free
                sample_rresult = @test_nowarn SFH.sample_sfh(rresult, smodels, logsdata, logAge, MH, Nsteps;
                                                             ϵ=0.2, reporter = NoProgressReport(),
                                                             show_convergence=false)
                @test sample_rresult.posterior_matrix isa Matrix{T}
                @test size(sample_rresult.posterior_matrix) == (length(true_vals), Nsteps)
                # Test with fixed parameters
                sample_fresult = @test_nowarn SFH.sample_sfh(fresult, smodels, logsdata, logAge, MH, Nsteps;
                                                             ϵ=0.2, reporter = NoProgressReport(),
                                                             show_convergence=false)
                @test sample_fresult.posterior_matrix isa Matrix{T}
                @test size(sample_fresult.posterior_matrix) == (length(true_vals), Nsteps)
                # Test that all samples have correct fixed parameters
                @test all(sample_fresult.posterior_matrix[end-2:end,:] .≈ [logMHmodel.α, logMHmodel.β, disp.σ])
            end
            # BFGSResult and CompositeBFGSResult are tested in mzr_test.jl
        end
    end
end
