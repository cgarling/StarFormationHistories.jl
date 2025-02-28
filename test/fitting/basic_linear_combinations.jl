using Distributions: Poisson
using StableRNGs: StableRNG
import MCMCChains
import StarFormationHistories as SFH
using Test

const seedval = 58392 # Seed to use when instantiating new StableRNG objects
const float_types = (Float32, Float64) # Float types to test most functions with
const float_type_labels = ("Float32", "Float64") # String labels for the above float_types

@testset "Basic Linear Combinations" begin
    # Try an easy example with an exact result and only one model
    T = Float64 # LBFGSB.jl wants Float64s so it can pass doubles to the Fortran subroutine
    tset_rtol = 1e-7
    let x0=T[1], models=[T[0 0 0; 0 0 0; 1 1 1]], data=Int64[0 0 0; 0 0 0; 3 3 3]
        # Test LBFGS.jl
        lbfgsb_result = SFH.fit_templates_lbfgsb(models, data; x0=x0, iprint=-1)
        @test lbfgsb_result[2][1] ≈ 3 rtol=tset_rtol
        # Test LBFGS.jl flattened call signature
        smodels = SFH.stack_models(models)
        sdata = vec(data)
        lbfgsb_result2 = SFH.fit_templates_lbfgsb(smodels, sdata; x0=x0, iprint=-1)
        @test lbfgsb_result2[2][1] ≈ 3 rtol=tset_rtol
        # Test fit_templates
        ft_result = SFH.fit_templates(models, data; x0=x0)
        @test ft_result.mle.μ[1] ≈ 3 rtol=tset_rtol
        # Test fit_templates flattened call signature
        ft_result2 = SFH.fit_templates(smodels, sdata; x0=x0)
        @test ft_result2.mle.μ[1] ≈ 3 rtol=tset_rtol
        # Test fit_templates_fast
        ftf_result = SFH.fit_templates_fast(models, data; x0=x0)
        @test ftf_result[1][1] ≈ 3 rtol=tset_rtol
        # Test fit_templates_fast flattened call signature
        ftf_result2 = SFH.fit_templates_fast(smodels, sdata; x0=x0)
        @test ftf_result2[1][1] ≈ 3 rtol=tset_rtol                    
    end
    # Try a harder example with multiple random models
    N_models = 10
    hist_size = (100,100)
    rng = StableRNG(seedval)
    let x=rand(rng,T,N_models), x0=rand(rng,T,N_models), models=[rand(rng,T,hist_size...) for i in 1:N_models], data=sum(x .* models)
        # Test LBFGS.jl
        lbfgsb_result = SFH.fit_templates_lbfgsb(models, data; x0=x0, iprint=-1)
        @test lbfgsb_result[2] ≈ x rtol=tset_rtol
        # Test LBFGS.jl flattened call signature
        smodels = SFH.stack_models(models)
        sdata = vec(data)
        lbfgsb_result = SFH.fit_templates_lbfgsb(smodels,
                                                 sdata; x0=x0, iprint=-1)
        @test lbfgsb_result[2] ≈ x rtol=tset_rtol
        # Test fit_templates
        ft_result = SFH.fit_templates(models, data; x0=x0)
        @test ft_result.mle.μ ≈ x rtol=tset_rtol
        # Test fit_templates flattened call signature
        ft_result2 = SFH.fit_templates(smodels, sdata; x0=x0)
        @test ft_result2.mle.μ ≈ x rtol=tset_rtol                    
        # Test fit_templates_fast
        ftf_result = SFH.fit_templates_fast(models, data; x0=x0)
        @test ftf_result[1] ≈ x rtol=tset_rtol
        # Test fit_templates_fast flattened call signature
        ftf_result2 = SFH.fit_templates_fast(smodels, sdata; x0=x0)
        @test ftf_result2[1] ≈ x rtol=tset_rtol                    
        ##################################################################
        # Make some entries in coefficient vector zero to test convergence
        x2 = copy(x)
        x2[begin] = 0
        x2[end] = 0
        data2 = sum(x2 .* models) 
        # Test LBFGS.jl
        lbfgsb_result2 = SFH.fit_templates_lbfgsb(models, data2; x0=x0, iprint=-1)
        @test lbfgsb_result2[2] ≈ x2 rtol=tset_rtol
        # Test LBFGS.jl flattened call signature
        sdata2 = vec(data2)
        lbfgsb_result2 = SFH.fit_templates_lbfgsb(smodels,
                                                  sdata2; x0=x0, iprint=-1)
        @test lbfgsb_result2[2] ≈ x2 rtol=tset_rtol
        # Test fit_templates
        ft_result = SFH.fit_templates(models, data2; x0=x0)
        @test ft_result.mle.μ ≈ x2 rtol=tset_rtol
        # Test fit_templates flattened call signature
        ft_result2 = SFH.fit_templates(smodels, sdata2; x0=x0)
        @test ft_result2.mle.μ ≈ x2 rtol=tset_rtol                    
        # Test fit_templates_fast
        ftf_result = SFH.fit_templates_fast(models, data2; x0=x0)
        @test ftf_result[1] ≈ x2 rtol=tset_rtol
        # Test fit_templates_fast flattened call signature
        ftf_result2 = SFH.fit_templates_fast(smodels, sdata2; x0=x0)
        @test ftf_result2[1] ≈ x2 rtol=tset_rtol
    end
    # Try an even harder example with Poisson sampling and larger dynamic range of variables
    tset_rtol=1e-2
    let x=rand(rng,N_models).*100, x0=ones(N_models), models=[rand(rng,hist_size...) for i in 1:N_models], data=rand.(rng,Poisson.(sum(x .* models)))
        # Test LBFGS.jl
        lbfgsb_result = SFH.fit_templates_lbfgsb(models, data; x0=x0, iprint=-1)
        @test lbfgsb_result[2] ≈ x rtol=tset_rtol
        # Test LBFGS.jl flattened call signature
        smodels = SFH.stack_models(models)
        sdata = vec(data)
        lbfgsb_result2 = SFH.fit_templates_lbfgsb(smodels,
                                                  sdata; x0=x0, iprint=-1)
        @test lbfgsb_result2[2] ≈ x rtol=tset_rtol
        @test lbfgsb_result[2] ≈ lbfgsb_result2[2] rtol=1e-5 # Test for agreement between signatures
        # Test fit_templates
        ft_result = SFH.fit_templates(models, data; x0=x0)
        @test ft_result.mle.μ ≈ x rtol=tset_rtol
        # Test fit_templates flattened call signature
        ft_result2 = SFH.fit_templates(smodels, sdata; x0=x0)
        @test ft_result2.mle.μ ≈ x rtol=tset_rtol
        @test ft_result.mle.μ ≈ ft_result2.mle.μ rtol=1e-5 # Test for agreement between signatures
        # Test fit_templates_fast
        ftf_result = SFH.fit_templates_fast(models, data; x0=x0)
        @test ftf_result[1] ≈ x rtol=tset_rtol
        # Test fit_templates_fast flattened call signature
        ftf_result2 = SFH.fit_templates_fast(smodels, sdata; x0=x0)
        @test ftf_result2[1] ≈ x rtol=tset_rtol
        @test ftf_result[1] ≈ ftf_result2[1] rtol=1e-5 # Test for agreement between signatures
    end

    @testset "mcmc_sample" begin
        for i in eachindex(float_types, float_type_labels)
            label = float_type_labels[i]
            @testset "$label" begin
                T = float_types[i]
                rng = StableRNG(seedval)
                kmc_conv = SFH.convert_kissmcmc([[T[1,2,3] for i in 1:5] for i in 1:10])
                @test kmc_conv isa Array{T, 3}
                x = rand(rng, T, N_models).* 100
                x0 = ones(T, N_models)
                # x0 = rand(rng, T, nwalkers, length(coeffs)) # Initial walker positions, matrix
                models = [rand(rng, T, hist_size...) for i in 1:N_models]
                data=rand.(rng, Poisson.(sum(x .* models)))
                smodels = SFH.stack_models(models)
                sdata = vec(data)
                nwalkers = 100
                nsteps = 20
                # Test mcmc_sample with x0::Vector{Vector{T}} 
                mc_result = SFH.mcmc_sample(models, data, [copy(x0) for i in 1:nwalkers], nsteps; use_progress_meter=false)
                @test mc_result isa MCMCChains.Chains
                @test size(mc_result) == (nsteps, length(x0), nwalkers)
                @test eltype(Array(mc_result)) == T
                # Test with Matrix x0
                mc_result = SFH.mcmc_sample(models, data, reduce(hcat, copy(x0) for i in 1:nwalkers), nsteps; use_progress_meter=false)
                @test mc_result isa MCMCChains.Chains
                @test size(mc_result) == (nsteps, length(x0), nwalkers)
                @test eltype(Array(mc_result)) == T
                # Test with flattened input, matrix x0
                mc_result = SFH.mcmc_sample(smodels, sdata, reduce(hcat, copy(x0) for i in 1:nwalkers), nsteps; use_progress_meter=false)
                @test mc_result isa MCMCChains.Chains
                @test size(mc_result) == (nsteps, length(x0), nwalkers)
                @test eltype(Array(mc_result)) == T
            end
        end
    end
end
