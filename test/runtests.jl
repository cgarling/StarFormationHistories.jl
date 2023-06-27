import StarFormationHistories as SFH
import Distributions: Poisson
import Random
import StableRNGs: StableRNG
# import Optim
using Test



const float_types = (Float32, Float64)
const float_type_labels = ("Float32", "Float64")
const rtols = (1e-3, 1e-7)
@assert length(float_types) == length(float_type_labels)

@testset verbose=true "SFH Fitting" begin
    @testset "composite!" begin
        for i in eachindex(float_types, float_type_labels)
            label = float_type_labels[i]
            @testset "$label" begin
                T = float_types[i]
                A = T[0 0 0; 1 1 1; 0 0 0]
                B = T[0 0 0; 0 0 0; 1 1 1]
                models = [A,B]
                coeffs = T[1,2]
                C = zeros(T, 3,3)
                SFH.composite!(C, coeffs, models )
                @test C == T[0 0 0; 1 1 1; 2 2 2]
            end
        end
    end
    @testset "loglikelihood" begin
        for i in eachindex(float_types, float_type_labels)
            label = float_type_labels[i]
            @testset "$label" begin
                T = float_types[i]
                C = T[1 1 1; 2 2 2; 3 3 3]
                data = Int64[1 1 1; 2 2 2; 2 2 2]
                @test SFH.loglikelihood( C, data ) ≈ -0.5672093513510137 rtol=rtols[i]
                @test SFH.loglikelihood( C, data ) isa T
            end
        end
    end
    @testset "∇loglikelihood" begin
        for i in eachindex(float_types, float_type_labels)
            label = float_type_labels[i]
            @testset "$label" begin
                T = float_types[i]
                model = T[0 0 0; 0 0 0; 1 1 1]
                C = T[1 1 1; 2 2 2; 3 3 3]
                data = Int64[1 1 1; 2 2 2; 2 2 2]
                result = SFH.∇loglikelihood( model, C, data )
                @test result ≈ -1 rtol=rtols[i]
                @test result isa T
                # Test the method for multiple models
                result = SFH.∇loglikelihood( [model, model], C, data )
                @test result ≈ [-1, -1] rtol=rtols[i]
                @test result isa Vector{T}
                @test length(result) == 2
                # Test the method for multiple models that takes `coeffs` rather than `composite`
                models = [ T[1 1 1; 0 0 0; 0 0 0],
                           T[0 0 0; 1 1 1; 0 0 0],
                           T[0 0 0; 0 0 0; 1 1 1] ]
                coeffs = T[1.5, 3, 3]
                result = SFH.∇loglikelihood( coeffs, models, data )
                @test result ≈ [-1, -1, -1] rtol=rtols[i]
                @test result isa Vector{T}
                @test length(result) == 3
            end
        end
    end
    @testset "∇loglikelihood!" begin
        for i in eachindex(float_types, float_type_labels)
            label = float_type_labels[i]
            @testset "$label" begin
                T = float_types[i]
                data = Int64[1 1 1; 2 2 2; 2 2 2]
                models = [ T[1 1 1; 0 0 0; 0 0 0],
                           T[0 0 0; 1 1 1; 0 0 0],
                           T[0 0 0; 0 0 0; 1 1 1] ]
                coeffs = T[1.5, 3, 3]
                C = sum( coeffs .* models )
                grad = Vector{T}(undef,3)
                SFH.∇loglikelihood!( grad, C, models, data )
                @test grad ≈ [-1, -1, -1] rtol=rtols[i]
            end
        end
    end
    @testset "fg!" begin
        for i in eachindex(float_types, float_type_labels)
            label = float_type_labels[i]
            @testset "$label" begin
                T = float_types[i]
                data = Int64[1 1 1; 2 2 2; 2 2 2]
                models = [ T[1 1 1; 0 0 0; 0 0 0],
                           T[0 0 0; 1 1 1; 0 0 0],
                           T[0 0 0; 0 0 0; 1 1 1] ]
                coeffs = T[1.5, 3, 3]
                C = Matrix{T}(undef,3,3)
                grad = Vector{T}(undef,3)
                result = SFH.fg!(true, grad, coeffs, models, data, C)
                @test -grad ≈ [-1, -1, -1] rtol=rtols[i]
                @test -result ≈ -1.4180233783775342 rtol=rtols[i]
                # Try again without providing G (grad)
                result = SFH.fg!(true, nothing, coeffs, models, data, C)
                @test -result ≈ -1.4180233783775342 rtol=rtols[i]
            end
        end
    end
    @testset "construct_x0" begin
        for i in eachindex(float_types, float_type_labels)
            label = float_type_labels[i]
            @testset "$label" begin
                T = float_types[i]
                result = SFH.construct_x0(repeat(T[1,2,3],3), 4; normalize_value=5)
                @test result ≈ repeat([0.015015015015015015, 0.15015015015015015, 1.5015015015015016], 3) rtol=rtols[i]
                @test sum(result) ≈ 5 rtol=rtols[i]
                @test result isa Vector{T}
                # Reverse order of input logAge to ensure it does not assume sorting
                result = SFH.construct_x0(reverse(repeat(T[1,2,3],3)), 4; normalize_value=5)
                @test result ≈ reverse(repeat([0.015015015015015015, 0.15015015015015015, 1.5015015015015016], 3)) rtol=rtols[i]
                @test sum(result) ≈ 5 rtol=rtols[i]
                @test result isa Vector{T}
            end
        end
    end
    @testset "calculate_cum_sfr" begin
        for i in eachindex(float_types, float_type_labels)
            label = float_type_labels[i]
            @testset "$label" begin
                T = float_types[i]
                coeffs = T[1,2,2,4]
                logAge = T[1,2,1,2]
                max_logAge = 3
                MH = T[-2,-2,-1,-1]
                result = SFH.calculate_cum_sfr(coeffs, logAge, max_logAge, MH; normalize_value=1, sorted=false)
                @test result[1] == T[1, 2]
                @test result[2] ≈ T[1, 2//3]
                @test result[3] ≈ T[1//30, 2//300]
                @test result[4] ≈ T[-4//3, -4//3]
                # Test normalize_value
                result = SFH.calculate_cum_sfr(coeffs, logAge, max_logAge, MH; normalize_value=5)
                @test result[1] == T[1, 2]
                @test result[2] ≈ T[1, 2//3]
                @test result[3] ≈ T[5//30, 10//300]
                @test result[4] ≈ T[-4//3, -4//3]
                # Test sorted version
                coeffs = T[1,2,2,4]
                logAge = T[1,1,2,2]
                max_logAge = 3
                MH = T[-2,-1,-2,-1]
                result = SFH.calculate_cum_sfr(coeffs, logAge, max_logAge, MH; normalize_value=1, sorted=true)
                @test result[1] == T[1, 2]
                @test result[2] ≈ T[1, 2//3]
                @test result[3] ≈ T[1//30, 2//300]
                @test result[4] ≈ T[-4//3, -4//3]
            end
        end
    end
    
    # Benchmarking
    # let x=[1.0], M=[Float64[0 0 0; 0 0 0; 1 1 1]], N=Int64[0 0 0; 0 0 0; 3 3 3], C=zeros(3,3), G=[1.0]
    #     @btime SFH.loglikelihood($M[1], $N)
    #     @btime SFH.∇loglikelihood($M[1], $M[1], $N) # @btime SFH.∇loglikelihood($x, $M, $N)
    #     @btime SFH.fg($M[1], $M[1], $N)
    #     @btime SFH.fg!($true, $G, $x, $M, $N, $C)
    # end

    @testset verbose=true "Solving" begin
        @testset "Basic Linear Combinations" begin
            # Try an easy example with an exact result and only one model
            T=Float64# LBFGSB.jl wants Float64s so it can pass doubles to the Fortran subroutine
            tset_rtol=1e-7
            let x0=T[1], models=[T[0 0 0; 0 0 0; 1 1 1]], data=Int64[0 0 0; 0 0 0; 3 3 3], C=zeros(T,3,3)
                lbfgsb_result = SFH.fit_templates_lbfgsb(models, data; composite=C, x0=x0, iprint=-1)
                @test lbfgsb_result[2][1] ≈ 3 rtol=tset_rtol
            end
            # Try a harder example with multiple random models
            N_models=10
            hist_size=(100,100)
            rng=StableRNG(58392)
            let x=rand(rng,T,N_models), x0=rand(rng,T,N_models), models=[rand(rng,T,hist_size...) for i in 1:N_models], data=sum(x .* models), C=zeros(T,hist_size)
                lbfgsb_result = SFH.fit_templates_lbfgsb(models, data; composite=C, x0=x0, iprint=-1)
                @test lbfgsb_result[2] ≈ x rtol=tset_rtol
                # Make some entries in coefficient vector zero to test convergence
                x2 = copy(x)
                x2[begin] = 0
                x2[end] = 0
                data2 = sum(x2 .* models) 
                lbfgsb_result2 = SFH.fit_templates_lbfgsb(models, data2; composite=C, x0=x0, iprint=-1)
                @test lbfgsb_result2[2] ≈ x2 rtol=tset_rtol
            end
            # Try an even harder example with Poisson sampling and larger dynamic range of variables
            tset_rtol=1e-2
            let x=rand(rng,N_models).*100, x0=ones(N_models), models=[rand(rng,hist_size...) for i in 1:N_models], data=rand.(rng,Poisson.(sum(x .* models))), C=zeros(hist_size)
                lbfgsb_result = SFH.fit_templates_lbfgsb(models, data; composite=C, x0=x0, iprint=-1)
                @test lbfgsb_result[2] ≈ x rtol=tset_rtol
            end
        end
    end
end


@testset "utilities" begin
    for i in eachindex(float_types, float_type_labels)
        label = float_type_labels[i]
        @testset "$label" begin
            T = float_types[i]
            @test SFH.distance_modulus( convert(T, 1e3) ) === convert(T, 10)
            @test SFH.distance_modulus_to_distance( convert(T, 10) ) === convert(T, 1e3)
            @test SFH.arcsec_to_pc(convert(T,20), convert(T,15)) ≈ big"0.9696273591803334731099601686313164294561427182958537274091716194331610209032407" rtol=rtols[i]
            @test SFH.pc_to_arcsec( convert(T, big"0.9696273591803334731099601686313164294561427182958537274091716194331610209032407"), convert(T, 15)) ≈ 20 rtol=rtols[i]

            @test SFH.Y_from_Z(convert(T,1e-3), 0.2485) ≈ 0.2502800000845455 rtol=rtols[i] # Return type not guaranteed
            @test SFH.X_from_Z(convert(T,1e-3)) ≈ 0.748719999867957 rtol=rtols[i] # Return type not guaranteed
            @test SFH.MH_from_Z(convert(T,1e-3), 0.01524) ≈ -1.206576807011171 rtol=rtols[i] # Return type not guaranteed
            @test SFH.Martin2016_complete(T[20.0, 1.0, 25.0, 1.0]...) ≈ big"0.9933071490757151444406380196186748196062559910927034697307877569401159160854199" rtol=rtols[i]
            @test SFH.Martin2016_complete(T[20.0, 1.0, 25.0, 1.0]...) isa T
            @test SFH.exp_photerr(T[20.0, 1.05, 10.0, 32.0, 0.01]...) ≈ big"0.01286605230281143891186877135084309862554426640053421106995766903206843498217022"
            @test SFH.exp_photerr(T[20.0, 1.05, 10.0, 32.0, 0.01]...) isa T
        end
    end
end
