import StarFormationHistories as SFH
using Test

const float_types = (Float32, Float64) # Float types to test most functions with
const float_type_labels = ("Float32", "Float64") # String labels for the above float_types
const rtols = (1e-3, 1e-7) # Relative tolerance levels to use for the above float types

@testset "Core Fitting Functions" begin
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
                # Test second call signature for flattened input
                A2 = T[0,1,0,0,1,0,0,1,0]
                B2 = T[0,0,1,0,0,1,0,0,1]
                models2 = [A2 B2]
                @test SFH.stack_models(models) == models2
                C2 = zeros(T, 9)
                SFH.composite!(C2, coeffs, models2)
                @test C2 == T[0,1,2,0,1,2,0,1,2]
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
                # Test for 3-argument signature
                A = T[1 1 1; 0 0 0; 0 0 0]
                B = T[0 0 0; 1 1 1; 1.5 1.5 1.5]
                models = [A,B]
                coeffs = T[1,2]
                @test SFH.loglikelihood(coeffs, models, data) ≈ -0.5672093513510137 rtol=rtols[i]
                @test SFH.loglikelihood(coeffs, models, data) isa T
                # Test 2-argument call signature for flattened input
                C2 = T[1,2,3,1,2,3,1,2,3]
                data2 = Int64[1,2,2,1,2,2,1,2,2]
                @test SFH.loglikelihood( C2, data2 ) ≈ -0.5672093513510137 rtol=rtols[i]
                @test SFH.loglikelihood( C2, data2 ) isa T
                # Test 3-argument call signature for flattened input
                A2 = T[1,0,0,1,0,0,1,0,0]
                B2 = T[0,1,1.5,0,1,1.5,0,1,1.5]
                models2 = [A2 B2]
                coeffs = T[1,2]
                @test SFH.loglikelihood(coeffs, models2, data2) ≈ -0.5672093513510137 rtol=rtols[i]
                @test SFH.loglikelihood(coeffs, models2, data2) isa T
                # Test zero-data bins: ni=0 should contribute -ci (not 0)
                # ni=0, ci=1.5: 3 bins × -1.5 = -4.5
                # ni=2, ci=3: 6 bins × (2-3-2*log(2/3)) ≈ -1.1344187027020260
                # total ≈ -5.6344187027020260
                C_zero = T[1.5 1.5 1.5; 3 3 3; 3 3 3]
                data_zero = Int64[0 0 0; 2 2 2; 2 2 2]
                @test SFH.loglikelihood( C_zero, data_zero ) ≈ -5.6344187027020260 rtol=rtols[i]
                @test SFH.loglikelihood( C_zero, data_zero ) isa T
            end
        end
    end
    @testset "∇loglikelihood" begin
        for i in eachindex(float_types, float_type_labels)
            label = float_type_labels[i]
            @testset "$label" begin
                T = float_types[i]
                # Test method for single model, matrix inputs
                model = T[0 0 0; 0 0 0; 1 1 1]
                C = T[1 1 1; 2 2 2; 3 3 3]
                data = Int64[1 1 1; 2 2 2; 2 2 2]
                result = SFH.∇loglikelihood( model, C, data )
                @test result ≈ -1 rtol=rtols[i]
                @test result isa T
                # Test method for single model, flattened inputs
                model2 = T[0,0,1,0,0,1,0,0,1]
                C2 = T[1,2,3,1,2,3,1,2,3]
                data2 = Int64[1,2,2,1,2,2,1,2,2]
                result = SFH.∇loglikelihood( model2, C2, data2 )
                @test result ≈ -1 rtol=rtols[i]
                @test result isa T                    
                # Test zero-data bins: ni=0 contributes -c_{i,j} (not 0)
                # Model 1 only has non-zero entries in the zero-data row → grad = -3
                model_zero = T[1 1 1; 0 0 0; 0 0 0]
                C_zero = T[1.5 1.5 1.5; 3 3 3; 3 3 3]
                data_zero = Int64[0 0 0; 2 2 2; 2 2 2]
                result_zero = SFH.∇loglikelihood( model_zero, C_zero, data_zero )
                @test result_zero ≈ -3 rtol=rtols[i]
                @test result_zero isa T
                # Test the method for multiple models, matrix inputs
                result = SFH.∇loglikelihood( [model, model], C, data )
                @test result ≈ [-1, -1] rtol=rtols[i]
                @test result isa Vector{T}
                @test length(result) == 2
                # Test the method for multiple models, flattened inputs
                result = SFH.∇loglikelihood( [model2 model2], C2, data2 )
                @test result ≈ [-1, -1] rtol=rtols[i]
                @test result isa Vector{T}
                @test length(result) == 2
                # Test the method for multiple models that takes `coeffs`, matrix inputs
                models = [ T[1 1 1; 0 0 0; 0 0 0],
                           T[0 0 0; 1 1 1; 0 0 0],
                           T[0 0 0; 0 0 0; 1 1 1] ]
                coeffs = T[1.5, 3, 3]
                result = SFH.∇loglikelihood( coeffs, models, data )
                @test result ≈ [-1, -1, -1] rtol=rtols[i]
                @test result isa Vector{T}
                @test length(result) == 3
                # Test the method for multiple models that takes `coeffs`, flattened inputs
                models2 = reduce(hcat, vec.(models)) # Flatten above models
                coeffs = T[1.5, 3, 3]
                result = SFH.∇loglikelihood( coeffs, models2, data2 )
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
                # Test the method for flattened inputs
                data2 = vec(data)
                models2 = reduce(hcat, vec.(models))
                C2 = models2 * coeffs
                grad2 = Vector{T}(undef,3)
                SFH.∇loglikelihood!( grad2, C2, models2, data2 )
                @test grad2 ≈ [-1, -1, -1] rtol=rtols[i]
                # Now try with zeros in `data`; standard form first
                data3 = Int64[0 0 0; 2 2 2; 2 2 2]
                C3 = sum( coeffs .* models )
                grad3 = Vector{T}(undef,3)
                SFH.∇loglikelihood!( grad3, C3, models, data3 )
                @test grad3 ≈ [-3, -1, -1] rtol=rtols[i]
                # zeros in `data`, flattened inputs
                data4 = vec(data3)
                C4 = models2 * coeffs
                grad4 = Vector{T}(undef,3)
                SFH.∇loglikelihood!( grad4, C4, models2, data4 )
                @test grad4 ≈ [-3, -1, -1] rtol=rtols[i]
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
                @test result isa T
                # Test flattened call signature
                data2 = vec(data)
                models2 = SFH.stack_models( models )
                C2 = Vector{T}(undef,9)
                grad2 = Vector{T}(undef,3)
                result2 = SFH.fg!(true, grad2, coeffs, models2, data2, C2)
                @test -grad2 ≈ [-1, -1, -1] rtol=rtols[i]
                @test -result2 ≈ -1.4180233783775342 rtol=rtols[i]
                @test result2 isa T
                # Try again without providing G (grad)
                result = SFH.fg!(true, nothing, coeffs, models, data, C)
                @test -result ≈ -1.4180233783775342 rtol=rtols[i]
                result2 = SFH.fg!(true, nothing, coeffs, models2, data2, C2)
                @test -result2 ≈ -1.4180233783775342 rtol=rtols[i]
            end
        end
    end
    @testset "construct_x0" begin
        for i in eachindex(float_types, float_type_labels)
            label = float_type_labels[i]
            @testset "$label" begin
                T = float_types[i]
                result = SFH.construct_x0(repeat(T[1,2,3],3), 1e-5; normalize_value=5)
                @test result ≈ repeat([0.015015015015015015, 0.15015015015015015, 1.5015015015015016], 3) rtol=rtols[i]
                @test sum(result) ≈ 5 rtol=rtols[i]
                @test result isa Vector{T}
                # Reverse order of input logAge to ensure it does not assume sorting
                result = SFH.construct_x0(reverse(repeat(T[1,2,3],3)), 1e-5; normalize_value=5)
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
                T_max = 1e-6
                MH = T[-2,-2,-1,-1]
                result = SFH.calculate_cum_sfr(coeffs, logAge, MH, T_max; normalize_value=1, sorted=false)
                @test result[1] == T[1, 2]
                @test result[2] ≈ T[1, 2//3]
                @test result[3] ≈ T[1//30, 2//300]
                @test result[4] ≈ T[-4//3, -4//3]
                # Test normalize_value
                result = SFH.calculate_cum_sfr(coeffs, logAge, MH, T_max; normalize_value=5)
                @test result[1] == T[1, 2]
                @test result[2] ≈ T[1, 2//3]
                @test result[3] ≈ T[5//30, 10//300]
                @test result[4] ≈ T[-4//3, -4//3]
                # Test sorted version
                coeffs = T[1,2,2,4]
                logAge = T[1,1,2,2]
                T_max = 1e-6
                MH = T[-2,-1,-2,-1]
                result = SFH.calculate_cum_sfr(coeffs, logAge, MH, T_max; normalize_value=1, sorted=true)
                @test result[1] == T[1, 2]
                @test result[2] ≈ T[1, 2//3]
                @test result[3] ≈ T[1//30, 2//300]
                @test result[4] ≈ T[-4//3, -4//3]
            end
        end
    end
end
