import StarFormationHistories as SFH
import Distributions: Poisson
import Random
import Optim
# Extras
using Test



const float_types = (Float32, Float64)
const float_type_labels = ("Float32", "Float64")
const rtols = (1e-3, 1e-7)
@assert length(float_types) == length(float_type_labels)

@testset "SFH Fitting" begin
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
    
    @testset "loglikelihood and gradient" begin
        tset_rtol = 1e-7
        @test SFH.loglikelihood( Float64[1 1 1; 2 2 2; 3 3 3], Float64[1 1 1; 2 2 2; 2 2 2] ) ≈ -0.5672093513510137 rtol=tset_rtol
        @test SFH.∇loglikelihood( Float64[0 0 0; 0 0 0; 1 1 1], Float64[1 1 1; 2 2 2; 3 3 3], Float64[1 1 1; 2 2 2; 2 2 2] ) ≈ -1.0 rtol=tset_rtol
        # These two should be equivalent and they are
        # SFH.∇loglikelihood([1.0], [Float64[0 0 0; 0 0 0; 1 1 1]], Float64[0 0 0; 0 0 0; 3 3 3])
        # FiniteDifferences.grad(FiniteDifferences.central_fdm(5,1), x->SFH.loglikelihood(x, [Float64[0 0 0; 0 0 0; 1 1 1]], Float64[0 0 0; 0 0 0; 3 3 3]), [1.0])
        # ForwardDiff.gradient(x->SFH.loglikelihood(x, [Float64[0 0 0; 0 0 0; 1 1 1]], Float64[0 0 0; 0 0 0; 3 3 3]), [1.0])
        @test SFH.∇loglikelihood([1.0], [Float64[0 0 0; 0 0 0; 1 1 1]], Float64[0 0 0; 0 0 0; 3 3 3])[1] ≈ 6.0 rtol=tset_rtol
        @test SFH.∇loglikelihood(Float64[0 0 0; 0 0 0; 1 1 1], Float64[0 0 0; 0 0 0; 1 1 1], Float64[0 0 0; 0 0 0; 3 3 3])[1] ≈ 6.0 rtol=tset_rtol
        @test all( isapprox.( SFH.fg(Float64[0 0 0; 0 0 0; 1 1 1], Float64[0 0 0; 0 0 0; 1 1 1], Float64[0 0 0; 0 0 0; 3 3 3]), (-3.8875105980129874, 6.0); rtol=tset_rtol) )
        let x=[1.0], M=[Float64[0 0 0; 0 0 0; 1 1 1]], N=Int64[0 0 0; 0 0 0; 3 3 3], C=zeros(3,3), G=[1.0]
            @btime SFH.loglikelihood($M[1], $N)
            @btime SFH.∇loglikelihood($M[1], $M[1], $N) # @btime SFH.∇loglikelihood($x, $M, $N)
            @btime SFH.fg($M[1], $M[1], $N)
            @btime SFH.fg!($true, $G, $x, $M, $N, $C)
        end
        
    end
    @testset "Solving" begin
        tset_rtol=1e-5
        # Try an easy example with an exact result and only one model
        let x=[1.0], M=[Float64[0 0 0; 0 0 0; 1 1 1]], N=Int64[0 0 0; 0 0 0; 3 3 3], C=zeros(3,3), G=[1.0]
            optim_result1 = @btime Optim.optimize(Optim.only_fg!( (F,G,x)->SFH.fg!(F,G,x,$M,$N,$C) ), $x, Optim.LBFGS()) # 6.162 μs; 14 f(x) and ∇f(x) calls; correct answer
            @test Optim.converged(optim_result1) # Test convergence
            @test Optim.minimizer(optim_result1)[1] ≈ 3.0 rtol=tset_rtol
            optim_result2 = @btime Optim.optimize(Optim.only_fg!( (F,G,x)->SFH.fg!(F,G,x,$M,$N,$C) ), [0.0], [Inf], $x, Optim.Fminbox(Optim.LBFGS())) # 14.7 μs; 22 f(x) and ∇f(x) calls; correct answer
            @test Optim.converged(optim_result2) # Test convergence
            @test Optim.minimizer(optim_result2)[1] ≈ 3.0 rtol=tset_rtol
            # lbfgsb_result = LBFGSB.lbfgsb(f, g!, x; lb=lb, ub=ub)  # LBFGSB has no callback or simultaneous f+∇f calculation interfance, meaning I'd need to calculate the composite model in both the f and ∇f which is inefficient. Try SPGBox
            spg_result1 = @btime SPGBox.spgbox((g,x)->SFH.fg!(true,g,x,$M,$N,$C), $x; eps=1e-8) # 2.2 μs, 9 f(x) and ∇f(x) evaluations; correct answer
            @test spg_result1.ierr == 0 # Test convergence achieved
            @test spg_result1.x[1] ≈ 3.0 rtol=tset_rtol
            spg_result2 = @btime SPGBox.spgbox((g,x)->SFH.fg!(true,g,x,$M,$N,$C), $x; lower=[0.0], upper=[Inf], eps=1e-8) # 2.3 μs, 9 f(x) and ∇f(x) evaluations; correct answer
            @test spg_result2.ierr == 0 # Test convergence achieved
            @test spg_result2.x[1] ≈ 3.0 rtol=tset_rtol
        end
        # Try a harder example with multiple random models
        N_models=10
        hist_size = (100,120)
        Random.seed!(58392) # Not the best way to do this but don't care right now
        let x=rand(N_models), x0=rand(N_models), M=[rand(hist_size...) for i in 1:N_models], N=sum(x .* M), C=zeros(hist_size), G=zeros(N_models)
            # @benchmark SFH.fg!($true, $G, $x, $M, $N, $C) $ ∼114 μs 
            optim_result1 = @btime Optim.optimize(Optim.only_fg!( (F,G,x)->SFH.fg!(F,G,x,$M,$N,$C) ), $x0, Optim.LBFGS()) # 10--30 ms; 70--200 f(x) and ∇f(x) calls; correct answer
            @test Optim.converged(optim_result1) # Test convergence
            @test Optim.minimizer(optim_result1) ≈ x rtol=tset_rtol
            optim_result2 = @btime Optim.optimize(Optim.only_fg!( (F,G,x)->SFH.fg!(F,G,x,$M,$N,$C) ), zeros(N_models), fill(Inf,N_models), $x0, Optim.Fminbox(Optim.LBFGS())) # 50--70 ms; 400--600 f(x) and ∇f(x) calls; correct answer
            @test Optim.converged(optim_result2) # Test convergence
            @test Optim.minimizer(optim_result2) ≈ x rtol=tset_rtol
            # lbfgsb_result = LBFGSB.lbfgsb(f, g!, x; lb=lb, ub=ub)  # LBFGSB has no callback or simultaneous f+∇f calculation interfance, meaning I'd need to calculate the composite model in both the f and ∇f which is inefficient. Try SPGBox
            spg_result1 = @btime SPGBox.spgbox((g,x)->SFH.fg!(true,g,x,$M,$N,$C), $x0; eps=1e-8) # 4 ms, 30 f(x) and ∇f(x) evaluations; correct answer
            @test spg_result1.ierr == 0 # Test convergence achieved
            @test spg_result1.x ≈ x rtol=tset_rtol
            spg_result2 = @btime SPGBox.spgbox((g,x)->SFH.fg!(true,g,x,$M,$N,$C), $x0; lower=zeros(N_models), upper=fill(Inf,N_models), eps=1e-8) # 4 ms, 30 f(x) and ∇f(x) evaluations; correct answer
            @test spg_result2.ierr == 0 # Test convergence achieved
            @test spg_result2.x ≈ x rtol=tset_rtol
        end
        # Try an even harder example with Poisson sampling and larger dynamic range of variables
        tset_rtol=1e-2
        let x=rand(N_models).*100, x0=ones(N_models), M=[rand(hist_size...) for i in 1:N_models], N=rand.(Poisson.(sum(x .* M))), C=zeros(hist_size), G=zeros(N_models)
            # @benchmark SFH.fg!($true, $G, $x, $M, $N, $C) $ ∼140 μs
            # Optim fails here unless I make ∇loglikelihood return NaN when composite < 0,
            # Actually, works if loglikelihood returns -Inf rather than 0 in fail state
            optim_result1 = @btime Optim.optimize(Optim.only_fg!( (F,G,x)->SFH.fg!(F,G,x,$M,$N,$C) ), $x0, Optim.LBFGS())  # 11 ms, 80 f(x) and ∇f(x) calls; correct answer if ∇loglikelihood returns NaN when composite < 0; if it returns 0, then this fails, giving negative coefficients.
            @test Optim.converged(optim_result1) # Test convergence
            @test Optim.minimizer(optim_result1) ≈ x rtol=tset_rtol
            optim_result2 = @btime Optim.optimize(Optim.only_fg!( (F,G,x)->SFH.fg!(F,G,x,$M,$N,$C) ), zeros(N_models), fill(Inf,N_models), $x0, Optim.Fminbox(Optim.LBFGS())) # 40 ms; 250 f(x) and ∇f(x) calls; correct answer even when ∇loglikelihood returns 0 for composite < 0
            @test Optim.converged(optim_result2) # Test convergence
            @test Optim.minimizer(optim_result2) ≈ x rtol=tset_rtol
            spg_result1 = @btime SPGBox.spgbox((g,x)->SFH.fg!(true,g,x,$M,$N,$C), $x0; eps=1e-8) # 5 ms, 30 f(x) and ∇f(x) evaluations; correct answer
            @test spg_result1.ierr == 0 # Test convergence achieved
            @test spg_result1.x ≈ x rtol=tset_rtol
            spg_result2 = @btime SPGBox.spgbox((g,x)->SFH.fg!(true,g,x,$M,$N,$C), $x0; lower=zeros(N_models), upper=fill(Inf,N_models), eps=1e-8) # 5 ms, 30 f(x) and ∇f(x) evaluations; correct answer
            @test spg_result2.ierr == 0 # Test convergence achieved
            @test spg_result2.x ≈ x rtol=tset_rtol
            # @test Optim.minimizer(optim_result1) ≈ Optim.minimizer(optim_result2) ≈ spg_result1.x ≈ spg_result2.x # Test all optimizations have similar result
            @test Optim.minimizer(optim_result1) ≈ Optim.minimizer(optim_result2) ≈ spg_result1.x ≈ spg_result2.x # Test all optimizations have similar result
        end
    end
end
