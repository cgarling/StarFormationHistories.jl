import StarFormationHistories as SFH
import InitialMassFunctions: Salpeter1955
import Distributions: Poisson, Uniform, pdf
import Random
import StableRNGs: StableRNG
import StaticArrays: SVector
import QuadGK: quadgk
# import Optim
using Test

const seedval = 58392 # Seed to use when instantiating new StableRNG objects
const float_types = (Float32, Float64) # Float types to test most functions with
const float_type_labels = ("Float32", "Float64") # String labels for the above float_types
const rtols = (1e-3, 1e-7) # Relative tolerance levels to use for the above float types 
@assert length(float_types) == length(float_type_labels) == length(rtols)

@testset verbose=true "CMD Simulation" begin
    @testset "ingest_mags" begin
        for i in eachindex(float_types, float_type_labels)
            label = float_type_labels[i]
            @testset "$label" begin
                T = float_types[i]
                # Test (::AbstractVector, ::AbstratMatrix) method, first (length(m_ini), nfilters) matrix
                result = SFH.ingest_mags(T[1,2,3,4], T[1 1; 2 2; 3 3; 4 4])
                @test result isa Base.ReinterpretArray
                @test eltype(result) == SVector{2,T}
                # Test with (nfilters, length(m_ini)) matrix
                result2 = SFH.ingest_mags(T[1,2,3,4], T[1 2 3 4; 1 2 3 4])
                @test result == result2
                # Test with vector of vectors; first (length(m_ini), nfilters)
                result3 = SFH.ingest_mags(T[1,2,3,4], [T[1,1], T[2,2], T[3,3],T[4,4]])
                @test result == result3
                # Test with vector of vectors (nfilters, length(m_ini))
                result4 = SFH.ingest_mags(T[1,2,3,4], [T[1,2,3,4], T[1,2,3,4]])
                @test result == result4
                @test_throws ArgumentError SFH.ingest_mags(T[1,2,3,4], zeros(T,3,3))
                @test_throws ArgumentError SFH.ingest_mags(T[1,2,3,4], [T[1,1], T[2,2], T[3,3]])
                @test_throws ArgumentError SFH.ingest_mags(T[1,2,3,4], [T[1,2,3,4,5], T[1,2,3,4,5]])
                @test_throws ArgumentError SFH.ingest_mags((1,2,3,4), T[1 2 3 4; 1 2 3 4])
            end
        end
    end
    @testset "sort_ingested" begin
        for i in eachindex(float_types, float_type_labels)
            label = float_type_labels[i]
            @testset "$label" begin
                T = float_types[i]
                result = SFH.sort_ingested(T[1,2,3,4], SFH.ingest_mags(T[1,2,3,4], [T[1,2,3,4], T[1,2,3,4]]))
                @test result[1] == T[1,2,3,4]
                @test eltype(result[1]) == T
                @test result[2] == SFH.ingest_mags(T[1,2,3,4], [T[1,2,3,4], T[1,2,3,4]])
                @test eltype(result[2]) == SVector{2,T}
                result = SFH.sort_ingested(T[2,3,1,4], SFH.ingest_mags(T[1,2,3,4], [T[1,2,3,4], T[1,2,3,4]]))
                @test result[1] == T[1,2,3,4]
                @test eltype(result[1]) == T
                @test result[2] ==
                    SFH.ingest_mags(T[1,2,3,4], [T[1,2,3,4], T[1,2,3,4]])[sortperm(T[2,3,1,4])]
                @test eltype(result[2]) == SVector{2,T}
            end
        end
    end
    @testset "mass_limits" begin
        for i in eachindex(float_types, float_type_labels)
            label = float_type_labels[i]
            @testset "$label" begin
                T = float_types[i]
                m_ini = T[0.05,0.1,0.2,0.3]
                mags = [T[4],T[3],T[2],T[1]]
                result = SFH.mass_limits(m_ini, mags, ["F090W"], T(5//2), "F090W")
                @test result == (T(15//100), T(30//100))
                # Test if mag_lim is infinite
                result2 = SFH.mass_limits(m_ini, mags, ["F090W"], T(Inf), "F090W")
                @test result2 == extrema(m_ini)
                # Test if mag_lim is fainter than maximum(mags)
                result3 = SFH.mass_limits(m_ini, mags, ["F090W"], T(10), "F090W")
                @test result3 == extrema(m_ini)
                # If mag_lim is brighter than all the stars provided, throw error
                @test_throws DomainError SFH.mass_limits(m_ini, mags, ["F090W"], T(0), "F090W")
                # If mag_lim_name is not in mag_names, throw error
                @test_throws ArgumentError SFH.mass_limits(m_ini, mags, ["F090W"], T(2.5), "F150W")
            end
        end
    end
    @testset "Binary Model Types" begin # Test type creation routines
        @test length(SFH.NoBinaries()) == 1 # No arguments
        for i in eachindex(float_types, float_type_labels)
            label = float_type_labels[i]
            @testset "$label" begin
                T = float_types[i]
                @test length(SFH.RandomBinaryPairs(T(4//10))) == 2
                @test length(SFH.BinaryMassRatio(T(4//10), Uniform(T(0), T(1)))) == 2
            end
        end
    end
    @testset "sample_system" begin
        rng=StableRNG(seedval)
        for i in eachindex(float_types, float_type_labels)
            label = float_type_labels[i]
            @testset "$label" begin
                T = float_types[i]
                imf = Salpeter1955(T(0.1), T(100))
                @test SFH.sample_system(imf, rng, SFH.NoBinaries()) isa SVector{1,T}
                @test SFH.sample_system(imf, rng, SFH.RandomBinaryPairs(T(4//10))) isa SVector{2,T}
                @test SFH.sample_system(imf, rng, SFH.BinaryMassRatio(T(4//10), Uniform(T(0), T(1)))) isa SVector{2,T}
            end
        end
    end
    @testset "generate_stars" begin
        rng = StableRNG(seedval)
        for i in eachindex(float_types, float_type_labels)
            label = float_type_labels[i]
            @testset "$label" begin
                T = float_types[i]
                imf = Salpeter1955(T(0.1), T(100))
                total_mass = T(10^5)
                dmod = T(20)
                # Use some masses in the 0.1 -> 0.8 solar mass range
                m_ini = range(T(1//10), T(4//5); step=T(1//10))
                # Mags interpolated from an old-metal poor isochrone for order-of-magnitude scaling
                f606w_mags = T[ 12.73830341586291, 10.017833567325411, 9.041398536300997, 8.331313172834694, 7.3892413746605765, 6.227971374447669, 4.93799980882831, -3.026]
                f814w_mags = T[11.137576129331675, 8.951605387408511, 8.064408199510426, 7.424658866406447, 6.6025729402403, 5.584714238650148, 4.448999828801471, -4.107]
                mags = [f606w_mags, f814w_mags]
                mag_names = ["F606W", "F814W"]
                # Figure out the percentage of total mass represented
                # by stars between miniumum(m_ini) and maximum(m_ini).
                # See the notes in the `generate_stars_mass` for more details.
                mass_frac = (quadgk(x->x*pdf(imf,x), minimum(m_ini), maximum(m_ini))[1] / quadgk(x->x*pdf(imf,x), minimum(imf), imf.upper)[1]) # maximum(imf))[1] )

                ###################################
                ####### Testing generate_stars_mass
                # Test with NoBinaries() model
                result = SFH.generate_stars_mass(m_ini, mags, mag_names, total_mass, imf; dist_mod=dmod, rng=rng, mag_lim=T(Inf), mag_lim_name=mag_names[2], binary_model=SFH.NoBinaries())
                @test result[1] isa Vector{SVector{1,T}}
                @test result[2] isa Vector{SVector{2,T}}
                @test length(result[1]) == length(result[2])
                # Test total mass of the returned stars
                # There will be less random variance with higher `total_mass`.
                @test reduce(+,reduce(+,result[1])) ≈ total_mass * mass_frac rtol=5e-2
                
                # Test with RandomBinaryPairs() model
                result = SFH.generate_stars_mass(m_ini, mags, mag_names, total_mass, imf; dist_mod=dmod, rng=rng, mag_lim=T(Inf), mag_lim_name=mag_names[2], binary_model=SFH.RandomBinaryPairs(T(4//10)))
                @test result[1] isa Vector{SVector{2,T}}
                @test result[2] isa Vector{SVector{2,T}}
                @test length(result[1]) == length(result[2])
                # println((reduce(+,reduce(+,result[1])) .- total_mass * mass_frac * 14//10) / total_mass / mass_frac / 14//10)
                @test reduce(+,reduce(+,result[1])) ≈ total_mass * mass_frac * 14//10 rtol=5e-2

                # Test with BinaryMassRatio() model
                result = SFH.generate_stars_mass(m_ini, mags, mag_names, total_mass, imf; dist_mod=dmod, rng=rng, mag_lim=T(Inf), mag_lim_name=mag_names[2], binary_model=SFH.BinaryMassRatio(T(4//10), Uniform(T(1//10), T(1))))
                @test result[1] isa Vector{SVector{2,T}}
                @test result[2] isa Vector{SVector{2,T}}
                @test length(result[1]) == length(result[2])
                @test reduce(+,reduce(+,result[1])) ≈ total_mass * mass_frac rtol=5e-2

                # Test errors
                @test_throws ArgumentError SFH.generate_stars_mass(m_ini, mags, mag_names, total_mass, imf; dist_mod=dmod, rng=rng, mag_lim=T(25), mag_lim_name="V", binary_model=SFH.BinaryMassRatio(T(4//10), Uniform(T(1//10), T(1))))

                ##################################
                ####### Testing generate_stars_mag
                absmaglim = T(-7)
                # Test with NoBinaries() model
                result = SFH.generate_stars_mag(m_ini, mags, mag_names, absmaglim, mag_names[2], imf; dist_mod=dmod, rng=rng, mag_lim=T(Inf), mag_lim_name=mag_names[2], binary_model=SFH.NoBinaries())
                @test result[1] isa Vector{SVector{1,T}}
                @test result[2] isa Vector{SVector{2,T}}
                @test length(result[1]) == length(result[2])
                # Test that total magnitude of sampled population is (slightly)
                # brighter than the requested absmaglim
                apparent_mag = SFH.flux2mag(sum(map(x->SFH.mag2flux(x[2]), result[2])))
                abs_mag = apparent_mag - dmod
                @test T(-0.05) <= (abs_mag - absmaglim) <= T(0)

                # Test with RandomBinaryPairs() model
                result = SFH.generate_stars_mag(m_ini, mags, mag_names, absmaglim, mag_names[2], imf; dist_mod=dmod, rng=rng, mag_lim=T(Inf), mag_lim_name=mag_names[2], binary_model=SFH.RandomBinaryPairs(T(4//10)))
                @test result[1] isa Vector{SVector{2,T}}
                @test result[2] isa Vector{SVector{2,T}}
                @test length(result[1]) == length(result[2])
                # Test that total magnitude of sampled population is (slightly)
                # brighter than the requested absmaglim
                apparent_mag = SFH.flux2mag(sum(map(x->SFH.mag2flux(x[2]), result[2])))
                abs_mag = apparent_mag - dmod
                @test T(-0.05) <= (abs_mag - absmaglim) <= T(0)

                # Test with BinaryMassRatio() model
                result = SFH.generate_stars_mag(m_ini, mags, mag_names, absmaglim, mag_names[2], imf; dist_mod=dmod, rng=rng, mag_lim=T(Inf), mag_lim_name=mag_names[2], binary_model=SFH.BinaryMassRatio(T(4//10), Uniform(T(1//10),T(1))))
                @test result[1] isa Vector{SVector{2,T}}
                @test result[2] isa Vector{SVector{2,T}}
                @test length(result[1]) == length(result[2])
                # Test that total magnitude of sampled population is (slightly)
                # brighter than the requested absmaglim
                apparent_mag = SFH.flux2mag(sum(map(x->SFH.mag2flux(x[2]), result[2])))
                abs_mag = apparent_mag - dmod
                @test T(-0.05) <= (abs_mag - absmaglim) <= T(0)

                # Test errors
                @test_throws ArgumentError SFH.generate_stars_mag(m_ini, mags, mag_names, absmaglim, "V", imf; dist_mod=dmod, rng=rng, mag_lim=T(Inf), mag_lim_name=mag_names[2], binary_model=SFH.BinaryMassRatio(T(4//10), Uniform(T(1//10),T(1))))
                @test_throws ArgumentError SFH.generate_stars_mag(m_ini, mags, mag_names, absmaglim, mag_names[2], imf; dist_mod=dmod, rng=rng, mag_lim=T(25), mag_lim_name="V", binary_model=SFH.BinaryMassRatio(T(4//10), Uniform(T(1//10),T(1))))
            end
        end
    end
end

#####################################################################
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
            rng=StableRNG(seedval)
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
            @test SFH.mag2flux(T(-5//2)) === T(10)
            @test SFH.mag2flux(T(-3//2), 1) === T(10)
            @test SFH.flux2mag(T(10)) === T(-5//2)
            @test SFH.flux2mag(T(10), 1) === T(-3//2)
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
