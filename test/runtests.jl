import StarFormationHistories as SFH
using InitialMassFunctions: Salpeter1955, Kroupa2001
using Distributions: Poisson, Uniform, pdf, median
import Random
using StableRNGs: StableRNG
using StaticArrays: SVector
using QuadGK: quadgk
import MCMCChains
import DynamicHMC
# import Optim
using Test, SafeTestsets
using Logging: with_logger, ConsoleLogger, Error

# Run doctests first
import Documenter: DocMeta, doctest
DocMeta.setdocmeta!(SFH, :DocTestSetup, :(using StarFormationHistories); recursive=true)
doctest(SFH)

# Setup for other tests
const seedval = 58392 # Seed to use when instantiating new StableRNG objects
const float_types = (Float32, Float64) # Float types to test most functions with
const float_type_labels = ("Float32", "Float64") # String labels for the above float_types
const rtols = (1e-3, 1e-7) # Relative tolerance levels to use for the above float types 
@assert length(float_types) == length(float_type_labels) == length(rtols)

@testset verbose=true "StarFormationHistories.jl" begin
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
            @test SFH.binary_system_fraction(SFH.NoBinaries()) == 0
            @test SFH.binary_mass_fraction(SFH.NoBinaries(), one) == 0
            for i in eachindex(float_types, float_type_labels)
                label = float_type_labels[i]
                @testset "$label" begin
                    T = float_types[i]
                    @test length(SFH.RandomBinaryPairs(T(4//10))) == 2
                    @test SFH.binary_system_fraction(SFH.RandomBinaryPairs(T(4//10))) == T(4//10)
                    @test SFH.binary_mass_fraction(SFH.RandomBinaryPairs(T(4//10)), one) isa T
                    @test length(SFH.BinaryMassRatio(T(4//10), Uniform(T(0), T(1)))) == 2
                    @test SFH.binary_system_fraction(SFH.BinaryMassRatio(T(4//10))) == T(4//10)
                    @test SFH.binary_mass_fraction(SFH.BinaryMassRatio(T(4//10)), one) isa T
                end
            end
        end
        @testset "sample_system" begin
            rng=StableRNG(seedval)
            for i in eachindex(float_types, float_type_labels)
                label = float_type_labels[i]
                @testset "$label" begin
                    T = float_types[i]
                    for imf in (Salpeter1955(T(0.1), T(100)), Kroupa2001(T(0.1), T(100)))
                        # Salpeter1955 is just a truncated Pareto distribution which always has eltype Float64
                        # while Kroupa2001 is a custom distribution whose eltype is based on its arguments
                        S = eltype(imf) 
                        @test SFH.sample_system(imf, rng, SFH.NoBinaries()) isa SVector{1,S}
                        @test SFH.sample_system(imf, rng, SFH.RandomBinaryPairs(T(4//10))) isa SVector{2,S}
                        @test SFH.sample_system(imf, rng, SFH.BinaryMassRatio(T(4//10), Uniform(T(0), T(1)))) isa SVector{2,S}
                    end
                end
            end
        end
        @testset "generate_stars and model_cmd" begin
            rng = StableRNG(seedval)
            for i in eachindex(float_types, float_type_labels)
                label = float_type_labels[i]
                @testset "$label" begin
                    T = float_types[i]
                    imf = Salpeter1955(T(0.1), T(100))
                    S = eltype(imf) # Type of rand(imf) for sampled stellar masses
                    total_mass = T(10^5)
                    dmod = T(20)
                    # Use some masses in the 0.1 -> 0.8 solar mass range
                    m_ini = range(T(1//10), T(4//5); step=T(1//10))
                    # Mags interpolated from an old-metal poor isochrone for order-of-magnitude scaling
                    f606w_mags = T[ 12.73830341586291, 10.017833567325411, 9.041398536300997, 8.331313172834694, 7.3892413746605765, 6.227971374447669, 4.93799980882831, -3.026]
                    f814w_mags = T[ 11.137576129331675, 8.951605387408511, 8.064408199510426, 7.424658866406447, 6.6025729402403, 5.584714238650148, 4.448999828801471, -4.107]
                    mags = [f606w_mags, f814w_mags]
                    mag_names = ["F606W", "F814W"]
                    # Figure out the percentage of total mass represented
                    # by stars between miniumum(m_ini) and maximum(m_ini).
                    # See the notes in the `generate_stars_mass` for more details.
                    mass_frac = (quadgk(x->x*pdf(imf,x), minimum(m_ini), maximum(m_ini))[1] / quadgk(x->x*pdf(imf,x), minimum(imf), maximum(imf))[1]) # maximum(imf))[1] )

                    ###################################
                    ####### Testing generate_stars_mass
                    # Test with NoBinaries() model
                    result = SFH.generate_stars_mass(m_ini, mags, mag_names, total_mass, imf; dist_mod=dmod, rng=rng, mag_lim=T(Inf), mag_lim_name=mag_names[2], binary_model=SFH.NoBinaries())
                    @test result[1] isa Vector{SVector{1,S}}
                    @test result[2] isa Vector{SVector{2,T}}
                    @test length(result[1]) == length(result[2])
                    # Test total mass of the returned stars
                    # There will be less random variance with higher `total_mass`.
                    @test reduce(+,reduce(+,result[1])) ≈ total_mass * mass_frac rtol=5e-2
                    
                    # Test with RandomBinaryPairs() model
                    result = SFH.generate_stars_mass(m_ini, mags, mag_names, total_mass, imf; dist_mod=dmod, rng=rng, mag_lim=T(Inf), mag_lim_name=mag_names[2], binary_model=SFH.RandomBinaryPairs(T(4//10)))
                    @test result[1] isa Vector{SVector{2,S}}
                    @test result[2] isa Vector{SVector{2,T}}
                    @test length(result[1]) == length(result[2])
                    # println((reduce(+,reduce(+,result[1])) .- total_mass * mass_frac * 14//10) / total_mass / mass_frac / 14//10)
                    @test reduce(+,reduce(+,result[1])) ≈ total_mass * mass_frac * 14//10 rtol=5e-2

                    # Test with BinaryMassRatio() model
                    result = SFH.generate_stars_mass(m_ini, mags, mag_names, total_mass, imf; dist_mod=dmod, rng=rng, mag_lim=T(Inf), mag_lim_name=mag_names[2], binary_model=SFH.BinaryMassRatio(T(4//10), Uniform(T(1//10), T(1))))
                    @test result[1] isa Vector{SVector{2,S}}
                    @test result[2] isa Vector{SVector{2,T}}
                    @test length(result[1]) == length(result[2])
                    @test reduce(+,reduce(+,result[1])) ≈ total_mass * mass_frac rtol=5e-2

                    # Test errors
                    @test_throws ArgumentError SFH.generate_stars_mass(m_ini, mags, mag_names, total_mass, imf; dist_mod=dmod, rng=rng, mag_lim=T(25), mag_lim_name="V", binary_model=SFH.BinaryMassRatio(T(4//10), Uniform(T(1//10), T(1))))

                    ##################################
                    ####### Testing generate_stars_mag
                    absmaglim = T(-8)
                    # Test with NoBinaries() model
                    result = SFH.generate_stars_mag(m_ini, mags, mag_names, absmaglim, mag_names[2], imf; dist_mod=dmod, rng=rng, mag_lim=T(Inf), mag_lim_name=mag_names[2], binary_model=SFH.NoBinaries())
                    @test result[1] isa Vector{SVector{1,S}}
                    @test result[2] isa Vector{SVector{2,T}}
                    @test length(result[1]) == length(result[2])
                    # Test that total magnitude of sampled population is (slightly)
                    # brighter than the requested absmaglim
                    apparent_mag = SFH.flux2mag(sum(map(x->SFH.mag2flux(x[2]), result[2])))
                    abs_mag = apparent_mag - dmod
                    @test T(-0.05) <= (abs_mag - absmaglim) <= T(0)

                    # Test with RandomBinaryPairs() model
                    result = SFH.generate_stars_mag(m_ini, mags, mag_names, absmaglim, mag_names[2], imf; dist_mod=dmod, rng=rng, mag_lim=T(Inf), mag_lim_name=mag_names[2], binary_model=SFH.RandomBinaryPairs(T(4//10)))
                    @test result[1] isa Vector{SVector{2,S}}
                    @test result[2] isa Vector{SVector{2,T}}
                    @test length(result[1]) == length(result[2])
                    # Test that total magnitude of sampled population is (slightly)
                    # brighter than the requested absmaglim
                    apparent_mag = SFH.flux2mag(sum(map(x->SFH.mag2flux(x[2]), result[2])))
                    abs_mag = apparent_mag - dmod
                    @test T(-0.05) <= (abs_mag - absmaglim) <= T(0)

                    # Test with BinaryMassRatio() model
                    result = SFH.generate_stars_mag(m_ini, mags, mag_names, absmaglim, mag_names[2], imf; dist_mod=dmod, rng=rng, mag_lim=T(Inf), mag_lim_name=mag_names[2], binary_model=SFH.BinaryMassRatio(T(4//10), Uniform(T(1//10),T(1))))
                    @test result[1] isa Vector{SVector{2,S}}
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

                    ####################################
                    # Test generate_stars_mass_composite
                    # We'll spoof a second isochrone by just shifting
                    # the F814W mags slightly lower and slightly altering m_ini
                    composite_masses = [m_ini, m_ini .+ T(0.01)]
                    composite_mags = [mags, [mags[1], mags[2] .- T(0.02)]]
                    @test length(composite_masses) == length(composite_mags)
                    # nisochrones = length(composite_masses)
                    
                    result = SFH.generate_stars_mass_composite(composite_masses, composite_mags, mag_names, total_mass, T[1//2, 1//2], imf; dist_mod=dmod, rng=rng, mag_lim=T(Inf), mag_lim_name=mag_names[2], binary_model=SFH.NoBinaries())
                    @test length(result) == 2    # Masses and magnitudes
                    @test length(result[1]) == length(composite_masses) # Number of isochrones
                    @test length(result[2]) == length(composite_masses) # Number of isochrones
                    for i in eachindex(composite_masses, composite_mags) # Isochrone i 
                        @test length(result[1][i]) == length(result[2][i]) # Number of masses equals number of mags
                        @test result[1][i] isa Vector{SVector{1,S}} # Masses
                        @test result[2][i] isa Vector{SVector{2,T}} # Magnitudes
                    end
                    # Test total mass of the returned stars
                    # There will be less random variance with higher `total_mass`.
                    @test sum( reduce(+,reduce(+,i)) for i in result[1] ) ≈ total_mass * mass_frac rtol=5e-2

                    # Test errors
                    @test_throws ArgumentError SFH.generate_stars_mass_composite([m_ini, composite_masses...], composite_mags, mag_names, total_mass, T[1//2, 1//2], imf; dist_mod=dmod, rng=rng, mag_lim=T(Inf), mag_lim_name=mag_names[2], binary_model=SFH.NoBinaries()) # Test bad array input

                    # Test add_metadata, values arbitrary
                    am_result = SFH.add_metadata(result, (:F606W, :F814W); logAge=[6.6,6.7], MH=[1.0, 1.0])
                    @test am_result isa Vector{<:NamedTuple}
                    @test all([a.F606W for a in am_result] .== [a[1] for a in reduce(vcat, result[2])])
                    @test all([a.F814W for a in am_result] .== [a[2] for a in reduce(vcat, result[2])])
                    @test all(am_result[i].logAge == 6.6 for i in eachindex(result[1][1]))
                    @test all(am_result[i].logAge == 6.7 for i in eachindex(result[1][2]) .+ length(result[1][1]))
                    @test all(am_result[i].MH == 1.0 for i in eachindex(am_result))
                    @test_throws "length" SFH.add_metadata((result[1][1:1], result[2]), (:F606W, :F814W); logAge=[6.6,6.7], MH=[1.0, 1.0])

                    @test_throws "mag_symbols" SFH.add_metadata(result, (:F606W,))
                    @test_throws "keyword" SFH.add_metadata(result, (:F606W,:F814W); logAge=[6.6])
                    @test_throws "length" SFH.add_metadata((result[1][1], result[1][2]), 
                                 (:F606W, :F814W); logAge=[6.6,6.7], MH=[1.0, 1.0])

                    ####################################
                    # Test generate_stars_mag_composite
                    result = SFH.generate_stars_mag_composite(composite_masses, composite_mags, mag_names, absmaglim, mag_names[2], T[1//2, 1//2], imf; frac_type=:lum, dist_mod=dmod, rng=rng, mag_lim=T(Inf), mag_lim_name=mag_names[2], binary_model=SFH.NoBinaries())
                    @test length(result) == 2    # Masses and magnitudes
                    @test length(result[1]) == length(result[2]) == length(composite_masses)
                    for i in eachindex(composite_masses, composite_mags) # Isochrone i 
                        @test length(result[1][i]) == length(result[2][i]) # Number of masses equals number of mags
                        @test result[1][i] isa Vector{SVector{1,S}} # Masses
                        @test result[2][i] isa Vector{SVector{2,T}} # Magnitudes
                    end
                    # Test that total magnitude of sampled population is only slightly
                    # brighter than the requested absmaglim
                    # absmag = SFH.flux2mag( sum( sum(map(x->SFH.mag2flux(x[2] - dmod), i)) for i in result[2]) )
                    # @test T(-0.05) <= (abs_mag - absmaglim) <= T(0)
                    # Test total flux instead of magnitude
                    flux_total = sum( sum(map(x->SFH.mag2flux(x[2] - dmod), i)) for i in result[2])
                    @test 1 ≤ flux_total / SFH.mag2flux(absmaglim) ≤ 1.05

                    # Test errors
                    @test_throws ArgumentError SFH.generate_stars_mag_composite(composite_masses, composite_mags, mag_names, absmaglim, "V", T[1//2, 1//2], imf; frac_type=:lum, dist_mod=dmod, rng=rng, mag_lim=T(Inf), mag_lim_name=mag_names[2], binary_model=SFH.NoBinaries()) # Test bad absmag_name
                    @test_throws ArgumentError SFH.generate_stars_mag_composite(composite_masses, composite_mags, mag_names, absmaglim, "V", T[1//2, 1//2], imf; frac_type=:asdf, dist_mod=dmod, rng=rng, mag_lim=T(Inf), mag_lim_name=mag_names[2], binary_model=SFH.NoBinaries()) # Test bad frac_type
                    @test_throws ArgumentError SFH.generate_stars_mag_composite([m_ini, composite_masses...], composite_mags, mag_names, absmaglim, mag_names[2], T[1//2, 1//2], imf; frac_type=:lum, dist_mod=dmod, rng=rng, mag_lim=T(Inf), mag_lim_name=mag_names[2], binary_model=SFH.NoBinaries()) # Test bad array input

                    ################
                    # Test model_cmd
                    errfuncs = [x->T(1//10)*x, x->T(2//10)*x] # Placeholder anonymous functions
                    completefuncs = [x->T(4//10), x->T(4//10)] # Placeholder anonymous functions
                    model_mags = vcat(result[2]...)
                    # Use Vector{SVector} for model_mags
                    cmd = SFH.model_cmd( model_mags, errfuncs, completefuncs; rng=rng)
                    @test typeof(cmd) == typeof(model_mags)
                    @test length(cmd) ≈ length(model_mags) * T(4//25) rtol=T(1//10)
                    # Use Vector{Vector} for model_mags
                    model_mags2 = convert(Vector{Vector{T}}, model_mags)
                    cmd = SFH.model_cmd( model_mags2, errfuncs, completefuncs; rng=rng)
                    @test typeof(cmd) == typeof(model_mags2)
                    @test length(cmd) ≈ length(model_mags2) * T(4//25) rtol=T(1//10)
                    # Try ret_idxs
                    cmd, idxs = SFH.model_cmd( model_mags, errfuncs, completefuncs; rng=rng, ret_idxs=true)
                    @test idxs isa Vector{Int}
                    cmd, idxs = SFH.model_cmd( model_mags2, errfuncs, completefuncs; rng=rng, ret_idxs=true)
                    @test idxs isa Vector{Int}

                    # Test errors
                    # Vector{SVector} mags argument
                    @test_throws ArgumentError SFH.model_cmd( model_mags, [one, errfuncs...], completefuncs; rng=rng) 
                    # Vector{Vector} mags argument
                    @test_throws ArgumentError SFH.model_cmd( model_mags2, [one, errfuncs...], completefuncs; rng=rng)
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
                    @test grad3 ≈ [0, -1, -1] rtol=rtols[i]
                    # zeros in `data`, flattened inputs
                    data4 = vec(data3)
                    C4 = models2 * coeffs
                    grad4 = Vector{T}(undef,3)
                    SFH.∇loglikelihood!( grad4, C4, models2, data4 )
                    @test grad4 ≈ [0, -1, -1] rtol=rtols[i]
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
        
        # Benchmarking
        # let x=[1.0], M=[Float64[0 0 0; 0 0 0; 1 1 1]], N=Int64[0 0 0; 0 0 0; 3 3 3], C=zeros(3,3), G=[1.0]
        #     @btime SFH.loglikelihood($M[1], $N)
        #     @btime SFH.∇loglikelihood($M[1], $M[1], $N) # @btime SFH.∇loglikelihood($x, $M, $N)
        #     @btime SFH.fg($M[1], $M[1], $N)
        #     @btime SFH.fg!($true, $G, $x, $M, $N, $C)
        # end
        
        @safetestset "Template Kernels" include("templates/kernel_test.jl")
        @safetestset "Template Construction" include("templates/template_test.jl")

        @testset verbose=true "Solving + Sampling" begin
            @safetestset "Basic Linear Combinations" include("fitting/basic_linear_combinations.jl")
            @safetestset "Age-Metallicity Relations" include("fitting/amr_test.jl")
            @safetestset "Mass-Metallicity Relations" include("fitting/mzr_test.jl")
            @safetestset "Fixed AMR" include("fitting/fixed_amr.jl")
        end
    end


    @testset "utilities" begin
        # Artifical star tests use extensions, requires Julia >= 1.9
        if VERSION >= v"1.9"
            # We are intentionally causing some warnings, they are not significant
            with_logger(ConsoleLogger(Error)) do
                @safetestset "process_ASTs" include("utilities/process_ASTs_test.jl")
            end
        end

        @testset "mdf_amr" begin
            mdf_result1 = SFH.mdf_amr([1.0,2.0,3.0,4.0],[1.0,2.0,1.0,2.0],[-2.0,-2.0,-1.0,-1.0])
            @test mdf_result1[1] ≈ [-2.0, -1.0]
            @test mdf_result1[2] ≈ [0.3, 0.7]
            # Test mdf_x is always returned in sorted order
            mdf_result2 = SFH.mdf_amr(reverse([1.0,2.0,3.0,4.0]),[1.0,2.0,1.0,2.0],reverse([-2.0,-2.0,-1.0,-1.0]))
            @test all(mdf_result1 .== mdf_result2)
        end

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
                @test SFH.L_from_MV(T(483//100)) === (T(1))
                @test SFH.MV_from_L(T(1)) === (T(483//100))
                @test SFH.Y_from_Z(convert(T,1e-3), 0.2485) ≈ 0.2502800000845455 rtol=rtols[i] # Return type not guaranteed
                @test SFH.X_from_Z(convert(T,1e-3)) ≈ 0.748719999867957 rtol=rtols[i] # Return type not guaranteed
                @test SFH.X_from_Z(convert(T,1e-3), convert(T,0.25)) ≈ 0.74722 rtol=rtols[i] # Return type not guaranteed
                @test SFH.X_from_Z(convert(T,1e-3), convert(T,0.25), convert(T,1.78)) ≈ 0.74722 rtol=rtols[i] # Return type not guaranteed
                @test SFH.MH_from_Z(convert(T,1e-3), convert(T,0.01524)) ≈ -1.206576807011171 rtol=rtols[i] # Return type not guaranteed
                @test SFH.Z_from_MH(convert(T,-2), convert(T,0.01524); Y_p=convert(T,0.2485)) ≈ 0.00016140871730361718 rtol=rtols[i] # Return type not guaranteed
                # These two functions should be inverses; test that they are
                @test SFH.MH_from_Z(SFH.Z_from_MH(convert(T,-2), convert(T,0.01524); Y_p=convert(T,0.2485), γ=convert(T,1.78)), convert(T,0.01524); Y_p=convert(T,0.2485)) ≈ -2 rtol=rtols[i] # Return type not guaranteed
                # Due to a previous bug, Z_from_MH passed the above test but diverged at higher Z. Test at positive [M/H] here.
                @test SFH.MH_from_Z(SFH.Z_from_MH(convert(T,1.0), convert(T,0.01524); Y_p=convert(T,0.2485), γ=convert(T,1.78)), convert(T,0.01524); Y_p=convert(T,0.2485)) ≈ 1.0 rtol=rtols[i] # Return type not guaranteed

                @test SFH.dMH_dZ(convert(T,1e-3), convert(T,0.01524); Y_p = convert(T,0.2485), γ = convert(T,1.78)) ≈ 435.9070188458886 rtol=rtols[i] # Return type not guaranteed
                @test SFH.Martin2016_complete(T[20.0, 1.0, 25.0, 1.0]...) ≈ big"0.9933071490757151444406380196186748196062559910927034697307877569401159160854199" rtol=rtols[i]
                @test SFH.Martin2016_complete(T[20.0, 1.0, 25.0, 1.0]...) isa T
                @test SFH.exp_photerr(T[20.0, 1.05, 10.0, 32.0, 0.01]...) ≈ big"0.01286605230281143891186877135084309862554426640053421106995766903206843498217022"
                @test SFH.exp_photerr(T[20.0, 1.05, 10.0, 32.0, 0.01]...) isa T
            end
        end
    end
end
