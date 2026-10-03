import StarFormationHistories as SFH
using InitialMassFunctions: Salpeter1955, Kroupa2001
using Distributions: Uniform, pdf
using StableRNGs: StableRNG
using StaticArrays: SVector
using QuadGK: quadgk
using Test

const seedval = 58392 # Seed to use when instantiating new StableRNG objects
const float_types = (Float32, Float64) # Float types to test most functions with
const float_type_labels = ("Float32", "Float64") # String labels for the above float_types
const rtols = (1e-3, 1e-7) # Relative tolerance levels to use for the above float types

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
                # Deduplication does not mutate the caller's vector
                dup = T[1,2,2,3]
                result = SFH.sort_ingested(dup, SFH.ingest_mags(dup, [T[1,2,3,4], T[1,2,3,4]]))
                @test allunique(result[1]) && dup == T[1,2,2,3]
            end
        end
    end
    @testset "mass_limits" begin
        for i in eachindex(float_types, float_type_labels)
            label = float_type_labels[i]
            @testset "$label" begin
                T = float_types[i]
                m_ini = T[0.05, 0.1, 0.2, 0.3]
                mags = [T[4], T[3], T[2], T[1]]
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
        rng = StableRNG(seedval)
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
                f606w_mags = T[12.73830341586291, 10.017833567325411, 9.041398536300997, 8.331313172834694, 7.3892413746605765, 6.227971374447669, 4.93799980882831, -3.026]
                f814w_mags = T[11.137576129331675, 8.951605387408511, 8.064408199510426, 7.424658866406447, 6.6025729402403, 5.584714238650148, 4.448999828801471, -4.107]
                mags = [f606w_mags, f814w_mags]
                mag_names = ["F606W", "F814W"]
                # Figure out the percentage of total mass represented
                # by stars between miniumum(m_ini) and maximum(m_ini).
                # See the notes in the `generate_stars_mass` for more details.
                mass_frac = (quadgk(x -> x * pdf(imf, x), minimum(m_ini), maximum(m_ini))[1] / quadgk(x -> x * pdf(imf,x), minimum(imf), maximum(imf))[1]) # maximum(imf))[1] )

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
                # A finite mag_lim keeps exactly the systems whose combined light is brighter than mag_lim
                for bm in (SFH.NoBinaries(), SFH.RandomBinaryPairs(T(4//10)), SFH.BinaryMassRatio(T(4//10), Uniform(T(1//2), T(1))))
                    lim_res = SFH.generate_stars_mass(m_ini, mags, mag_names, total_mass, imf; dist_mod=dmod, rng=StableRNG(seedval), mag_lim=T(30), mag_lim_name=mag_names[2], binary_model=bm)
                    inf_res = SFH.generate_stars_mass(m_ini, mags, mag_names, total_mass, imf; dist_mod=dmod, rng=StableRNG(seedval), mag_lim=T(Inf), mag_lim_name=mag_names[2], binary_model=bm)
                    keep = findall(x -> x[2] < T(30), inf_res[2])
                    @test lim_res == (inf_res[1][keep], inf_res[2][keep])
                end

                ##################################
                ####### Testing generate_stars_mag
                absmaglim = T(-8)
                # Test with NoBinaries() model
                result = SFH.generate_stars_mag(m_ini, mags, mag_names, absmaglim, mag_names[2], imf; dist_mod=dmod, rng=rng, mag_lim=T(Inf), mag_lim_name=mag_names[2], binary_model=SFH.NoBinaries())
                @test result[1] isa Vector{SVector{1,S}}
                @test result[2] isa Vector{SVector{2,T}}
                @test length(result[1]) == length(result[2])
                # Test that total magnitude of sampled population is close to the requested absmaglim
                apparent_mag = SFH.flux2mag(sum(map(x->SFH.mag2flux(x[2]), result[2])))
                abs_mag = apparent_mag - dmod
                @test abs(abs_mag - absmaglim) <= T(0.05)

                # Test with RandomBinaryPairs() model
                result = SFH.generate_stars_mag(m_ini, mags, mag_names, absmaglim, mag_names[2], imf; dist_mod=dmod, rng=rng, mag_lim=T(Inf), mag_lim_name=mag_names[2], binary_model=SFH.RandomBinaryPairs(T(4//10)))
                @test result[1] isa Vector{SVector{2,S}}
                @test result[2] isa Vector{SVector{2,T}}
                @test length(result[1]) == length(result[2])
                # Test that total magnitude of sampled population is close to the requested absmaglim
                apparent_mag = SFH.flux2mag(sum(map(x->SFH.mag2flux(x[2]), result[2])))
                abs_mag = apparent_mag - dmod
                @test abs(abs_mag - absmaglim) <= T(0.05)

                # Test with BinaryMassRatio() model
                result = SFH.generate_stars_mag(m_ini, mags, mag_names, absmaglim, mag_names[2], imf; dist_mod=dmod, rng=rng, mag_lim=T(Inf), mag_lim_name=mag_names[2], binary_model=SFH.BinaryMassRatio(T(4//10), Uniform(T(1//10),T(1))))
                @test result[1] isa Vector{SVector{2,S}}
                @test result[2] isa Vector{SVector{2,T}}
                @test length(result[1]) == length(result[2])
                # Test that total magnitude of sampled population is close to the requested absmaglim
                apparent_mag = SFH.flux2mag(sum(map(x->SFH.mag2flux(x[2]), result[2])))
                abs_mag = apparent_mag - dmod
                @test abs(abs_mag - absmaglim) <= T(0.05)

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
                # Results are reproducible for a given rng state even though populations are sampled in parallel
                gen_seeded() = SFH.generate_stars_mass_composite(composite_masses, composite_mags, mag_names, total_mass, T[1//2, 1//2], imf; dist_mod=dmod, rng=StableRNG(seedval), mag_lim=T(Inf), mag_lim_name=mag_names[2], binary_model=SFH.NoBinaries())
                @test gen_seeded() == gen_seeded()

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
                # Single-star systems give a scalar `mini` equal to the sampled mass
                @test keys(first(am_result))[1:3] == (:mini, :F606W, :F814W)
                @test [a.mini for a in am_result] == [m[1] for m in reduce(vcat, result[1])]
                # A leading SSP with no sampled stars is handled
                @test SFH.add_metadata(([similar(result[1][1], 0), result[1][2]], [similar(result[2][1], 0), result[2][2]]), (:F606W, :F814W)) isa Vector{<:NamedTuple}
                # A catalog with no sampled stars gives an empty result
                @test isempty(SFH.add_metadata(([similar(result[1][1], 0)], [similar(result[2][1], 0)]), (:F606W, :F814W)))
                # Binary systems are split into primary `mini` and companion `mini2`
                bresult = SFH.generate_stars_mass_composite(composite_masses, composite_mags, mag_names, total_mass, T[1//2, 1//2], imf; dist_mod=dmod, rng=rng, mag_lim=T(Inf), mag_lim_name=mag_names[2], binary_model=SFH.RandomBinaryPairs(T(4//10)))
                b_am = SFH.add_metadata(bresult, (:F606W, :F814W))
                @test keys(first(b_am)) == (:mini, :mini2, :F606W, :F814W)
                @test [(a.mini, a.mini2) for a in b_am] == [(m[1], m[2]) for m in reduce(vcat, bresult[1])]
                @test all(a.mini >= a.mini2 for a in b_am)
                @test any(a.mini2 == 0 for a in b_am) && any(a.mini2 > 0 for a in b_am)
                @test_throws ArgumentError SFH.add_metadata((result[1][1:1], result[2]), (:F606W, :F814W); logAge=[6.6,6.7], MH=[1.0, 1.0])

                @test_throws ArgumentError SFH.add_metadata(result, (:F606W,))
                @test_throws ArgumentError SFH.add_metadata(result, (:F606W,:F814W); logAge=[6.6])
                @test_throws ArgumentError SFH.add_metadata((result[1][1], result[1][2]), 
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
                # Test that total magnitude of sampled population is close to the requested absmaglim
                # absmag = SFH.flux2mag( sum( sum(map(x->SFH.mag2flux(x[2] - dmod), i)) for i in result[2]) )
                # @test T(-0.05) <= (abs_mag - absmaglim) <= T(0)
                # Test total flux instead of magnitude
                flux_total = sum( sum(map(x->SFH.mag2flux(x[2] - dmod), i)) for i in result[2])
                @test flux_total / SFH.mag2flux(absmaglim) ≈ 1 rtol=5e-2
                # Results are reproducible for a given rng state even though populations are sampled in parallel
                gen_mag_seeded() = SFH.generate_stars_mag_composite(composite_masses, composite_mags, mag_names, absmaglim, mag_names[2], T[1//2, 1//2], imf; frac_type=:lum, dist_mod=dmod, rng=StableRNG(seedval), mag_lim=T(Inf), mag_lim_name=mag_names[2], binary_model=SFH.NoBinaries())
                @test gen_mag_seeded() == gen_mag_seeded()

                # Test errors
                @test_throws ArgumentError SFH.generate_stars_mag_composite(composite_masses, composite_mags, mag_names, absmaglim, "V", T[1//2, 1//2], imf; frac_type=:lum, dist_mod=dmod, rng=rng, mag_lim=T(Inf), mag_lim_name=mag_names[2], binary_model=SFH.NoBinaries()) # Test bad absmag_name
                @test_throws ArgumentError SFH.generate_stars_mag_composite(composite_masses, composite_mags, mag_names, absmaglim, "V", T[1//2, 1//2], imf; frac_type=:asdf, dist_mod=dmod, rng=rng, mag_lim=T(Inf), mag_lim_name=mag_names[2], binary_model=SFH.NoBinaries()) # Test bad frac_type
                @test_throws ArgumentError SFH.generate_stars_mag_composite([m_ini, composite_masses...], composite_mags, mag_names, absmaglim, mag_names[2], T[1//2, 1//2], imf; frac_type=:lum, dist_mod=dmod, rng=rng, mag_lim=T(Inf), mag_lim_name=mag_names[2], binary_model=SFH.NoBinaries()) # Test bad array input

                # Test luminosity_per_mass; constant magnitudes on a fine grid give an analytic result
                fine_mini = range(T(1//10), T(4//5); length=10_000)
                lpm = SFH.luminosity_per_mass(fine_mini, [fill(T(5), length(fine_mini)), fill(T(6), length(fine_mini))], imf)
                @test lpm isa SVector{2}
                @test lpm ≈ SFH.mag2flux.(T[5, 6]) .* (SFH.cdf(imf, T(4//5)) - SFH.cdf(imf, T(1//10))) ./ SFH.mean(imf) rtol=1e-4
                # Test mass_from_absmag against the analytic result for two constant-magnitude populations
                fine_mags = [[fill(T(5), length(fine_mini)), fill(T(6), length(fine_mini))], [fill(T(6), length(fine_mini)), fill(T(7), length(fine_mini))]]
                lpm2 = SFH.luminosity_per_mass(fine_mini, fine_mags[2], imf)
                mfa = SFH.mass_from_absmag([fine_mini, fine_mini], fine_mags, mag_names, absmaglim, mag_names[1], T[1, 3], imf)
                @test mfa ≈ SFH.mag2flux(absmaglim) / (lpm[1] / 4 + 3 * lpm2[1] / 4)
                @test SFH.mass_from_absmag([fine_mini, fine_mini], fine_mags, mag_names, absmaglim, mag_names[1], T[1//4, 3//4], imf) ≈ mfa # massfrac need not be normalized
                @test_throws ArgumentError SFH.mass_from_absmag([fine_mini, fine_mini], fine_mags, mag_names, absmaglim, "V", T[1, 3], imf)
                # A callable IMF with an explicit mean_mass gives the same result
                @test SFH.luminosity_per_mass(fine_mini, [fill(T(5), length(fine_mini)), fill(T(6), length(fine_mini))], x -> pdf(imf, x); mean_mass=SFH.mean(imf)) == lpm
                # Test generate_stars_mag_composite with frac_type=:mass is identical to frac_type=:lum
                # with mass fractions converted to luminosity fractions
                mfracs = T[1//4, 3//4]
                lfracs = mfracs .* [SFH.luminosity_per_mass(composite_masses[i], composite_mags[i], imf)[2] for i in eachindex(composite_masses)]
                r_mass = SFH.generate_stars_mag_composite(composite_masses, composite_mags, mag_names, absmaglim, mag_names[2], mfracs, imf; frac_type=:mass, dist_mod=dmod, rng=StableRNG(seedval), binary_model=SFH.NoBinaries())
                r_lum = SFH.generate_stars_mag_composite(composite_masses, composite_mags, mag_names, absmaglim, mag_names[2], lfracs, imf; frac_type=:lum, dist_mod=dmod, rng=StableRNG(seedval), binary_model=SFH.NoBinaries())
                @test r_mass == r_lum
                @test_throws ArgumentError SFH.generate_stars_mag_composite(composite_masses, composite_mags, mag_names, absmaglim, "V", mfracs, imf; frac_type=:mass, dist_mod=dmod, rng=rng, binary_model=SFH.NoBinaries()) # Test bad absmag_name

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

                # Joint models that are products of the per-filter models give the same results
                biasfuncs = [x->T(1//100)*x, x->T(-1//100)*x]
                jerr(a, b) = (errfuncs[1](a), errfuncs[2](b))
                jcomp(a, b) = completefuncs[1](a) * completefuncs[2](b)
                jbias(a, b) = (biasfuncs[1](a), biasfuncs[2](b))
                for mm in (model_mags, model_mags2)
                    ref = SFH.model_cmd(mm, errfuncs, completefuncs, biasfuncs; rng=StableRNG(seedval), ret_idxs=true)
                    @test SFH.model_cmd(mm, jerr, jcomp, jbias; rng=StableRNG(seedval), ret_idxs=true) == ref
                    @test SFH.model_cmd(mm, errfuncs, jcomp, biasfuncs; rng=StableRNG(seedval), ret_idxs=true) == ref
                end
                # A joint completeness that depends on color keeps only the selected stars
                mean_color = sum(m[1] - m[2] for m in model_mags) / length(model_mags)
                jcut(a, b) = a - b < mean_color
                _, idxs = SFH.model_cmd(model_mags, errfuncs, jcut; rng=rng, ret_idxs=true)
                @test idxs == findall(m -> jcut(m...), model_mags)

                ################
                # Test surviving_fraction and friends
                s = SFH.surviving_fraction([m_ini, m_ini[1:end-1]], T[1//2, 1//2], imf)
                @test s isa T
                @test s ≈ 0.9337494000577464 rtol=rtols[i]
                r = SFH.recycling_fraction([m_ini, m_ini[1:end-1]], T[1//2, 1//2], imf)
                @test r isa T
                @test r ≈ 1 - s rtol=rtols[i]
                s = SFH.surviving_mass_fraction([m_ini, m_ini[1:end-1]], T[1//2, 1//2], imf)
                @test s isa T
                @test s ≈ 0.5937806939236231 rtol=rtols[i]
                r = SFH.recycling_mass_fraction([m_ini, m_ini[1:end-1]], T[1//2, 1//2], imf)
                @test r isa T
                @test r ≈ 1 - s rtol=rtols[i]
            end
        end
    end
end
