import StarFormationHistories as SFH
using Test

const float_types = (Float32, Float64) # Float types to test most functions with
const float_type_labels = ("Float32", "Float64") # String labels for the above float_types
const rtols = (1e-3, 1e-7) # Relative tolerance levels to use for the above float types

@testset "Utilities" begin
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
