import StarFormationHistories as SFH
import QuadGK: quadgk
import StaticArrays: SMatrix
using Test

# Test numerical accuracy of the Gauss-Legendre integration for general 2D Gaussian

@testset "Gaussian2D" begin
    let scale=1.0, xx=1.4*scale, yy=1.7*scale, x0=0.75*scale, y0=1.2*scale, Σ=SMatrix{2,2}( 3.70899, 1.64674, 1.64674, 1.64653 ) .* scale, A=1.0
        # Single pixel, equal width case (integral from (x-0.5 -> x+0.5) and (y-0.5 -> y+0.5))
        qgk_result = quadgk(y -> quadgk( x -> SFH.gauss2D(x, y, x0, y0, Σ, A), xx-0.5, xx+0.5)[1], yy-0.5, yy+0.5)[1]
        halfpix_result = SFH.gauss2d_integral_halfpix(xx,yy,x0,y0,Σ,A,1.0)
        # Test that you get same thing using Gaussian2D type interface
        gaussian2d_result = SFH.evaluate(SFH.Gaussian2D(x0,y0,Σ,A,1.0),xx,yy)
        @test halfpix_result == gaussian2d_result
        # println( (qgk_result, halfpix_result) )
        relerr = abs(qgk_result - halfpix_result) / qgk_result
        # println(relerr)
        @test relerr < 1e-5

        # Single pixel, different bin size in x and y
        # (integral from (x-0.5 -> x+0.5) and (y-1 -> y+1))
        qgk_result = quadgk(y -> quadgk( x -> SFH.gauss2D(x, y, x0, y0, Σ, A), xx-0.5, xx+0.5)[1], yy-1, yy+1)[1]
        halfpix_result = SFH.gauss2d_integral_halfpix(xx,yy,x0,y0,Σ,A,2.0)
        relerr = abs(qgk_result - halfpix_result) / qgk_result
        # println(relerr)
        @test relerr < 1e-3
    end
end

# @benchmark SFH.gauss2d_integral_halfpix($5.2,$3.4,$0.75,$1.2,$(SMatrix{2,2}(3.7, 1.6, 1.6, 1.6)),$1.0,$1.0)
# @benchmark SFH.gauss2d_integral_halfpix($5.2,$3.4,$0.75,$1.2,$(SMatrix{2,2}(3.7, 1.6, 1.6, 1.6)),$1.0,$2.0)
