import StarFormationHistories: GaussianPSFAsymmetric, GaussianPSFCovariant, addstar!
import StatsBase: fit, Histogram
using Test

# Tests for kernels where mags are {B,V,R} for three different cases of Hess
# diagram configuration.
# Y=B, X=B-V
# Y=V, X=B-V
# Y=R, X=B-V

const npoints = 100_000
const mags = ("B", "V", "R")
const σ = (B=0.1, V=0.1, R=0.1) # Magnitude errors
const centers = (B=20.0, V=19.0, R=18.0)

# cov_mult = 1 for y=V and x=B-V, -1 for y=B and x=B-V, 0 for y=R and x=B-V
@testset "B-V color" begin
    # For mags {B,V}, test case y=B, x=B-V
    yy = randn(npoints) .* σ.B .+ centers.B
    xx = yy .- (randn(npoints) .* σ.V .+ centers.V)
    x_cen = centers.B - centers.V
    xbins = (x_cen-5*σ.V):σ.V/10:(x_cen+5*σ.V)
    ybins = (centers.B-5*σ.B):σ.B/10:(centers.B+5*σ.B)
    h = Histogram((xbins, ybins), zeros(length(xbins)-1, length(ybins)-1), :left, false)
    p = GaussianPSFCovariant(x_cen, centers.B, σ.V, σ.B, -1.0, 1.0, 0.0)
    addstar!(h, p)
    @test any(!=(0), h.weights)
end
