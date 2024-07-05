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
@testset "y=B, x=B-V" begin
    x_cen = centers.B - centers.V
    xbins = (x_cen-5*σ.V):σ.V/10:(x_cen+5*σ.V)
    ybins = (centers.B-5*σ.B):σ.B/10:(centers.B+5*σ.B)
    h = Histogram((xbins, ybins), zeros(length(xbins)-1, length(ybins)-1), :left, false)
    p = GaussianPSFCovariant(x_cen, centers.B, σ.V, σ.B, -1.0, 1.0, 0.0)
    addstar!(h, p)
    @test any(!=(0), h.weights)
end
@testset "y=V, x=B-V" begin
    x_cen = centers.B - centers.V
    xbins = (x_cen-5*σ.V):σ.V/10:(x_cen+5*σ.V)
    ybins = (centers.V-5*σ.V):σ.V/10:(centers.V+5*σ.V)
    h = Histogram((xbins, ybins), zeros(length(xbins)-1, length(ybins)-1), :left, false)
    p = GaussianPSFCovariant(x_cen, centers.V, σ.B, σ.V, 1.0, 1.0, 0.0)
    addstar!(h, p)
    @test any(!=(0), h.weights)
end
@testset "y=R, x=B-V" begin
    x_cen = centers.B - centers.V
    xbins = (x_cen-5*σ.V):σ.V/10:(x_cen+5*σ.V)
    ybins = (centers.R-5*σ.R):σ.R/10:(centers.R+5*σ.R)
    h = Histogram((xbins, ybins), zeros(length(xbins)-1, length(ybins)-1), :left, false)
    p = GaussianPSFAsymmetric(x_cen, centers.R, sqrt(σ.B^2 + σ.V^2), σ.R, 1.0, 0.0)
    addstar!(h, p)
    @test any(!=(0), h.weights)
end
