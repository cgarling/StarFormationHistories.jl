import StarFormationHistories: GaussianKernel, addstar!
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
    p = GaussianKernel(x_cen, centers.B, σ.V, σ.B, -1.0, 1.0)
    addstar!(h, p)
    @test any(!=(0), h.weights)
end
@testset "y=V, x=B-V" begin
    x_cen = centers.B - centers.V
    xbins = (x_cen-5*σ.V):σ.V/10:(x_cen+5*σ.V)
    ybins = (centers.V-5*σ.V):σ.V/10:(centers.V+5*σ.V)
    h = Histogram((xbins, ybins), zeros(length(xbins)-1, length(ybins)-1), :left, false)
    p = GaussianKernel(x_cen, centers.V, σ.B, σ.V, 1.0, 1.0)
    addstar!(h, p)
    @test any(!=(0), h.weights)
end
@testset "y=R, x=B-V" begin
    x_cen = centers.B - centers.V
    xbins = (x_cen-5*σ.V):σ.V/10:(x_cen+5*σ.V)
    ybins = (centers.R-5*σ.R):σ.R/10:(centers.R+5*σ.R)
    h = Histogram((xbins, ybins), zeros(length(xbins)-1, length(ybins)-1), :left, false)
    p = GaussianKernel(x_cen, centers.R, sqrt(σ.B^2 + σ.V^2), σ.R, 0.0, 1.0)
    addstar!(h, p)
    @test any(!=(0), h.weights)
end

# Reference for the kernel: numerical integral over y and exact integral over color for each pixel
using Distributions: Normal, cdf, pdf
using QuadGK: gauss
const gl_nodes, gl_weights = gauss(200) # Gauss-Legendre rule on (-1, 1)
covariant_pixel(xlo, xhi, ylo, yhi, x0, y0, σx, σy, c) = sum(zip(gl_nodes, gl_weights)) do (t, w)
    y = (ylo + yhi) / 2 + t * (yhi - ylo) / 2
    w * (yhi - ylo) / 2 * pdf(Normal(y0, σy), y) * (cdf(Normal(x0 - c * (y - y0), σx), xhi) - cdf(Normal(x0 - c * (y - y0), σx), xlo))
end
@testset "GaussianKernel accuracy" begin
    # Narrow color bins and tall magnitude bins as in typical Hess diagrams, including σy much smaller than the bin height
    xe, ye = range(0.0, 1.0; step=0.02), range(19.0, 21.0; step=0.1)
    newhist() = Histogram((xe, ye), zeros(length(xe) - 1, length(ye) - 1), :left, false)
    for (σx, σy, c) in ((0.02, 0.01, 1), (0.02, 0.02, -1), (0.03, 0.05, 1), (0.05, 0.2, -1), (0.02, 0.01, 0), (0.05, 0.2, 0)), y0 in (20.0, 20.037, 20.081)
        h = newhist()
        addstar!(h, GaussianKernel(0.5, y0, σx, σy, c))
        # Reference only for rows within 6σy of the kernel, where it is nonzero to within the test tolerance
        ref = [abs((ye[j] + ye[j+1]) / 2 - y0) < 6σy + step(ye) ? covariant_pixel(xe[i], xe[i+1], ye[j], ye[j+1], 0.5, y0, σx, σy, c) : 0.0
               for i in 1:length(xe)-1, j in 1:length(ye)-1]
        # Weight is conserved, apart from what falls outside the image in both the kernel and the reference
        @test sum(h.weights) ≈ sum(ref) atol=2e-4
        @test maximum(abs, h.weights .- ref) < 1e-3
    end
    # A kernel overlapping only the first row of the image still contributes
    h = newhist()
    addstar!(h, GaussianKernel(0.5, 18.95, 0.02, 0.02, 1))
    @test sum(h.weights) ≈ cdf(Normal(18.95, 0.02), 19.1) - cdf(Normal(18.95, 0.02), 19.0) rtol=1e-3
    # Without covariance the kernel is separable and each pixel is the exact product of the x and y integrals
    h = newhist()
    addstar!(h, GaussianKernel(0.5013, 20.037, 0.03, 0.05, 0, 2.0))
    exact = [2 * (cdf(Normal(0.5013, 0.03), xe[i+1]) - cdf(Normal(0.5013, 0.03), xe[i])) * (cdf(Normal(20.037, 0.05), ye[j+1]) - cdf(Normal(20.037, 0.05), ye[j]))
             for i in 1:length(xe)-1, j in 1:length(ye)-1]
    within = [abs((xe[i] + xe[i+1]) / 2 - 0.5013) < 3 * 0.03 && abs((ye[j] + ye[j+1]) / 2 - 20.037) < 3 * 0.05 for i in 1:length(xe)-1, j in 1:length(ye)-1]
    @test h.weights[within] ≈ exact[within] rtol=1e-12
    @test_throws ArgumentError GaussianKernel(0.5, 20.0, 0.02, 0.02, 0.5)
end
